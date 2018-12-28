"""
reference:
https://zhuanlan.zhihu.com/p/49271699
https://jalammar.github.io/illustrated-bert/
"""

import numpy as np
import tensorflow as tf
import utils
import time

MODEL_DIM = 128
N_LAYER = 3
N_HEAD = 3
N_CLS = 2


class GPT:
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, n_cls, lambda_=0.5, drop_rate=0.1, padding_idx=0):
        self.model_dim = model_dim
        self.max_len = max_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.drop_rate = drop_rate
        self.padding_idx = padding_idx
        self.attentions = []    # for visualization

        # inputs
        self.training = tf.placeholder(tf.bool, None)               # used in unsupervised & supervised learning
        self.tfseq = tf.placeholder(tf.int32, [None, max_len])      # used in unsupervised & supervised learning
        self.tfcls = tf.placeholder(tf.int32, [None, ])  # target in supervised learning

        word_embeddings = tf.Variable(tf.random_normal((n_vocab, model_dim), 0., 0.01))      # [n_vocab, dim]
        position_embeddings = tf.Variable(tf.random_normal((1, max_len, model_dim)))          # [1, step, dim]
        x_embedded = tf.nn.embedding_lookup(word_embeddings, self.tfseq) + position_embeddings  # [n, step, dim]

        mask = self.make_mask(tf.concat((self.tfseq[:, :-1], tf.ones_like(self.tfseq[:, -1:])), axis=1))
        z = self.build_net(x_embedded, mask=mask)

        # output heads
        self.seq_logits = tf.layers.dense(z[:, :-1, :], n_vocab)                # [n, step-1, n_vocab]
        self.cls_logits = tf.layers.dense(z[:, -1, :], n_cls)                   # [n, n_cls]

        # losses
        self.unsupervised_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.tfseq[:, 1:], logits=self.seq_logits))
        cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.tfcls, logits=self.cls_logits))
        self.semi_supervised_loss = cls_loss + lambda_ * self.unsupervised_loss

        # train
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):   # for batch norm
            self.unsupervised_train = tf.train.AdamOptimizer(0.002).minimize(self.unsupervised_loss)
            self.semi_supervised_train = tf.train.AdamOptimizer(0.002).minimize(self.semi_supervised_loss)

        self.sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': 1}))
        self.sess.run(tf.global_variables_initializer())

    def build_net(self, z, mask):
        for n in range(self.n_layer):
            z = self.lnorm(self.multi_head(z, z, z, mask) + z)
            z = self.lnorm(self.position_wise_ffn(z) + z)
        return z

    def scaled_dot_product_attention(self, q, k, v, mask):
        dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
        score = tf.matmul(q, tf.transpose(k, (0, 2, 1))) / tf.sqrt(dk)          # [h*n, q_step, step]
        mask = tf.tile(mask, [self.n_head, 1, 1])                               # repeat for n_head
        score = tf.where(mask, score, tf.fill(tf.shape(score), -np.inf))        # make softmax not select padded value
        attention = tf.nn.softmax(score, axis=-1)                               # [h*n, q_step, step]
        attention = tf.where(tf.is_nan(attention), tf.zeros_like(attention), attention)  # replace nan caused softmax of all -inf
        self.attentions.append(
            tf.transpose(tf.split(attention, self.n_head, axis=0), (1, 0, 2, 3)))  # [n, h, q_step, step]
        attention = tf.layers.dropout(attention, rate=self.drop_rate, training=self.training)
        context = tf.matmul(attention, v)           # [h*n, q_step, step] @ [h*n, step, dv] = [h*n, q_step, dv]
        return context

    def multi_head(self, query, key, value, mask):
        head_dim = self.model_dim // self.n_head
        q = tf.layers.dense(query, self.n_head * head_dim)          # [n, q_step, h_dim*h]
        k = tf.layers.dense(key, self.n_head * head_dim)            # [n, step, h_dim*h]
        v = tf.layers.dense(value, self.n_head * head_dim)          # [n, step, h_dim*h]
        q_ = tf.concat(tf.split(q, self.n_head, axis=2), axis=0)    # [h*n, q_step, h_dim]
        k_ = tf.concat(tf.split(k, self.n_head, axis=2), axis=0)    # [h*n, step, h_dim]
        v_ = tf.concat(tf.split(v, self.n_head, axis=2), axis=0)    # [h*n, step, h_dim]

        context = self.scaled_dot_product_attention(q_, k_, v_, mask)
        o = tf.concat(tf.split(context, self.n_head, axis=0), axis=2)  # [n, q_step, h*h_dim=model_dim]
        o = tf.layers.dense(o, self.model_dim, use_bias=False)
        o = tf.layers.dropout(o, rate=self.drop_rate, training=self.training)
        return o

    def position_wise_ffn(self, x):
        o = tf.layers.conv1d(x, filters=126, kernel_size=1, activation=tf.nn.relu)
        o = tf.layers.conv1d(o, filters=self.model_dim, kernel_size=1)
        o = tf.layers.dropout(o, rate=self.drop_rate, training=self.training)
        return o

    def lnorm(self, x):
        return tf.layers.batch_normalization(x, training=self.training)

    def make_mask(self, seqs):
        """
        output mask will look like this:
        input|  G,1,2,P,P
        target|
        1      [1,0,0,0,0]
        2      [1,1,0,0,0]
        P      [1,1,1,0,0]
        P      [0,0,0,0,0]
        cls    [1,1,1,0,0]     this line for predicting class
        """
        pad_mask = tf.where(tf.equal(seqs, self.padding_idx), tf.zeros_like(seqs), tf.ones_like(seqs))  # 0 idx is padding
        pad_mask = tf.cast(tf.expand_dims(pad_mask, axis=1) * tf.expand_dims(pad_mask, axis=2), dtype=tf.bool)  # [n, step, step]
        np_m = ~np.triu(np.ones((self.max_len, self.max_len), dtype=np.bool), 1)
        np_m[-1, -1] = False
        mask = tf.constant(np_m)
        mask = tf.tile(tf.expand_dims(mask, axis=0), [tf.shape(seqs)[0], 1, 1])     # [n, step, step]
        return tf.where(mask, pad_mask, tf.zeros_like(pad_mask))


# get and process data
v2i, i2v, max_len, unsupervised_x, supervised_x, supervised_label = utils.gpt_mrpc("./MRPC")
model = GPT(model_dim=MODEL_DIM, max_len=max_len, n_layer=N_LAYER, n_head=N_HEAD,
            n_vocab=len(v2i), n_cls=N_CLS)

# unsupervised training
t0 = time.time()
for t in range(5000):
    bi = np.random.randint(0, len(unsupervised_x), size=32)
    bx = unsupervised_x[bi]
    _, loss_ = model.sess.run([model.unsupervised_train, model.unsupervised_loss], {model.tfseq: bx, model.training: True})
    if t % 50 == 0:
        seq_logits_ = model.sess.run(model.seq_logits, {model.tfseq: bx, model.training: False})
        t1 = time.time()
        print(
            "Unsupervised step: ", t,
            "| time: %.2f" % (t1-t0),
            "| loss: %.3f" % loss_,
            "\n| target: ", " ".join([i2v[i] for i in bx[0] if i != v2i["<PAD>"]]),
            "\n| predict: ", " ".join([i2v[i] for i in np.argmax(seq_logits_[0], axis=1) if i != v2i["<PAD>"]]),
        )
        t0 = t1

# supervised learning
for t in range(1000):
    bi = np.random.randint(0, len(supervised_x), size=32)
    bx, by = supervised_x[bi], supervised_label[bi]
    _, loss_ = model.sess.run([model.semi_supervised_train, model.semi_supervised_loss],
                              {model.tfseq: bx, model.tfcls: by, model.training: True})
    if t % 50 == 0:
        bl = supervised_label[bi]
        cls_logits_ = model.sess.run(model.cls_logits, {model.tfseq: bx, model.training: False})
        correct = cls_logits_.argmax(axis=1) == by
        acc = np.sum(correct) / len(correct)
        t1 = time.time()
        print(
            "Semi-supervised step: ", t,
            "| time: %.2f" % (t1 - t0),
            "| loss: %.3f" % loss_,
            "| acc: %.3f" % acc,
        )
        t0 = t1

# save attention matrix for visualization
attentions, cls_logits_, seq_logits_ = model.sess.run([model.attentions, model.cls_logits, model.seq_logits], {model.tfseq: bx, model.training: False})
data = {"src": bx, "label": cls_logits_.argmax(axis=1), "seq": seq_logits_.argmax(axis=-1), "attentions": attentions, "i2v": i2v}
import pickle
with open("./visual_helper/GPT_attention_matrix.pkl", "wb") as f:
    pickle.dump(data, f)

