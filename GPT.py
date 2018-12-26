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
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, n_cls, lambda_=0.5, drop_rate=0.1):
        self.model_dim = model_dim
        self.max_len = max_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.drop_rate = drop_rate
        self.attentions = []    # for visualization

        # inputs
        self.training = tf.placeholder(tf.bool, None)               # used in unsupervised & supervised learning
        self.tfx = tf.placeholder(tf.int32, [None, max_len])      # used in unsupervised & supervised learning
        self.tfcls = tf.placeholder(tf.int32, [None, ])  # target in supervised learning

        self.target_seq = tf.concat(
            (self.tfx[:, 1:], tf.zeros((tf.shape(self.tfx)[0], 1), dtype=tf.int32)), axis=1) # [n, step] add pad at end

        word_embeddings = tf.Variable(tf.random_normal((n_vocab, model_dim), 0., 0.01))      # [n_vocab, dim]
        position_embeddings = tf.Variable(tf.random_normal((1, max_len, model_dim)))          # [1, step, dim]
        x_embedded = tf.nn.embedding_lookup(word_embeddings, self.tfx) + position_embeddings  # [n, step, dim]

        decoded_z = self._build_decoder(x_embedded)

        # output heads
        self.seq_logits = tf.layers.dense(decoded_z, n_vocab)                # [n, step, n_vocab]
        reshaped_decoded_z = tf.reshape(decoded_z, [-1, max_len*model_dim])
        self.cls_logits = tf.layers.dense(reshaped_decoded_z, n_cls)                    # [n, n_cls]

        # losses
        self.unsupervised_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.target_seq, logits=self.seq_logits))
        cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.tfcls, logits=self.cls_logits))
        self.semi_supervised_loss = cls_loss + lambda_ * self.unsupervised_loss

        # train
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):   # for batch norm
            self.unsupervised_train = tf.train.AdamOptimizer(0.001).minimize(self.unsupervised_loss)
            self.supervised_train = tf.train.AdamOptimizer(0.001).minimize(self.semi_supervised_loss)

        self.sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': 1}))
        self.sess.run(tf.global_variables_initializer())

    def _build_decoder(self, z):
        mask = self.output_mask(self.tfx)
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

    @staticmethod
    def pad_mask(seqs):
        mask = tf.where(tf.equal(seqs, 0), tf.zeros_like(seqs), tf.ones_like(seqs))                # 0 idx is padding
        return tf.cast(tf.expand_dims(mask, axis=1) * tf.expand_dims(mask, axis=2), dtype=tf.bool) # [n, step+1, step+1]

    def output_mask(self, seqs):
        pad_mask = self.pad_mask(seqs)
        mask = tf.constant(~np.triu(np.ones((self.max_len, self.max_len), dtype=np.bool), 1))
        mask = tf.tile(tf.expand_dims(mask, axis=0), [tf.shape(seqs)[0], 1, 1])     # [n, step+1, step+1]
        return tf.where(mask, pad_mask, tf.zeros_like(pad_mask))


# get and process data
utils.download_mrpc(save_dir="./MRPC/", proxy="http://web-proxy.oa.com:8080")
data, vocab, v2i, i2v, max_len = utils.process_mrpc("./MRPC")
print("data include: ", data.keys())
print("Samples ids: ", data["train"]["s1id"][0], "\nSamples words: ", data["train"]["s1"][0])
max_len = max_len*2 + 1
unsupervised_x = data["train"]["s1id"] + data["train"]["s2id"]
supervised_x = [data["train"]["s1id"][i] + [v2i["<SEP>"], ] + data["train"]["s2id"][i] for i in range(len(data["train"]["s1id"]))]

model = GPT(model_dim=MODEL_DIM, max_len=max_len, n_layer=N_LAYER, n_head=N_HEAD,
            n_vocab=len(v2i), n_cls=N_CLS)

# unsupervised training
t0 = time.time()
for t in range(7000):
    bi = np.random.randint(0, len(unsupervised_x), size=64)
    bx = utils.pad_zero([unsupervised_x[i] for i in bi], max_len=max_len)     # add one more zero padding for fake task
    _, loss_ = model.sess.run([model.unsupervised_train, model.unsupervised_loss], {model.tfx: bx, model.training: True})
    if t % 50 == 0:
        seq_logits_ = model.sess.run(model.seq_logits, {model.tfx: bx, model.training: False})
        target = []
        for i in bx[0]:
            target.append(i2v[i])
            if i == v2i["<EOS>"]: break
        target = " ".join(target)
        res = []
        for i in np.argmax(seq_logits_[0], axis=1):
            res.append(i2v[i])
            if i == v2i["<EOS>"]: break
        res = " ".join(res)
        t1 = time.time()
        print(
            "Unsupervised step: ", t,
            "| time: %.2f" % (t1-t0),
            "| loss: %.3f" % loss_,
            "\n| target: ", target,
            "\n| predict: ", res,
        )
        t0 = t1

# supervised learning
supervised_label = data["train"]["is_same"]
for t in range(3000):
    bi = np.random.randint(0, len(supervised_x), size=64)
    bx = utils.pad_zero([supervised_x[i] for i in bi], max_len=max_len)
    bx[:, -1] = 1
    _, loss_ = model.sess.run([model.supervised_train, model.semi_supervised_loss],
                              {model.tfx: bx, model.tfcls: supervised_label[bi], model.training: True})
    if t % 50 == 0:
        bl = supervised_label[bi]
        cls_logits_ = model.sess.run(model.cls_logits, {model.tfx: bx, model.training: False})
        correct = cls_logits_.argmax(axis=1) == supervised_label[bi]
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
attentions, cls_logits_, seq_logits_ = model.sess.run([model.attentions, model.cls_logits, model.seq_logits], {model.tfx: bx, model.training: False})
data = {"src": bx, "label": cls_logits_.argmax(axis=1), "seq": seq_logits_.argmax(axis=-1), "attentions": attentions, "i2v": i2v}
import pickle
with open("./visual_helper/GPT_attention_matrix.pkl", "wb") as f:
    pickle.dump(data, f)

