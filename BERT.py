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
N_HEAD = 4
N_CLS = 2
MAX_SEG = 3   # sentence 1, sentence 2, padding
WORD_REPLACE_RATE = 0.15


class BERT:
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, n_cls, max_seg=3, n_task=2, drop_rate=0.1, padding_idx=0):
        self.model_dim = model_dim
        self.max_len = max_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.drop_rate = drop_rate
        self.max_seg = max_seg      # including padding segment
        self.n_vocab = n_vocab
        self.n_task = n_task
        self.padding_idx = padding_idx
        self.attentions = []    # for visualization

        # inputs
        self.training = tf.placeholder(tf.bool, None)
        self.tfseq = tf.placeholder(tf.int32, [None, max_len])
        self.tfseq_replaced = tf.placeholder(tf.int32, [None, max_len])
        self.tflabel_rep = tf.cast(tf.placeholder(tf.bool, [None, max_len]), tf.float32)
        self.tftask = tf.placeholder(tf.int32, [None, 1])
        self.tfseg = tf.placeholder(tf.int32, [None, max_len+1])    # +1 to max_len is to consider 1 task index
        self.tfcls = tf.placeholder(tf.int32, [None, ])

        inputs_emb = self.embed_inputs()

        # self.tftask+1 to avoid 0 task because padding is 0
        make_mask = self.make_mask(tf.concat((self.tftask+1, self.tfseq), axis=1))
        z = self.build_net(inputs_emb, make_mask)

        # output heads
        self.seq_logits = tf.layers.dense(z[:, 1:], n_vocab)                # [n, step, n_vocab]
        self.cls_logits = tf.layers.dense(z[:, 0], n_cls)                   # [n, n_cls]

        # losses on the masked or replaced words or whole sentence
        self.loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.tfseq, logits=self.seq_logits) * self.tflabel_rep)
        self.loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.tfcls, logits=self.cls_logits)) + self.loss1

        # train
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):   # for batch norm
            self.train1 = tf.train.AdamOptimizer(0.001).minimize(self.loss1)
            self.train2 = tf.train.AdamOptimizer(0.001).minimize(self.loss2)

        self.sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': 1}))
        self.sess.run(tf.global_variables_initializer())

    def embed_inputs(self):
        word_embeddings = tf.Variable(tf.random_normal((self.n_vocab, self.model_dim), 0., 0.01))
        task_embeddings = tf.Variable(tf.random_normal((self.n_task, self.model_dim), 0., 0.01))    # [n_task, dim]
        segment_embeddings = tf.Variable(tf.random_normal((self.max_seg, self.model_dim), 0., 0.01))  # [max_seg, dim]
        position_embed = tf.Variable(tf.random_normal((1, self.max_len+1, self.model_dim)))  # [1, step+1, dim]

        seg_embed = tf.nn.embedding_lookup(segment_embeddings, self.tfseg)      # [n, step+1, dim]
        seq_embed = tf.nn.embedding_lookup(word_embeddings, self.tfseq_replaced)  # [n, step, dim]
        task_embed = tf.nn.embedding_lookup(task_embeddings, self.tftask)       # [n, 1, dim]
        return tf.concat((task_embed, seq_embed), axis=1) + seg_embed + position_embed  # [n. step+1, dim]

    def build_net(self, z, mask):
        for n in range(self.n_layer):
            z = self.lnorm(self.multi_head(z, z, z, mask) + z)
            z = self.lnorm(self.position_wise_ffn(z) + z)
        return z

    def scaled_dot_product_attention(self, q, k, v, mask):
        dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
        score = tf.matmul(q, tf.transpose(k, (0, 2, 1))) / tf.sqrt(dk)          # [h*n, q_step, step]
        mask = tf.tile(mask, [self.n_head, 1, 1])                               # repeat for n_head [n, q_step, step]
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
        o = tf.layers.conv1d(x, filters=32, kernel_size=1, activation=tf.nn.relu)
        o = tf.layers.conv1d(o, filters=self.model_dim, kernel_size=1)
        o = tf.layers.dropout(o, rate=self.drop_rate, training=self.training)
        return o

    def lnorm(self, x):
        return tf.layers.batch_normalization(x, training=self.training)

    def make_mask(self, seqs):
        """
        pad mask will look like this:
        input|  T,1,2,P,P
        target|
        cls    [1,1,1,0,0]      - this line for predicting class
        1      [1,1,1,0,0]
        2      [1,1,1,0,0]
        P      [0,0,0,0,0]
        P      [0,0,0,0,0]
        """
        pad_mask = tf.where(tf.equal(seqs, self.padding_idx), tf.zeros_like(seqs), tf.ones_like(seqs))  # 0 idx is padding
        pad_mask = tf.cast(tf.expand_dims(pad_mask, axis=1) * tf.expand_dims(pad_mask, axis=2), dtype=tf.bool)  # [n, step, step]
        return pad_mask


def rand_replace(replace_tmp, bl):
    # The replace percentage in this method is different from which in original paper
    sizes = np.maximum(WORD_REPLACE_RATE * (bl-1), 1).astype(np.int)
    replace_change = np.random.rand(len(bl))
    label_rep = np.zeros_like(replace_tmp, dtype=np.bool)
    for i in range(len(replace_tmp)):
        if replace_change[i] < 0.9:     # replace with 90% chance
            if bl[i][0] == 0:   # single sentence
                label_rep[i, np.random.randint(low=1, high=bl[i][1]-1, size=sizes[i][1])] = True
            else:               # two sentence
                label_rep[i, np.random.randint(low=1, high=bl[i][0]-1, size=sizes[i][0])] = True
                label_rep[i, np.random.randint(low=bl[i][0]+1, high=sum(bl[i]), size=sizes[i][1])] = True

            if replace_change[i] < 0.8:     # use mask with 80% chance
                replace_tmp[i, label_rep[i]] = v2i["<MASK>"]
            else:                           # use replace with 10% chance
                replace_tmp[i, label_rep[i]] = [v2i[v] for v in
                                                   np.random.choice(normal_words, size=np.sum(label_rep[i]))]
        else:   # > 90% chance
            label_rep[i, :] = True

    return replace_tmp, label_rep


def train_task1(x1, seg1, task, slen):
    bi = np.random.randint(0, len(x1), size=b_size)
    bx, bs, bl = x1[bi], seg1[bi], slen[bi]
    replaced, label_rep = rand_replace(replace_tmp=bx.copy(), bl=bl)
    _, loss_, seq_logits_ = model.sess.run([model.train1, model.loss1, model.seq_logits], {
        model.tfseq: bx, model.tfseq_replaced: replaced, model.tflabel_rep: label_rep,
        model.tfseg: bs, model.tftask: task, model.training: True})
    return loss_, seq_logits_[0], bx[0], replaced[0]


def train_task2(x2, seg2, y2, task, slen):
    bi = np.random.randint(0, len(x2), size=b_size)
    bx, bs, by, bl = x2[bi], seg2[bi], y2[bi], slen[bi]
    replaced, label_rep = rand_replace(replace_tmp=bx.copy(), bl=bl)
    _, loss_, cls_logits_, seq_logits_ = model.sess.run([model.train2, model.loss2, model.cls_logits, model.seq_logits], {
        model.tfseq: bx, model.tfseq_replaced: replaced, model.tflabel_rep: label_rep, model.tfseg: bs,
        model.tfcls: by, model.tftask: task, model.training: True})
    return loss_, cls_logits_, by, replaced[0], seq_logits_[0], bx[0]


# get and process data
x1, x2, y2, seg1, seg2, max_len, v2i, i2v, len1, len2, normal_words = utils.bert_mrpc("./MRPC")
model = BERT(model_dim=MODEL_DIM, max_len=max_len, n_layer=N_LAYER, n_head=N_HEAD, n_vocab=len(v2i), n_cls=N_CLS,
             max_seg=MAX_SEG, n_task=2, drop_rate=0.1, padding_idx=0)

t0 = time.time()
b_size = 32
task1 = np.full((b_size, 1), 0, np.int32)
task2 = np.full((b_size, 1), 1, np.int32)
for t in range(20000):
    loss1, t1logits, tx1, trpl1 = train_task1(x1, seg1, task1, len1)
    if t % 10 == 0:     # task 2 in this case is easy to learn
        loss2, cls_logits_, by2, trpl2, t2logits, tx2 = train_task2(x2, seg2, y2, task2, len2)
    if t % 50 == 0:
        correct = cls_logits_.argmax(axis=1) == by2
        acc = float(np.sum(correct) / len(cls_logits_))
        t1 = time.time()
        print(
            "\n\nstep: ", t,
            "| time: %.2f" % (t1 - t0),
            "| loss1: %.3f, loss2: %.3f" % (loss1, loss2),
            "\n| t1 tgt: ", u" ".join([i2v[i] for i in tx1 if i != v2i["<PAD>"]]).encode("utf-8"),
            "\n| t1 rpl: ", u" ".join([i2v[i] for i in trpl1 if i != v2i["<PAD>"]]).encode("utf-8"),
            "\n| t1 prd: ", u" ".join([i2v[i] for i in np.argmax(t1logits, axis=1) if i != v2i["<PAD>"]]).encode("utf-8"),
            "\n| t2 acc: %.2f" % acc,
        )
        t0 = t1


# save attention matrix for visualization
# attentions, cls_logits_, seq_logits_ = model.sess.run([model.attentions, model.cls_logits, model.seq_logits], {model.tfx: bx, model.training: False})
# data = {"src": bx, "label": cls_logits_.argmax(axis=1), "seq": seq_logits_.argmax(axis=-1), "attentions": attentions, "i2v": i2v}
# import pickle
# with open("./visual_helper/GPT_attention_matrix.pkl", "wb") as f:
#     pickle.dump(data, f)

