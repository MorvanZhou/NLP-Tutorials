"""
reference:
https://qianqianqiao.github.io/2018/10/23/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0-Attention-is-All-You-Need/
https://github.com/Kyubyong/transformer/blob/master/modules.py
http://nlp.seas.harvard.edu/2018/04/03/attention.html
https://jalammar.github.io/illustrated-transformer/
https://github.com/pengshuang/Transformer
http://pgn2oyk5g.bkt.clouddn.com/WechatIMG360.png
https://zhuanlan.zhihu.com/p/35571412      (key-value query)
"""

import tensorflow as tf
import numpy as np
import utils

MODEL_DIM = 32
MAX_LEN = 15
N_LAYER = 3
N_HEAD = 4
DROP_RATE = 0.1


class Transformer:
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, drop_rate=0.1):
        self.model_dim = model_dim
        self.max_len = max_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.drop_rate = drop_rate
        self.attentions = []    # for visualization
        self.training = tf.placeholder(tf.bool, None)
        self.tfx = tf.placeholder(tf.int32, [None, max_len])            # [n, step]
        self.tfy = tf.placeholder(tf.int32, [None, max_len+1])          # [n, step+1]

        embeddings = tf.Variable(tf.random_normal((n_vocab, model_dim), 0., 0.01))           # [n_vocab, dim]
        x_embedded = tf.nn.embedding_lookup(embeddings, self.tfx) + self.position_embedding()   # [n, step, dim]
        y_embedded = tf.nn.embedding_lookup(embeddings, self.tfy[:, :-1]) + self.position_embedding()  # [n, step, dim]

        encoded_z = self._build_encoder(x_embedded)
        decoded_z = self._build_decoder(y_embedded, encoded_z)

        self.logits = tf.layers.dense(decoded_z, n_vocab)                # [n, step, n_vocab]
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.tfy[:, 1:], logits=self.logits)
        self.loss = tf.reduce_mean(cross_entropy)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):   # for batch norm
            self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)

        self.sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.sess.run(tf.global_variables_initializer())

    def _build_encoder(self, xz):
        mask = self.pad_mask(self.tfx)
        for n in range(self.n_layer):
            xz = self.lnorm(self.multi_head(xz, xz, xz, mask) + xz)
            xz = self.lnorm(self.position_wise_ffn(xz) + xz)
        return xz

    def _build_decoder(self, yz, xz):
        mask = self.output_mask(self.tfy[:, :-1])
        for n in range(self.n_layer):
            yz = self.lnorm(self.multi_head(yz, yz, yz, mask) + yz)
            yz = self.lnorm(self.multi_head(yz, xz, xz, None) + yz)
            yz = self.lnorm(self.position_wise_ffn(yz) + yz)
        return yz

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
        score = tf.matmul(q, tf.transpose(k, (0, 2, 1))) / tf.sqrt(dk)          # [h*n, q_step, step]
        if mask is not None:
            mask = tf.tile(mask, [self.n_head, 1, 1])                           # repeat for n_head
            score = tf.where(mask, score, tf.fill(tf.shape(score), -np.inf))    # make softmax not select padded value
        attention = tf.nn.softmax(score, axis=-1)                               # [h*n, q_step, step]
        attention = tf.layers.dropout(attention, rate=self.drop_rate, training=self.training)
        context = tf.matmul(attention, v)           # [h*n, q_step, step] @ [h*n, step, dv] = [h*n, q_step, dv]
        return attention, context

    def multi_head(self, query, key, value, mask=None):
        head_dim = self.model_dim // self.n_head
        q = tf.layers.dense(query, self.n_head * head_dim)          # [n, q_step, h_dim*h]
        k = tf.layers.dense(key, self.n_head * head_dim)            # [n, step, h_dim*h]
        v = tf.layers.dense(value, self.n_head * head_dim)          # [n, step, h_dim*h]
        q_ = tf.concat(tf.split(q, self.n_head, axis=2), axis=0)    # [h*n, q_step, h_dim]
        k_ = tf.concat(tf.split(k, self.n_head, axis=2), axis=0)    # [h*n, step, h_dim]
        v_ = tf.concat(tf.split(v, self.n_head, axis=2), axis=0)    # [h*n, step, h_dim]

        attention, context = self.scaled_dot_product_attention(q_, k_, v_, mask)
        self.attentions.append(tf.transpose(tf.split(attention, self.n_head, axis=0), (1, 0, 2, 3))) # [n, h, q_step, step]
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

    def position_embedding(self):
        pos = np.arange(self.max_len)[:, None]
        pe = pos / np.power(10000, 2. * np.arange(self.model_dim)[None, :] / self.model_dim)  # [max_len, model_dim]
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = pe[None, :, :]                         # [1, max_len, model_dim]    for batch adding
        return tf.constant(pe, dtype=tf.float32)

    @staticmethod
    def pad_mask(seqs):
        mask = tf.where(seqs == 0, tf.zeros_like(seqs), tf.ones_like(seqs))                         # 0 idx is padding
        return tf.cast(tf.expand_dims(mask, axis=1) * tf.expand_dims(mask, axis=2), dtype=tf.bool)  # [n, step, step]

    def output_mask(self, seqs):
        pad_mask = self.pad_mask(seqs)
        mask = tf.constant(~np.triu(np.ones((self.max_len, self.max_len), dtype=np.bool), 1))
        mask = tf.tile(tf.expand_dims(mask, axis=0), [tf.shape(seqs)[0], 1, 1])     # [n, step, step]
        return tf.where(mask, pad_mask, tf.zeros_like(pad_mask))

    def translate(self, src, v2i, i2v):
        src_pad = utils.pad_zero(np.array([v2i[v] for v in src])[None, :], MAX_LEN)
        tgt_seq = "<GO>"
        tgt = utils.pad_zero(np.array([v2i[tgt_seq], ])[None, :], MAX_LEN + 1)
        tgti = 0
        while True:
            logit = self.sess.run(self.logits, {self.tfx: src_pad, self.tfy: tgt, self.training: False})[0, tgti, :]
            idx = np.argmax(logit)
            tgti += 1
            tgt[0, tgti] = idx
            if idx == v2i["<EOS>"] or tgti >= self.max_len:
                break
        return "".join([i2v[i] for i in tgt[0, 1:tgti]])


# get and process data
vocab, x, y, v2i, i2v, date_cn, date_en = utils.get_date_data()
print("Chinese time order: ", date_cn[:3], "\nEnglish time order: ", date_en[:3])
print("vocabularies: ", vocab)
print("x index sample: \n", x[:2], "\ny index sample: \n", y[:2])

model = Transformer(MODEL_DIM, MAX_LEN, N_LAYER, N_HEAD, DROP_RATE)
# training
for t in range(1000):
    bi = np.random.randint(0, len(x), size=64)
    bx, by = utils.pad_zero(x[bi], max_len=MAX_LEN), utils.pad_zero(y[bi], max_len=MAX_LEN+1)
    _, loss_ = model.sess.run([model.train_op, model.loss], {model.tfx: bx, model.tfy: by, model.training: True})
    if t % 50 == 0:
        logits_ = model.sess.run(model.logits, {model.tfx: bx[:1, :], model.tfy: by[:1, :], model.training: False})
        target = []
        for i in by[0, 1:]:
            if i == v2i["<EOS>"]:break
            target.append(i2v[i])
        target = "".join(target)
        res = []
        for i in np.argmax(logits_[0], axis=1):
            if i == v2i["<EOS>"]:break
            res.append(i2v[i])
        res = "".join(res)
        print(
            "step: ", t,
            "| loss: %.3f" % loss_,
            "| target: ", target,
            "| inference: ", res,
        )

# prediction
src_seq = "02-11-30"
print("src: ", src_seq, "\nprediction: ", model.translate(src_seq, v2i, i2v))

# save attention matrix for visualization
# attentions = model.sess.run(model.attentions, {model.tfx: bx[:1, :], model.tfy: by[:1, :], model.training: False})
# data = {"src": [i2v[i] for i in x[0]], "tgt": [i2v[i] for i in y[0]], "attentions": attentions}
# import pickle
# with open("./visual_helper/attention_matrix.pkl", "wb") as f:
#     pickle.dump(data, f)
