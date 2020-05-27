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
from tensorflow import keras
import numpy as np
import utils
import time
import pickle


MODEL_DIM = 32
MAX_LEN = 12
N_LAYER = 3
N_HEAD = 4
DROP_RATE = 0.1


class PositionEmbedding(keras.layers.Layer):
    def __init__(self, max_len, model_dim, n_vocab):
        super().__init__()
        pos = np.arange(max_len)[:, None]
        pe = pos / np.power(10000, 2. * np.arange(model_dim)[None, :] / model_dim)  # [max_len, dim]
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = pe[None, :, :]  # [1, max_len, model_dim]    for batch adding
        self.pe = tf.constant(pe, dtype=tf.float32)
        self.embeddings = keras.layers.Embedding(
            input_dim=n_vocab, output_dim=model_dim,  # [n_vocab, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )

    def __call__(self, x, y):
        x_embed = self.embeddings(x) + self.pe  # [n, step, dim]
        y_embed = self.embeddings(y[:, :-1]) + self.pe  # [n, step, dim]
        return x_embed, y_embed


class MultiHead(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.head_dim = model_dim // n_head
        self.n_head = n_head
        self.model_dim = model_dim
        self.wq = keras.layers.Dense(n_head * self.head_dim)
        self.wk = keras.layers.Dense(n_head * self.head_dim)
        self.wv = keras.layers.Dense(n_head * self.head_dim)      # [n, step, h*h_dim]

        self.scaled_drop = keras.layers.Dropout(rate=drop_rate)
        self.o_dense = keras.layers.Dense(model_dim, use_bias=False)
        self.o_drop = keras.layers.Dropout(rate=drop_rate)

        self.attention = None

    def __call__(self, q, k, v, mask, training):
        _q = self.wq(q)      # [n, q_step, h*h_dim]
        _k, _v = self.wk(k), self.wv(v)     # [n, step, h*h_dim]
        _q = self.split_heads(_q)  # [n, h, q_step, h_dim]
        _k, _v = self.split_heads(_k), self.split_heads(_v)  # [n, h, step, h_dim]
        context = self.scaled_dot_product_attention(_q, _k, _v, training, mask)     # [n, q_step, h*dv]
        o = self.o_dense(context)
        o = self.o_drop(o, training=training)
        return o

    def split_heads(self, x):
        x = tf.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.head_dim))  # [n, step, h, h_dim]
        return tf.transpose(x, perm=[0, 2, 1, 3])       # [n, h, step, h_dim]

    def scaled_dot_product_attention(self, q, k, v, training, mask=None):
        dk = tf.cast(k.shape[-1], dtype=tf.float32)
        score = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)  # [n, h_dim, q_step, step]
        if mask is not None:
            score += mask * -1e9
        self.attention = tf.nn.softmax(score, axis=-1)                               # [n, h, q_step, step]
        attention = self.scaled_drop(self.attention, training=training)
        context = tf.matmul(attention, v)               # [n, h, q_step, step] @ [n, h, step, dv] = [n, h, q_step, dv]
        context = tf.transpose(context, perm=[0, 2, 1, 3])   # [n, q_step, h, dv]
        context = tf.reshape(context, (context.shape[0], context.shape[1], -1))     # [n, q_step, h*dv]
        return context


class PositionWiseFFN(keras.layers.Layer):
    def __init__(self, model_dim, drop_rate):
        super().__init__()
        self.l = keras.layers.Conv1D(filters=32, kernel_size=1, activation=tf.nn.relu)
        self.o = keras.layers.Conv1D(filters=model_dim, kernel_size=1)
        self.drop = keras.layers.Dropout(rate=drop_rate)

    def __call__(self, x, training):
        o = self.l(x)
        o = self.o(o)
        o = self.drop(o, training=training)
        return o


class EncodeLayer(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.multi_head = MultiHead(n_head, model_dim, drop_rate)
        self.ffn = PositionWiseFFN(model_dim, drop_rate)

    def __call__(self, xz, training, mask):
        xz = self.bn1(self.multi_head(xz, xz, xz, mask, training) + xz, training)
        o = self.bn2(self.ffn(xz, training) + xz, training)
        return o


class Encoder(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.ls = [EncodeLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]

    def __call__(self, xz, training, mask):
        for l in self.ls:
            xz = l(xz, training, mask)
        return xz


class DecoderLayer(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()
        self.multi_head1 = MultiHead(n_head, model_dim, drop_rate)
        self.multi_head2 = MultiHead(n_head, model_dim, drop_rate)
        self.ffn = PositionWiseFFN(model_dim, drop_rate)

    def __call__(self, yz, xz, training, mask):
        yz = self.bn1(self.multi_head1(yz, yz, yz, mask, training) + yz, training)
        yz = self.bn2(self.multi_head2(yz, xz, xz, None, training) + yz, training)
        o = self.bn3(self.ffn(yz, training) + yz, training)
        return o


class Decoder(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.ls = [DecoderLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]

    def __call__(self, yz, xz, training, mask):
        for l in self.ls:
            xz = l(xz, yz, training, mask)
        return xz


class Transformer(keras.Model):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, drop_rate=0.1, padding_idx=0):
        super().__init__()
        self.max_len = max_len
        self.padding_idx = padding_idx

        self.embedding = PositionEmbedding(max_len, model_dim, n_vocab)
        self.encoder = Encoder(n_head, model_dim, drop_rate, n_layer)
        self.decoder = Decoder(n_head, model_dim, drop_rate, n_layer)
        self.o = keras.layers.Dense(n_vocab)

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.002)

    def __call__(self, x, y, training=None):
        x_embed, y_embed = self.embedding(x, y)
        encoded_z = self.encoder(x_embed, training, mask=self._pad_mask(x))
        decoded_z = self.decoder(y_embed, encoded_z, training, mask=self._look_ahead_mask())
        o = self.o(decoded_z)
        return o

    def step(self, x, y):
        with tf.GradientTape() as tape:
            _logits = self(x, y, training=True)
            _loss = self.cross_entropy(y[:, 1:], _logits)
        grads = tape.gradient(_loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return _loss.numpy()

    def _pad_mask(self, seqs):
        seqs = tf.cast(tf.math.equal(seqs, self.padding_idx), tf.float32)
        return seqs[:, tf.newaxis, tf.newaxis, :]  # (n, 1, 1, step)

    def _look_ahead_mask(self):
        mask = 1 - tf.linalg.band_part(tf.ones((self.max_len, self.max_len)), -1, 0)
        return mask  # (step, step)

    def translate(self, src, v2i, i2v):
        src_pad = utils._pad_zero(np.array([v2i[v] for v in src])[None, :], self.max_len)
        tgt_seq = "<GO>"
        tgt = utils._pad_zero(np.array([v2i[tgt_seq], ])[None, :], self.max_len + 1)
        tgti = 0
        while True:
            logit = self(src_pad, tgt, False)[0, tgti, :]
            idx = np.argmax(logit)
            tgti += 1
            tgt[0, tgti] = idx
            if idx == v2i["<EOS>"] or tgti >= self.max_len:
                break
        return "".join([i2v[i] for i in tgt[0, 1:tgti]])

    @property
    def attentions(self):
        attentions = {
            "encoder": [l.multi_head.attention.numpy() for l in self.encoder.ls],
            "decoder": {
                "mh1": [l.multi_head1.attention.numpy() for l in self.decoder.ls],
                "mh2": [l.multi_head2.attention.numpy() for l in self.decoder.ls],
        }}
        return attentions


if __name__ == "__main__":
    # get and process data
    data = utils.DateData(2000)
    print("Chinese time order: yy/mm/dd ", data.date_cn[:3], "\nEnglish time order: dd/M/yyyy ", data.date_en[:3])
    print("vocabularies: ", data.vocab)
    print("x index sample: \n{}\n{}".format(data.idx2str(data.x[0]), data.x[0]),
          "\ny index sample: \n{}\n{}".format(data.idx2str(data.y[0]), data.y[0]))

    model = Transformer(MODEL_DIM, MAX_LEN, N_LAYER, N_HEAD, data.num_word, DROP_RATE)
    # training
    t0 = time.time()
    for t in range(1000):
        bx, by, seq_len = data.sample(64)
        bx, by = utils._pad_zero(bx, max_len=MAX_LEN), utils._pad_zero(by, max_len=MAX_LEN + 1)
        loss = model.step(bx, by)
        if t % 50 == 0:
            logits = model(bx[:1, :], by[:1, :], False).numpy()
            t1 = time.time()
            print(
                "step: ", t,
                "| time: %.2f" % (t1-t0),
                "| loss: %.3f" % loss,
                "| target: ", "".join([data.i2v[i] for i in by[0, 1:] if i != data.v2i["<PAD>"]]),
                "| inference: ", "".join([data.i2v[i] for i in np.argmax(logits[0], axis=1) if i != data.v2i["<PAD>"]]),
            )
            t0 = t1

    # prediction
    src_seq = "02-11-30"
    print("src: ", src_seq, "\nprediction: ", model.translate(src_seq, data.v2i, data.i2v))

    # save attention matrix for visualization
    _ = model(bx[:1, :], by[:1, :], training=False)

    data = {"src": [data.i2v[i] for i in data.x[0]], "tgt": [data.i2v[i] for i in data.y[0]], "attentions": model.attentions}
    with open("./visual_helper/transformer_attention_matrix.pkl", "wb") as f:
        pickle.dump(data, f)
