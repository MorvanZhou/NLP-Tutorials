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
import os


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

    def __call__(self, x):
        x_embed = self.embeddings(x) + self.pe  # [n, step, dim]
        return x_embed


class MultiHead(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.head_dim = model_dim // n_head
        self.n_head = n_head
        self.model_dim = model_dim
        self.wq = keras.layers.Dense(n_head * self.head_dim)
        self.wk = keras.layers.Dense(n_head * self.head_dim)
        self.wv = keras.layers.Dense(n_head * self.head_dim)      # [n, step, h*h_dim]

        self.o_dense = keras.layers.Dense(model_dim)
        self.o_drop = keras.layers.Dropout(rate=drop_rate)
        self.attention = None

    def __call__(self, q, k, v, mask, training):
        _q = self.wq(q)      # [n, q_step, h*h_dim]
        _k, _v = self.wk(k), self.wv(v)     # [n, step, h*h_dim]
        _q = self.split_heads(_q)  # [n, h, q_step, h_dim]
        _k, _v = self.split_heads(_k), self.split_heads(_v)  # [n, h, step, h_dim]
        context = self.scaled_dot_product_attention(_q, _k, _v, mask)     # [n, q_step, h*dv]
        o = self.o_dense(context)       # [n, step, dim]
        o = self.o_drop(o, training=training)
        return o

    def split_heads(self, x):
        x = tf.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.head_dim))  # [n, step, h, h_dim]
        return tf.transpose(x, perm=[0, 2, 1, 3])       # [n, h, step, h_dim]

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = tf.cast(k.shape[-1], dtype=tf.float32)
        score = tf.matmul(q, k, transpose_b=True) / (tf.math.sqrt(dk) + 1e-8)  # [n, h_dim, q_step, step]
        if mask is not None:
            score += mask * -1e9
        self.attention = tf.nn.softmax(score, axis=-1)                               # [n, h, q_step, step]
        context = tf.matmul(self.attention, v)         # [n, h, q_step, step] @ [n, h, step, dv] = [n, h, q_step, dv]
        context = tf.transpose(context, perm=[0, 2, 1, 3])   # [n, q_step, h, dv]
        context = tf.reshape(context, (context.shape[0], context.shape[1], -1))     # [n, q_step, h*dv]
        return context


class PositionWiseFFN(keras.layers.Layer):
    def __init__(self, model_dim):
        super().__init__()
        dff = model_dim * 4
        self.l = keras.layers.Dense(dff, activation=keras.activations.relu)
        self.o = keras.layers.Dense(model_dim)

    def __call__(self, x):
        o = self.l(x)
        o = self.o(o)
        return o         # [n, step, dim]


class EncodeLayer(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.bn = [keras.layers.BatchNormalization() for _ in range(2)]
        self.mh = MultiHead(n_head, model_dim, drop_rate)
        self.ffn = PositionWiseFFN(model_dim)
        self.drop = keras.layers.Dropout(drop_rate)

    def __call__(self, xz, training, mask):
        attn = self.mh(xz, xz, xz, mask, training)       # [n, step, dim]
        o1 = self.bn[0](attn + xz, training)
        ffn = self.drop(self.ffn(o1), training)
        o = self.bn[1](ffn + xz, training)         # [n, step, dim]
        return o


class Encoder(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.ls = [EncodeLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]

    def __call__(self, xz, training, mask):
        for l in self.ls:
            xz = l(xz, training, mask)
        return xz       # [n, step, dim]


class DecoderLayer(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.bn = [keras.layers.BatchNormalization() for _ in range(3)]
        self.drop = keras.layers.Dropout(drop_rate)
        self.mh = [MultiHead(n_head, model_dim, drop_rate) for _ in range(2)]
        self.ffn = PositionWiseFFN(model_dim)

    def __call__(self, yz, xz, training, mask):
        attn = self.mh[0](yz, yz, yz, mask, training)       # decoder self attention
        o1 = self.bn[0](attn + yz, training)
        attn = self.mh[1](o1, xz, xz, None, training)       # decoder + encoder attention
        o2 = self.bn[1](attn + o1, training)
        ffn = self.drop(self.ffn(o2), training)
        o = self.bn[2](ffn + o2, training)
        return o


class Decoder(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.ls = [DecoderLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]

    def __call__(self, yz, xz, training, mask):
        for l in self.ls:
            yz = l(yz, xz, training, mask)
        return yz


class Transformer(keras.Model):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, drop_rate=0.1, padding_idx=0):
        super().__init__()
        self.max_len = max_len
        self.padding_idx = padding_idx

        self.embed = PositionEmbedding(max_len, model_dim, n_vocab)
        self.encoder = Encoder(n_head, model_dim, drop_rate, n_layer)
        self.decoder = Decoder(n_head, model_dim, drop_rate, n_layer)
        self.o = keras.layers.Dense(n_vocab)

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.002)

    def __call__(self, x, y, training=None):
        x_embed, y_embed = self.embed(x), self.embed(y)
        encoded_z = self.encoder(x_embed, training, mask=self._pad_mask(x))
        decoded_z = self.decoder(y_embed, encoded_z, training, mask=self._look_ahead_mask())
        o = self.o(decoded_z)
        return o

    def step(self, x, y):
        with tf.GradientTape() as tape:
            _logits = self(x, y[:, :-1], training=True)
            _loss = self.cross_entropy(y[:, 1:], _logits)
        grads = tape.gradient(_loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return _loss

    def _pad_mask(self, seqs):
        _seqs = tf.cast(tf.math.equal(seqs, self.padding_idx), tf.float32)
        return _seqs[:, tf.newaxis, tf.newaxis, :]  # (n, 1, 1, step)

    def _look_ahead_mask(self):
        mask = 1 - tf.linalg.band_part(tf.ones((self.max_len, self.max_len)), -1, 0)
        return mask  # (step, step)

    def translate(self, src, v2i, i2v):
        src_pad = utils.pad_zero(np.array([v2i[v] for v in src])[None, :], self.max_len)
        tgt = utils.pad_zero(np.array([v2i["<GO>"], ])[None, :], self.max_len+1)
        tgti = 0
        x_embed = self.embed(src_pad)
        encoded_z = self.encoder(x_embed, False, mask=self._pad_mask(src_pad))
        while True:
            y_embed = self.embed(tgt[:, :-1])
            decoded_z = self.decoder(y_embed, encoded_z, False, mask=self._look_ahead_mask())
            logit = self.o(decoded_z)[0, tgti, :].numpy()
            idx = np.argmax(logit)
            tgti += 1
            tgt[0, tgti] = idx
            if idx == v2i["<EOS>"] or tgti >= self.max_len:
                break
        return "".join([i2v[i] for i in tgt[0, 1:tgti]])

    @property
    def attentions(self):
        attentions = {
            "encoder": [l.mh.attention.numpy() for l in self.encoder.ls],
            "decoder": {
                "mh1": [l.mh[0].attention.numpy() for l in self.decoder.ls],
                "mh2": [l.mh[1].attention.numpy() for l in self.decoder.ls],
        }}
        return attentions


def main():
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
        bx, by = utils.pad_zero(bx, max_len=MAX_LEN), utils.pad_zero(by, max_len=MAX_LEN + 1)
        loss = model.step(bx, by)
        if t % 50 == 0:
            logits = model(bx[:1], by[:1, :-1], False)[0].numpy()
            t1 = time.time()
            print(
                "step: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.4f" % loss.numpy(),
                "| target: ", "".join([data.i2v[i] for i in by[0, 1:] if i != data.v2i["<PAD>"]]),
                "| inference: ", "".join([data.i2v[i] for i in np.argmax(logits, axis=1) if i != data.v2i["<PAD>"]]),
            )
            t0 = t1

    os.makedirs("./visual_helper/transformer", exist_ok=True)
    model.save_weights("./visual_helper/transformer/model.ckpt")
    with open("./visual_helper/transformer_v2i_i2v.pkl", "wb") as f:
        pickle.dump({"v2i": data.v2i, "i2v": data.i2v}, f)

    # prediction
    src_seq = "02-11-30"
    print("src: ", src_seq, "\nprediction: ", model.translate(src_seq, data.v2i, data.i2v))

    # save attention matrix for visualization
    _ = model(bx[:1], by[:1, :-1], training=False)

    data = {"src": [data.i2v[i] for i in data.x[0]], "tgt": [data.i2v[i] for i in data.y[0]],
            "attentions": model.attentions}
    with open("./visual_helper/transformer_attention_matrix.pkl", "wb") as f:
        pickle.dump(data, f)


def load():
    with open("./visual_helper/transformer_v2i_i2v.pkl", "rb") as f:
        dic = pickle.load(f)
    model = Transformer(MODEL_DIM, MAX_LEN, N_LAYER, N_HEAD, len(dic["v2i"]), DROP_RATE)
    model.load_weights("./visual_helper/transformer/model.ckpt")
    src = "08-10-08"
    print(src, model.translate(src, dic["v2i"], dic["i2v"]))


if __name__ == "__main__":
    main()
    # load()