# [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
'''
created by YuYang github.com/W1Fl
'''
import numpy as np
import tensorflow as tf
from tensorflow import keras

import utils

MODEL_DIM = 32
MAX_LEN = 12
N_LAYER = 3
N_HEAD = 4
DATA_SIZE = 6400
BATCH_SIZE = 64
LEARN_RATE = 0.001
EPOCHS = 60


class MultiHead(keras.layers.Layer):
    def __init__(self, n_head):
        super().__init__()
        self.n_head = n_head
        self.attention = None

    def build(self, input_shape):
        (q_b, q_t, q_f), (k_b, k_t, k_f), (v_b, v_t, v_f) = input_shape
        self.k_f = tf.cast(q_f, tf.float32)
        h_dim = q_f // self.n_head
        self.wq = self.add_weight('wq', [self.n_head, q_f, h_dim])
        self.wk = self.add_weight('wk', [self.n_head, k_f, h_dim])
        self.wv = self.add_weight('wv', [self.n_head, v_f, h_dim])
        self.wo = self.add_weight('wo', [self.n_head * h_dim, v_f])
        super(MultiHead, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        i_q, i_k, i_v = [i[:, tf.newaxis, ...] for i in inputs]  # add multihead axis
        q = i_q @ self.wq  # [b,h,s,h_dim]
        k = i_k @ self.wk
        v = i_v @ self.wv
        s = q @ tf.transpose(k, [0, 1, 3, 2]) / (tf.math.sqrt(self.k_f) + 1e-8)
        if mask is not None:
            s += mask * -1e9
        a = tf.nn.softmax(s)  # [b,h,attention,s]
        self.attention = a
        b = a @ v
        o = tf.concat(tf.unstack(b, axis=1), 2) @ self.wo
        return o


class PositionWiseFFN(keras.layers.Layer):
    def build(self, input_shape):
        model_dim = input_shape[-1]
        dff = model_dim * 4
        self.l = keras.layers.Dense(dff, activation=keras.activations.relu)
        self.o = keras.layers.Dense(model_dim)
        super(PositionWiseFFN, self).build(input_shape)

    def call(self, x, **kwargs):
        o = self.l(x)
        o = self.o(o)
        return o  # [n, step, dim]


class EncodeLayer(keras.layers.Layer):
    def __init__(self, n_head):
        self.n_head = n_head
        super().__init__()

    def build(self, input_shape):
        model_dim = input_shape[-1]
        self.ln = [keras.layers.LayerNormalization() for _ in range(2)]
        self.mh = MultiHead(self.n_head)
        self.ffn = PositionWiseFFN(model_dim)
        super(EncodeLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        attn = self.mh([inputs] * 3, mask)  # [n, step, dim]
        o1 = self.ln[0](attn + inputs)
        ffn = self.ffn(o1)
        o = self.ln[1](ffn + o1)  # [n, step, dim]
        return o


class Encoder(keras.layers.Layer):
    def __init__(self, n_head, n_layer):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head

    def build(self, input_shape):
        self.ls = [EncodeLayer(self.n_head) for _ in range(self.n_layer)]
        super(Encoder, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        xz = inputs
        for l in self.ls:
            xz = l(xz, mask)
        return xz  # [n, step, dim]


class DecoderLayer(keras.layers.Layer):
    def __init__(self, n_head):
        super().__init__()
        self.n_head = n_head

    def build(self, input_shape):
        self.mh = [MultiHead(self.n_head) for _ in range(2)]
        self.ffn = PositionWiseFFN(input_shape[-1])
        self.ln = [keras.layers.LayerNormalization() for i in range(3)]
        super(DecoderLayer, self).build(input_shape)

    def call(self, inputs, look_ahead_mask=None, pad_mask=None, **kwargs):
        xz, yz = inputs
        attn = self.mh[0]([yz] * 3, mask=look_ahead_mask)  # decoder self attention
        o1 = self.ln[0](attn + yz)
        attn = self.mh[1]([o1, xz, xz], mask=pad_mask)  # decoder + encoder attention
        o2 = self.ln[1](attn + o1)
        ffn = self.ffn(o2)
        o = self.ln[2](ffn + o2)
        return o


class Decoder(keras.layers.Layer):
    def __init__(self, n_head, n_layer):
        super().__init__()
        self.n_head = n_head
        self.n_layer = n_layer

    def build(self, input_shape):
        self.ls = [DecoderLayer(self.n_head) for _ in range(self.n_layer)]
        super(Decoder, self).build(input_shape)

    def call(self, inputs, look_ahead_mask=None, pad_mask=None):
        xz, yz = inputs
        for l in self.ls:
            yz = l((xz, yz), look_ahead_mask, pad_mask)
        return yz


class PositionEmbedding(keras.layers.Layer):
    def __init__(self, max_len, model_dim, n_vocab):
        super().__init__()
        self.n_vocab = n_vocab
        self.max_len = max_len
        self.model_dim = model_dim

    def build(self, input_shape):
        pos = np.arange(self.max_len)[:, None]
        pe = pos / np.power(10000, 2. * np.arange(self.model_dim)[None, :] / self.model_dim)  # [max_len, dim]
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = pe[None, :, :]  # [1, max_len, model_dim]    for batch adding
        self.pe = tf.constant(pe, dtype=tf.float32)
        self.embeddings = keras.layers.Embedding(
            input_dim=self.n_vocab, output_dim=self.model_dim,  # [n_vocab, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )
        super(PositionEmbedding, self).build(input_shape)

    def call(self, x, **kwargs):
        x_embed = self.embeddings(x) + self.pe  # [n, step, dim]
        return x_embed


class Transformer(keras.Model):
    def __init__(self, model_dim, max_len, n_encoder_layer, n_decoder_layer, n_head, n_vocab, padding_idx=0):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_decoder_layer = n_decoder_layer
        self.n_encoder_layer = n_encoder_layer
        self.n_head = n_head
        self.model_dim = model_dim
        self.max_len = max_len
        self.padding_idx = padding_idx

    def build(self, input_shape):
        self.embed = PositionEmbedding(self.max_len, self.model_dim, self.n_vocab)
        self.encoder = Encoder(self.n_head, self.n_encoder_layer)
        self.decoder = Decoder(self.n_head, self.n_decoder_layer)
        self.o = keras.layers.Dense(self.n_vocab)
        super(Transformer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        x, y = inputs
        x_embed, y_embed = self.embed(x), self.embed(y)
        pad_mask = self._pad_mask(x)
        encoded_z = self.encoder(x_embed, mask=pad_mask)
        decoded_z = self.decoder(
            (encoded_z, y_embed), look_ahead_mask=self._look_ahead_mask(x), pad_mask=pad_mask)
        o = self.o(decoded_z)
        return o

    def _pad_mask(self, seqs):
        mask = tf.cast(tf.math.equal(seqs, self.padding_idx), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]  # (n, 1, 1, step)

    def _look_ahead_mask(self, seqs):
        mask = 1. - tf.linalg.band_part(tf.ones((self.max_len, self.max_len)), -1, 0)
        pad_mask = self._pad_mask(seqs)
        mask = tf.sign(pad_mask + mask[tf.newaxis, tf.newaxis, ...])
        return mask  # (step, step)

    def translate(self, src, i2v, v2i):
        src = tf.reshape(src, (-1, src.shape[-1]))
        src_pad = utils.pad_zero(src, self.max_len)
        tgt = utils.pad_zero(v2i["<GO>"] * tf.ones_like(src), self.max_len + 1)
        tgti = 0
        x_embed = self.embed(src_pad)
        encoded_z = self.encoder(x_embed, mask=self._pad_mask(src_pad))
        while True:
            y = tgt[:, :-1]
            y_embed = self.embed(y)
            decoded_z = self.decoder(
                (encoded_z, y_embed), look_ahead_mask=self._look_ahead_mask(src_pad), pad_mask=self._pad_mask(src_pad))
            logit = self.o(decoded_z)[:, tgti, :].numpy()
            idx = np.argmax(logit, 1)
            tgti += 1
            tgt[:, tgti] = idx
            if tgti >= self.max_len:
                break
        return ["".join([i2v[i] for i in tgt[j, 1:tgti]]) for j in range(len(src))]


class Loss(keras.losses.Loss):
    def __init__(self, padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        self.crossentropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], tf.shape(y_true)[1]])
        pad_mask = tf.math.not_equal(y_true, self.padding_idx)
        loss = tf.reduce_mean(tf.boolean_mask(self.crossentropy(y_true, y_pred), pad_mask))
        return loss


class myTensorboard(keras.callbacks.TensorBoard):
    def __init__(self, data, log_dir='logs/transformer', histogram_freq=1, write_graph=True, write_images=True,
                 embeddings_freq=10, **kwargs):
        self.data = data
        super().__init__(log_dir=log_dir, histogram_freq=histogram_freq, write_graph=write_graph,
                         write_images=write_images, embeddings_freq=embeddings_freq, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        idx2str=lambda idx:[self.data.idx2str(i) for i in idx]
        if (not epoch % 1):
            (x, y), _ = load_data(self.data,3)
            res = self.model.translate(x, self.data.i2v, self.data.v2i)
            target =idx2str(y)
            src = idx2str(x)
            print(
                '\n',
                "| input: ", *src,'\n',
                "| target: ",*target,'\n',
                "| inference: ", *res,'\n',
            )
        super(myTensorboard, self).on_epoch_end(epoch, logs)


def load_data(data,size):
    x, y, seq_len = data.sample(size)
    x = utils.pad_zero(x, MAX_LEN)
    y = utils.pad_zero(y, MAX_LEN + 1)
    return (x, y[:, :-1]), y[:, 1:]


def train(model: Transformer, data):
    x, y = load_data(data,DATA_SIZE)
    tb = myTensorboard(data)
    model.compile(keras.optimizers.Adam(LEARN_RATE), loss=Loss())
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[tb])


if __name__ == "__main__":
    d = utils.DateData(DATA_SIZE)
    print("Chinese time order: yy/mm/dd ", d.date_cn[:3], "\nEnglish time order: dd/M/yyyy ", d.date_en[:3])
    print("vocabularies: ", d.vocab)
    print("x index sample: \n{}\n{}".format(d.idx2str(d.x[0]), d.x[0]),
          "\ny index sample: \n{}\n{}".format(d.idx2str(d.y[0]), d.y[0]))
    m = Transformer(MODEL_DIM, MAX_LEN, N_LAYER, N_LAYER, N_HEAD, d.num_word)
    m.build([[None, 12], [None, 12]])
    train(m, d)
