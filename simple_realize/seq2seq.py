# [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
import tensorflow as tf
from tensorflow import keras
import utils

Batch_size = 64
Learn_rate = 0.001
Epochs = 30
DataSize = 8192


class Seq2Seq(keras.Model):
    def __init__(self, enc_v_dim, dec_v_dim, emb_dim, units, max_pred_len, start_token, end_token):
        super().__init__()
        self.enc_v_dim = enc_v_dim
        self.emb_dim = emb_dim
        self.units = units
        self.dec_v_dim = dec_v_dim
        self.max_pred_len = max_pred_len
        self.start_token = start_token
        self.end_token = end_token

    def build(self, input_shape):
        # encoder
        self.enc_embeddings = keras.layers.Embedding(
            input_dim=self.enc_v_dim,
            output_dim=self.emb_dim,  # [enc_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
            name='encoder/embeddings'
        )
        self.encoder = keras.layers.LSTM(units=self.units, return_state=True, name='encoder/LSTM')

        # decoder
        self.dec_embeddings = keras.layers.Embedding(
            input_dim=self.dec_v_dim, output_dim=self.emb_dim,  # [dec_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
            name='decoder/embeddings'

        )
        self.dec_embeddings.build((None, self.dec_v_dim))
        self.decoder = keras.layers.LSTM(units=self.units, return_state=True, return_sequences=True,
                                         name='decoder/LSTM')
        self.decoder_dense = keras.layers.Dense(self.dec_v_dim, activation=keras.activations.softmax,
                                                name='decoder/Dense')

        self.batch = input_shape[0][0]
        super(Seq2Seq, self).build([*input_shape])

    def encode(self, x):
        embedded = self.enc_embeddings(x)
        o, h, c = self.encoder(embedded)
        return h, c

    def decode(self, batch, h, c, y=None, training=None):
        if training: #将上一时刻的标签作为当前时刻的输入
            y = self.dec_embeddings(y)
            y, h, c = self.decoder(y, (h, c))
            y = self.decoder_dense(y)
        else:#将上一时刻的输出作为当前时刻的输入
            y = []
            o = tf.zeros((batch, 1, self.dec_v_dim))
            for i in range(self.max_pred_len):
                o = o @ self.dec_embeddings.weights
                o, h, c = self.decoder(o, (h, c))
                o = self.decoder_dense(o)
                y.append(o)
            y = tf.concat(y, 1)
        return y

    # @tf.function
    def call(self, inputs, training=None, mask=None):
        x = inputs[0]
        y = inputs[1]
        if training:
            y = tf.pad(y[:, :-1], [[0, 0], [1, 0]])
        h, c = self.encode(x)
        batch = tf.shape(x)[0]
        y = self.decode(batch, h, c, y, training)
        return y


class myTensorboard(keras.callbacks.TensorBoard):
    def __init__(self, data, log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
                 embeddings_freq=True, **kwargs):
        self.data = data
        super().__init__(log_dir=log_dir, histogram_freq=histogram_freq, write_graph=write_graph,
                         write_images=write_images, embeddings_freq=embeddings_freq, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if (not epoch % 5):
            x, y, l = self.data.sample(1)
            y_ = self.model((x, y), training=False)
            y_ = tf.argmax(y_, -1).numpy()
            target = self.data.idx2str(y[0, 1:-1])
            res = self.data.idx2str(y_[0])
            src = self.data.idx2str(x[0])
            print(
                "t: ", epoch,
                "| input: ", src,
                "| target: ", target,
                "| inference: ", res,
            )
        super(myTensorboard, self).on_epoch_end(epoch, logs)


def train():
    # get and process data
    data = utils.DateData(DataSize)
    print("Chinese time order: yy/mm/dd ", data.date_cn[:3], "\nEnglish time order: dd/M/yyyy ", data.date_en[:3])
    print("vocabularies: ", data.vocab)
    print("x index sample: \n{}\n{}".format(data.idx2str(data.x[0]), data.x[0]),
          "\ny index sample: \n{}\n{}".format(data.idx2str(data.y[0]), data.y[0]))

    model = Seq2Seq(
        data.num_word, data.num_word, emb_dim=16, units=32,
        max_pred_len=11, start_token=data.start_token, end_token=data.end_token)

    model.compile(optimizer=keras.optimizers.Adam(Learn_rate), loss=keras.losses.SparseCategoricalCrossentropy(False),
                  metrics=[keras.metrics.sparse_categorical_accuracy])
    train_x, train_y, train_l = data.sample(DataSize)
    model.fit((train_x, train_y), train_y, callbacks=[myTensorboard(data)], batch_size=Batch_size, epochs=Epochs)


if __name__ == "__main__":
    train()
