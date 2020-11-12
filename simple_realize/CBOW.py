# [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
from io import BytesIO

import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from utils import process_w2v_data

Batch_size = 32
Learn_rate = 0.01
Epochs = 256
DataSize = 512

corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]

SkipGram = lambda v_dim, emb_dim: keras.Sequential([
    keras.layers.Embedding(
        input_dim=v_dim, output_dim=emb_dim,  # [n_vocab, emb_dim]
        embeddings_initializer=keras.initializers.RandomNormal(0., 0.1),
    ),
    keras.layers.Lambda(lambda x:tf.reduce_mean(x,1))
])


class myTensorboard(keras.callbacks.TensorBoard):
    def __init__(self, data, log_dir='logs/CBOW', histogram_freq=1, write_graph=True, write_images=True,
                 embeddings_freq=10, **kwargs):
        super().__init__(log_dir=log_dir, histogram_freq=histogram_freq, write_graph=write_graph,
                         write_images=write_images, embeddings_freq=embeddings_freq, **kwargs)
        self.buffer = BytesIO()
        self.data = data


    def plot(self, data):
        word_emb = model.layers[0].get_weights()[0]
        for i in range(data.num_word):
            c = "blue"
            try:
                int(data.i2v[i])
            except ValueError:
                c = "red"
            plt.text(word_emb[i, 0], word_emb[i, 1], s=data.i2v[i], color=c, weight="bold")
        plt.xlim(word_emb[:, 0].min() - .5, word_emb[:, 0].max() + .5)
        plt.ylim(word_emb[:, 1].min() - .5, word_emb[:, 1].max() + .5)
        plt.xticks(())
        plt.yticks(())
        plt.xlabel("embedding dim1")
        plt.ylabel("embedding dim2")
        plt.savefig(self.buffer, format='png')
        plt.close()
        self.buffer.seek(0)

    def on_epoch_end(self, epoch, logs=None):
        writer = self._get_writer(self._train_run_name)
        if (not epoch % 1):
            self.plot(self.data)
            with writer.as_default():
                tf.summary.image('embedding', imageio.imread(self.buffer)[None, :], step=epoch)
                self.buffer.seek(0)
        super(myTensorboard, self).on_epoch_end(epoch, logs)


class nce_loss(keras.losses.Loss):
    # negative sampling: take one positive label and num_sampled negative labels to compute the loss
    # in order to reduce the computation of full softmax
    def __init__(self, model, v_dim, emb_dim):
        super(nce_loss, self).__init__()
        # noise-contrastive estimation
        self.nce_w = model.add_weight(
            name="nce_w", shape=[v_dim, emb_dim],
            initializer=keras.initializers.TruncatedNormal(0., 0.1))  # [n_vocab, emb_dim]
        self.nce_b = model.add_weight(
            name="nce_b", shape=(v_dim,),
            initializer=keras.initializers.Constant(0.1))  # [n_vocab, ]
        self.v_dim = v_dim

    def call(self, y_true, y_pred):
        # return keras.losses.SparseCategoricalCrossentropy()(y_true,y_pred)
        return tf.nn.nce_loss(
            weights=self.nce_w, biases=self.nce_b, labels=y_true,
            inputs=y_pred, num_sampled=5, num_classes=self.v_dim)


if __name__ == "__main__":
    d = process_w2v_data(corpus, skip_window=2, method="cbow")
    bx, by = d.sample(DataSize)
    model = SkipGram(d.num_word, 2)
    model.compile(optimizer=keras.optimizers.Adam(Learn_rate), loss=nce_loss(model, d.num_word, 2))
    model.fit(bx, by, Batch_size, Epochs, callbacks=[myTensorboard(d)], verbose=2)

    #use tensorboard --logdir logs --samples_per_plugin=images=255 to show all images
