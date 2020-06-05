from tensorflow import keras
import tensorflow as tf
import utils

BATCH_SIZE = 32

class ELMO(keras.Model):
    def __init__(self, v_dim, emb_dim, units):
        super().__init__()
        self.units = units

        # encoder
        self.embed = keras.layers.Embedding(
            input_dim=v_dim, output_dim=emb_dim,  # [n_vocab, emb_dim]
            embeddings_initializer=keras.initializers.RandomNormal(0., 0.1),
        )
        self.lstm_forward = keras.layers.LSTM(units=units, return_sequences=True)
        self.lstm_backward = keras.layers.LSTM(units=units, return_sequences=True, go_backwards=True)
        self.word_pred = keras.layers.Dense(v_dim)
        self.nsp = keras.layers.Dense(2)

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.05, clipnorm=5.0)

    def __call__(self, x):
        o = self.embed(x)       # [n, step, dim]
        init_s = [tf.zeros((x.shape[0], self.units)), tf.zeros((x.shape[0], self.units))]
        f = self.lstm_forward(o[:, 1:], initial_state=init_s)
        b = self.lstm_backward(o[:, :-1], initial_state=init_s)
        o = tf.concat((f, b), axis=0)
        word_pred = self.word_pred(o)
        nsp = self.nsp(o)
        return word_pred, nsp

    def inference(self, x):
        s = self.encode(x)
        done, i, s = self.decoder_eval.initialize(
            self.dec_embeddings.variables[0],
            start_tokens=tf.fill([x.shape[0], ], self.start_token),
            end_token=self.end_token,
            initial_state=s,
        )
        pred_id = np.zeros((x.shape[0], self.max_pred_len), dtype=np.int32)
        for l in range(self.max_pred_len):
            o, s, i, done = self.decoder_eval.step(
                time=l, inputs=i, state=s, training=False)
            pred_id[:, l] = o.sample_id
        return pred_id


def main():
    data = utils.MRPCData("./MRPC")
    print("num word: ", data.num_word)
    model = ELMO(data.num_word, 32, 32)
    for t in range(2500):
        seqs, _, _, nsp_labels = data.sample(BATCH_SIZE)
        loss, pred = model(seqs, )

if __name__ == "__main__":
    main()