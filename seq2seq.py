"""
More details about NMT: https://github.com/tensorflow/nmt
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils
import tensorflow_addons as tfa


class Seq2Seq(keras.Model):
    def __init__(self, enc_v_dim, dec_v_dim, emb_dim, units, max_pred_len, start_token, end_token, v2i):
        super().__init__()

        self.enc_embeddings = keras.layers.Embedding(
            input_dim=enc_v_dim, output_dim=emb_dim,       # [n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        self.dec_embeddings = keras.layers.Embedding(
            input_dim=dec_v_dim, output_dim=emb_dim,  # [n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        self.units = units
        self.encoder = keras.layers.LSTM(units=units, return_sequences=True, return_state=True)
        decoder_dense = keras.layers.Dense(1)
        self.decoder_train = tfa.seq2seq.BasicDecoder(
            cell=keras.layers.LSTM(units=units, return_sequences=True, return_state=True),
            sampler=tfa.seq2seq.sampler.TrainingSampler(), output_layer=decoder_dense
        )
        self.decoder_eval = tfa.seq2seq.BasicDecoder(
            cell=keras.layers.LSTM(units=units, return_sequences=True, return_state=True),
            sampler=tfa.seq2seq.sampler.GreedyEmbeddingSampler(), output_layer=decoder_dense
        )

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.0001, clipnorm=5.0)
        self.train_sampler = tfa.seq2seq.sampler.TrainingSampler()
        self.max_pred_len = max_pred_len
        self.start_token = start_token
        self.end_token = end_token
        self.v2i = v2i

    def call(self, x, training=None, mask=None):
        o, s = self.encode(x)
        o, s = self.decode(o, s, training)
        return o

    def encode(self, x: tf.Tensor):
        o = self.enc_embeddings(x)
        init_s = (tf.zeros((x.shape[0], self.units)), tf.zeros(x.shape[0], self.units))
        o, h, c = self.encoder(o, initial_state=init_s)
        return o, (h, c)

    def decode(self, x, s, training=False):
        o = self.dec_embeddings(x)
        if training is None or training:
            o, h, c = self.decoder_train(o, initial_state=s)
        else:
            finished1, in1, s1 = self.decoder_eval.initialize(
                self.dec_embeddings.variables[0],
                start_tokens=tf.fill([x.shape[0]], self.v2i["<GO>"]),
                end_tokens=self.v2i["EOS"],
            )
            for i in range(self.max_pred_len):
                o, h, c = self.decoder_eval(o, initial_state=s)
        logits = self.dense(o)
        return logits, (h, c)

    def loss(self, x, y, training=None):
        logits = self.call(x, training=training)
        _loss = self.cross_entropy(y, logits)
        return tf.reduce_mean(_loss)

    def step(self, x, y):
        with tf.GradientTape() as tape:
            _loss: tf.Tensor = self.loss(x, y, True)
            grads = tape.gradient(_loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return _loss.numpy()


# get and process data
data = utils.DateData()
print("Chinese time order: ", data.date_cn[:3], "\nEnglish time order: ", data.date_en[:3])
print("vocabularies: ", data.vocab)
print("x index sample: \n", data.x[:2], "\ny index sample: \n", data.y[:2])

model = Seq2Seq(
    data.num_word, data.num_word, 16, 16,
    max_pred_len=20, start_token=data.start_token, end_token=data.end_token, v2i=data.v2i)

# training
for t in range(2500):
    bx, by, decoder_len = data.sample(8)
    loss = model.step(bx, by)
    if t % 200 == 0:
        print("step: {} | loss: {}".format(t, loss))



# net
tfx = tf.placeholder(tf.int32, [None, None])    # [n, time_step]
tfy = tf.placeholder(tf.int32, [None, None])    # [n, time_step]

embeddings = tf.Variable(tf.random_normal((len(vocab), 16), mean=0., stddev=0.01))      # [n_vocab, 16]
x_embedded = tf.nn.embedding_lookup(embeddings, tfx)                                    # [n, step, 16]

# encoding
encoder_cell = tf.nn.rnn_cell.LSTMCell(UNITS)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    cell=encoder_cell, inputs=x_embedded, sequence_length=None,
    initial_state=encoder_cell.zero_state(tf.shape(tfx)[0], dtype=tf.float32),
    time_major=False)


# decoding for training and inference
def decoding(helper):
    # Decoder
    decoder_cell = tf.nn.rnn_cell.LSTMCell(UNITS)
    projection_layer = tf.layers.Dense(len(vocab))
    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, encoder_state,
        output_layer=projection_layer)
    # Dynamic decoding
    outputs = tf.contrib.seq2seq.dynamic_decode(
        decoder, output_time_major=False, impute_finished=False, maximum_iterations=20)[0]
    logits = outputs.rnn_output
    return logits


# decoding in training
with tf.variable_scope("decoder"):
    decoder_lengths = tf.placeholder(tf.int32, [None, ])
    y_embedded = tf.nn.embedding_lookup(embeddings, tfy[:, :-1])  # [n, step, 16]
    train_logits = decoding(helper=tf.contrib.seq2seq.TrainingHelper(
        inputs=y_embedded, sequence_length=decoder_lengths, time_major=False))

# decoding in inference
with tf.variable_scope("decoder", reuse=True):
    infer_logits = decoding(helper=tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embedding=embeddings, start_tokens=tf.fill([tf.shape(tfx)[0]], v2i["<GO>"]), end_token=v2i["<EOS>"]))

# loss and training
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tfy[:, 1:], logits=train_logits)
loss = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer(0.05)
# clipping gradients
gradients = opt.compute_gradients(loss)
clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
train_op = opt.apply_gradients(clipped_gradients)

sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': 0}))
sess.run(tf.global_variables_initializer())

for t in range(1000):
    bi = np.random.randint(0, len(x), size=64)
    bx, by = x[bi], y[bi]
    decoder_len = np.full((len(bx),), by.shape[1]-1)
    _, loss_ = sess.run([train_op, loss], {tfx: bx, tfy: by, decoder_lengths: decoder_len})
    if t % 10 == 0:
        logits_ = sess.run(infer_logits, {tfx: bx[:1, :]})
        target = "".join([i2v[i] for i in by[0, 1:-1]])
        res = []
        for i in np.argmax(logits_[0], axis=1):
            if i == v2i["<EOS>"]:
                break
            res.append(i2v[i])
        res = "".join(res)
        print(
            "step: ", t,
            "| loss: %.3f" % loss_,
            "| target: ", target,
            "| inference: ", res,
        )