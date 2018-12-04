import tensorflow as tf
import numpy as np
from utils import generate_date


# get and process data
date_cn, date_en = generate_date()
print("Chinese time order: ", date_cn[:3], "\nEnglish time order: ", date_en[:3])
vocab = set([str(i) for i in range(0, 10)] + ["-", "/", "<GO>", "<EOS>"] + [i.split("/")[1] for i in date_en])
print("vocabularies: ", vocab)
v2i = {v: i for i, v in enumerate(vocab)}
i2v = {i: v for v, i in v2i.items()}
x, y = [], []
for cn, en in zip(date_cn, date_en):
    x.append([v2i[v] for v in cn])
    y.append([v2i["<GO>"], ] + [v2i[v] for v in en[:3]] + [v2i[en[3:6]], ] + [v2i[v] for v in en[6:]] + [v2i["<EOS>"], ])
x, y = np.array(x), np.array(y)
print(x[:2], "\n", y[:2])

# net
tfx = tf.placeholder(tf.int32, [None, None])
tfy = tf.placeholder(tf.int32, [None, None])

embeddings = tf.Variable(tf.random_normal((len(vocab), 16), mean=0., stddev=0.01))      # [n_vocab, 16]
x_embedded = tf.nn.embedding_lookup(embeddings, tfx)                                    # [n, step, 16]

# encoding
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(32)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    cell=encoder_cell, inputs=x_embedded, sequence_length=None,
    initial_state=encoder_cell.zero_state(tf.shape(tfx)[0], dtype=tf.float32),
    time_major=False)


# decoding for training and inference
def decoding(helper):
    # Decoder
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(32)
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
opt = tf.train.AdamOptimizer(0.01)
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
        logits_ = sess.run(infer_logits, {tfx: bx})
        inputs = "".join([i2v[i] for i in bx[0]])
        res = []
        for i in np.argmax(logits_[0], axis=1):
            if i == v2i["<EOS>"]:
                break
            res.append(i2v[i])
        res = "".join(res)
        print(
            "step: ", t,
            "| loss: %.3f" % loss_,
            "| target: ", inputs,
            "| inference: ", res,
        )