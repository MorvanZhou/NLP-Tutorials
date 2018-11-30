import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools

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

    # alphabets
    "a t g q e h u f",
    "e q y u o i p s",
    "q o p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e e a d",
    "o p d g s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o l j s",
    "y g i s h k j l f r f",
]

all_words = [sentence.split(" ") for sentence in corpus]
all_words = np.array(list(itertools.chain(*all_words)))
# vocab sort by decreasing frequency for the negative sampling below (nce_loss).
vocab, v_count = np.unique(all_words, return_counts=True)
vocab = vocab[np.argsort(v_count)[::-1]]

# plot the default tf.nn.log_uniform_candidate_sampler in nce_loss
# pk = lambda k: (np.log(k + 2) - np.log(k + 1)) / np.log(100 + 1)
# plt.plot(np.arange(100), pk(np.arange(100)))
# plt.title("Sampling probability for the negative sampling")
# plt.show()

print("all vocabularies sorted from more frequent to less frequent:\n", vocab)
v2i = {v: i for i, v in enumerate(vocab)}
i2v = {i: v for v, i in v2i.items()}

# pair data
pairs = []
skip_window = 2
js = list(range(-skip_window, skip_window+1))
js.remove(0)    # remove center
for c in corpus:
    words = c.split(" ")
    w_idx = [v2i[w] for w in words]
    for i in range(len(w_idx)):
        for j in js:
            if i + j < 0 or i + j >= len(w_idx):
                continue
            pairs.append((w_idx[i], w_idx[i+j]))  # (center, context) or (feature, target)
pairs = np.array(pairs)
print("5 example pairs:\n", pairs[:5])
x, y = pairs[:, 0], pairs[:, 1]

# inputs
tfx = tf.placeholder(tf.int32, [None, ])
tfy = tf.placeholder(tf.int32, [None, ])

emb_dim = 2
embeddings = tf.Variable(tf.random_uniform((len(vocab), emb_dim), -1., 1.))     # [n_vocab, emb_dim]
embedded = tf.nn.embedding_lookup(embeddings, tfx)                              # [n, emb_dim]

# noise-contrastive estimation
nce_w = tf.Variable(tf.random_uniform((len(vocab), emb_dim), -1., 1.))          # [n_vocab, emb_dim]
nce_b = tf.Variable(tf.constant(0.1, shape=(len(vocab),)))                      # [n_vocab, ]

# negative sampling: take one positive label and num_sampled negative labels to compute the loss
# in order to reduce the computation of full softmax
loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nce_w, biases=nce_b, labels=tf.expand_dims(tfy, axis=1),
                 inputs=embedded, num_sampled=5, num_classes=len(vocab)))

train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
sess.run(tf.global_variables_initializer())

# training
for i in range(3000):
    b_idx = np.random.randint(0, len(x), 8)
    _, loss_ = sess.run([train_op, loss], {tfx: x[b_idx], tfy: y[b_idx]})
    if i % 200 == 0:
        print("loss:", loss_)

# plotting
word_emb = sess.run(embeddings)
for i in range(len(i2v)):
    bc = "blue"
    try:
        int(i2v[i])
    except ValueError:
        bc = "red"
    plt.text(word_emb[i, 0], word_emb[i, 1], s=i2v[i], color="w", weight="bold", backgroundcolor=bc)
plt.xlim(word_emb[:, 0].min()-.5, word_emb[:, 0].max()+.5)
plt.ylim(word_emb[:, 1].min()-.5, word_emb[:, 1].max()+.5)
plt.xticks(())
plt.yticks(())
plt.show()