import numpy as np
from collections import Counter
import itertools

docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
]


def get_tf(docs, method="log"):
    def safe_log(x):
        mask = x != 0
        x[mask] = np.log10(x[mask])
        return x

    # different tf computation
    def log_tf(tf): return 1 + safe_log(tf)

    def augmented_tf(tf): return 0.5 + 0.5 * tf / np.max(tf, axis=1, keepdims=True)

    def boolean_tf(tf): return np.minimum(tf, 1)

    def log_avg_tf(tf): return (1 + safe_log(tf)) / (1 + safe_log(np.mean(tf, axis=1, keepdims=True)))

    docs_words = [d.replace(",", "").split(" ") for d in docs]
    vocab = set(itertools.chain(*docs_words))
    v2i = {v: i for i, v in enumerate(vocab)}
    i2v = {i: v for v, i in v2i.items()}

    # term frequency
    tf = np.zeros((len(vocab), len(docs)), dtype=np.float64)    # [n_vocab, n_doc]
    for i, d in enumerate(docs_words):
        counter = Counter(d)
        for v in counter.keys():
            tf[v2i[v], i] = counter[v]

    if method == "log":
        weighted_tf = log_tf
    elif method == "augmented":
        weighted_tf = augmented_tf
    elif method == "boolean":
        weighted_tf = boolean_tf
    elif method == "log_avg":
        weighted_tf = log_avg_tf
    else:
        raise ValueError
    return weighted_tf(tf), docs_words, v2i, i2v,


# inverse document frequency
def get_idf(docs_words, i2v, method="log"):
    # different idf computation
    def log_idf(df): return np.log10(len(docs) / df)

    def prob_idf(df): return np.maximum(0, np.log((len(docs) - df) / df))

    def length_norm(x): return x / np.sum(np.square(x))

    df = np.zeros((len(i2v),))
    for i in range(len(i2v)):
        d_count = 0
        for d in docs_words:
            d_count += 1 if i2v[i] in d else 0
        df[i] = d_count

    if method == "log":
        idf = log_idf
    elif method == "prob":
        idf = prob_idf
    elif method == "len_norm":
        idf = length_norm
    else:
        raise ValueError
    return idf(df)


def cosine_similarity(q, ds):
    unit_q = q / np.sqrt(np.sum(np.square(q), axis=0, keepdims=True))
    unit_ds = ds / np.sqrt(np.sum(np.square(ds), axis=0, keepdims=True))
    similarity = unit_ds.T.dot(unit_q).ravel()
    return similarity


def docs_score(q, d_tf, d_idf, v2i, len_norm=False):
    q_words = q.replace(",", "").split(" ")

    # add unknown words
    unknown_v = 0
    for v in set(q_words):
        if v not in v2i:
            v2i[v] = len(v2i)
            i2v[len(v2i)-1] = v
            unknown_v += 1
    if unknown_v > 0:
        d_tf = np.concatenate((d_tf, np.zeros((unknown_v, d_tf.shape[1]), dtype=np.float)), axis=0)
        d_idf = np.concatenate((d_idf, np.zeros((unknown_v, ), dtype=np.float)), axis=0)

    counter = Counter(q_words)
    q_tf = np.zeros((len(d_tf), 1), dtype=np.float)     # [n, 1]
    for v in counter.keys():
        q_tf[v2i[v], 0] = counter[v]

    d_idf = d_idf[:, None]          # [n, 1]
    q_vec = q_tf * d_idf            # [n, 1]

    d_matrix = tf * d_idf       # [n, doc]
    scores = cosine_similarity(q_vec, d_matrix)
    if len_norm:
        len_docs = [len(d) for d in docs_words]
        scores = scores / np.array(len_docs)
    return scores


tf, docs_words, v2i, i2v = get_tf(docs)
idf = get_idf(docs_words, i2v)

# test
q = "I like coffee"
scores = docs_score(q, tf, idf, v2i)
d_ids = scores.argsort()[-3:][::-1]
print("top 3 docs:\n", [docs[i] for i in d_ids])