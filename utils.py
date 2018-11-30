import numpy as np


def pre_process(doc):
    vocab, v2i, i2v = None, None, None
    return vocab, v2i, i2v


def sub_sampling(int_words, threshold=1e-5):
    # drop_p = 1 - sqrt(threshold/f(w))
    vocab, counts = np.unique(int_words, return_counts=True)
    total_count = len(int_words)
    frequency = counts / total_count
    p_keep = {v: np.sqrt(threshold / frequency[i]) for i, v in enumerate(vocab)}    # p_drop = 1 - p_keep
    kept_words = [word for word in int_words if np.random.random() < p_keep[word]]
    return kept_words
