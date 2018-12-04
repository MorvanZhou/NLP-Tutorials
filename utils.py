import numpy as np
import datetime


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


def generate_date(n=5000):
    date_cn = []
    date_en = []
    for timestamp in np.random.randint(1043835585, 1643835585, n):
        date = datetime.datetime.fromtimestamp(timestamp)
        date_cn.append(date.strftime("%Y-%m-%d"))
        date_en.append(date.strftime("%d/%b/%Y"))
    return date_cn, date_en