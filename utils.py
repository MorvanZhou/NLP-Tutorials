import numpy as np
import datetime
import matplotlib.pyplot as plt


def sub_sampling(int_words, threshold=1e-5):
    # drop_p = 1 - sqrt(threshold/f(w))
    vocab, counts = np.unique(int_words, return_counts=True)
    total_count = len(int_words)
    frequency = counts / total_count
    p_keep = {v: np.sqrt(threshold / frequency[i]) for i, v in enumerate(vocab)}    # p_drop = 1 - p_keep
    kept_words = [word for word in int_words if np.random.random() < p_keep[word]]
    return kept_words


def get_date_data(n=5000):
    date_cn = []
    date_en = []
    for timestamp in np.random.randint(143835585, 2043835585, n):
        date = datetime.datetime.fromtimestamp(timestamp)
        date_cn.append(date.strftime("%y-%m-%d"))
        date_en.append(date.strftime("%d/%b/%Y"))
    vocab = set([str(i) for i in range(0, 10)] + ["-", "/", "<GO>", "<EOS>"] + [i.split("/")[1] for i in date_en])
    v2i = {v: i for i, v in enumerate(vocab)}
    i2v = {i: v for v, i in v2i.items()}
    x, y = [], []
    for cn, en in zip(date_cn, date_en):
        x.append([v2i[v] for v in cn])
        y.append(
            [v2i["<GO>"], ] + [v2i[v] for v in en[:3]] + [v2i[en[3:6]], ] + [v2i[v] for v in en[6:]] + [v2i["<EOS>"], ])
    x, y = np.array(x), np.array(y)
    return vocab, x, y, v2i, i2v, date_cn, date_en


def plot_attention(i2v, sample_x, sample_y, alignments):
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        x_vocab = [i2v[j] for j in np.ravel(sample_x[i])]
        y_vocab = [i2v[j] for j in sample_y[i, 1:]]
        plt.imshow(alignments[i], cmap="YlGn", vmin=0., vmax=1.)
        plt.yticks([j for j in range(len(y_vocab))], y_vocab)
        plt.xticks([j for j in range(len(x_vocab))], x_vocab)
        if i == 0 or i == 4:
            plt.ylabel("Output")
        if i > 3:
            plt.xlabel("Input")
    plt.tight_layout()
    plt.show()
