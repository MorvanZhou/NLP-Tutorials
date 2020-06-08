import matplotlib.pyplot as plt
import pickle
import numpy as np


def plot_attention(i2v, sample_x, sample_y, alignments):
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        x_vocab = [i2v[j] for j in np.ravel(sample_x[i])]
        y_vocab = [i2v[j] for j in sample_y[i, 1:]]
        plt.imshow(alignments[i], cmap="YlGn", vmin=0., vmax=1.)
        plt.yticks([j for j in range(len(y_vocab))], y_vocab)
        plt.xticks([j for j in range(len(x_vocab))], x_vocab)
        if i == 0 or i == 3:
            plt.ylabel("Output")
        if i >= 3:
            plt.xlabel("Input")
    plt.tight_layout()
    plt.savefig("attention.png", format="png", dpi=200)
    plt.show()


with open("./tmp/attention_align.pkl", "rb") as f:
    data = pickle.load(f)
    i2v, x, y, align = data["i2v"], data["x"], data["y"], data["align"]
plot_attention(i2v, x, y, align)