import matplotlib.pyplot as plt
import numpy as np
import pickle


def attention_matrix():
    with open("./bert_attention_matrix.pkl", "rb") as f:
        data = pickle.load(f)
    src = data["src"]
    attentions = data["attentions"]

    encoder_atten = attentions["encoder"]
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    s_len = 0
    for s in src:
        if s != "<PAD>":
            s_len += 1

    plt.figure(0, (14, 7))
    for j in range(2):
        plt.subplot(1, 2, j + 1)
        plt.imshow(encoder_atten[-1].squeeze()[j][:s_len, :s_len], vmax=1, vmin=0, cmap="rainbow")
        plt.xticks(range(s_len), src[:s_len], rotation=90)
        plt.yticks(range(s_len), src[:s_len])
        plt.xlabel("head %i" % j)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("bert_encoder_self_attention.png", dpi=300)
    plt.show()


attention_matrix()
