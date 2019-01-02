import matplotlib.pyplot as plt
import pickle


def attention_matrix():
    with open("./GPT_attention_matrix.pkl", "rb") as f:
        data = pickle.load(f)
    i2v = data["i2v"]
    src = [i2v[i] for i in data["src"][-1] if i != 0]
    attentions = data["attentions"][-1]
    decoder_src_atten = attentions
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    plt.figure(1, (12, 21))
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, i * 3 + j + 1)
            plt.imshow(decoder_src_atten[i].squeeze()[j][:len(src), :len(src)], vmax=1, vmin=0, cmap="rainbow")
            plt.xticks(range(len(src)), src, rotation=90, fontsize=6)
            plt.yticks(range(len(src)), src, fontsize=6)
            if j == 0:
                plt.ylabel("layer %i" % i)
            if i == 2:
                plt.xlabel("head %i" % j)
    plt.tight_layout()
    plt.show()


attention_matrix()