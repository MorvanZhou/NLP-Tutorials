import matplotlib.pyplot as plt
import numpy as np
import pickle

SEQS = ["I love you", "My name is M", "This is a very long seq", "Short one"]
vocabs = set((" ".join(SEQS)).split(" "))
i2v = {i: v for i, v in enumerate(vocabs, start=1)}
i2v["<PAD>"] = 0     # add 0 idx for <PAD>
v2i = {v: i for i, v in i2v.items()}

id_seqs = [[v2i[v] for v in seq.split(" ")] for seq in SEQS]
padded_id_seqs = np.array([l + [0] * (6-len(l)) for l in id_seqs])


def pad_mask(seqs):
    mask = np.where(seqs == 0, np.zeros_like(seqs), np.ones_like(seqs))  # 0 idx is padding
    mask = np.expand_dims(mask, axis=1) * np.expand_dims(mask, axis=2)  # [n, step, step]
    print(mask)
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    for i in range(1, 5):
        plt.subplot(2, 2, i)
        plt.imshow(mask[i-1], vmax=1, vmin=0, cmap="YlGn")
        plt.xticks(range(6), SEQS[i - 1].split(" "), rotation=45)
        plt.yticks(range(6), SEQS[i - 1].split(" "),)
        plt.grid(which="minor", c="w", lw=0.5, linestyle="-")
    plt.tight_layout()
    plt.show()


def output_mask(seqs):
    max_len = 6
    pmask = np.where(seqs == 0, np.zeros_like(seqs), np.ones_like(seqs))  # 0 idx is padding
    pmask = np.expand_dims(pmask, axis=1) * np.expand_dims(pmask, axis=2)  # [n, step, step]
    mask = ~np.triu(np.ones((max_len, max_len), dtype=np.bool), 1)
    mask = np.tile(np.expand_dims(mask, axis=0), [np.shape(seqs)[0], 1, 1])  # [n, step, step]
    omask = np.where(mask, pmask, np.zeros_like(pmask))

    print(mask)
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    for i in range(1, 5):
        plt.subplot(2, 2, i)
        plt.imshow(omask[i - 1], vmax=1, vmin=0, cmap="YlGn")
        plt.xticks(range(6), SEQS[i - 1].split(" "), rotation=45)
        plt.yticks(range(6), SEQS[i - 1].split(" "), )
        plt.grid(which="minor", c="w", lw=0.5, linestyle="-")
    plt.tight_layout()
    plt.show()


def position_embedding():
    max_len = 500
    model_dim = 512
    pos = np.arange(max_len)[:, None]
    pe = pos / np.power(10000, 2. * np.arange(model_dim)[None, :] / model_dim)  # [max_len, model_dim]
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    plt.imshow(pe, vmax=1, vmin=-1, cmap="rainbow")
    plt.ylabel("word position")
    plt.xlabel("embedding dim")
    plt.show()


def attention_matrix():
    with open("attention_matrix.pkl", "rb") as f:
        data = pickle.load(f)
    src = data["src"]
    tgt = data["tgt"]
    attentions = data["attentions"]

    encoder_atten = attentions[:3]
    decoder_tgt_atten = attentions[3::2]
    decoder_src_atten = attentions[4::2]
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    plt.figure(0, (7, 7))
    plt.suptitle("Encoder self-attention")
    for i in range(3):
        for j in range(4):
            plt.subplot(3, 4, i * 4 + j + 1)
            plt.imshow(encoder_atten[i].squeeze()[j][:len(src), :len(src)], vmax=1, vmin=0, cmap="rainbow")
            plt.xticks(range(len(src)), src)
            plt.yticks(range(len(src)), src)
            if j == 0:
                plt.ylabel("layer %i" % i)
            if i == 2:
                plt.xlabel("head %i" % j)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    plt.figure(1, (7, 7))
    plt.suptitle("Decoder self-attention")
    for i in range(3):
        for j in range(4):
            plt.subplot(3, 4, i * 4 + j + 1)
            plt.imshow(decoder_tgt_atten[i].squeeze()[j][:len(tgt) - 1, :len(tgt) - 1], vmax=1, vmin=0, cmap="rainbow")
            plt.xticks(range(len(tgt)-1), tgt[:-1], rotation=45, fontsize=7)
            plt.yticks(range(len(tgt)-1), tgt[1:], rotation=45, fontsize=7)
            if j == 0:
                plt.ylabel("layer %i" % i)
            if i == 2:
                plt.xlabel("head %i" % j)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    plt.figure(2, (7, 7))
    plt.suptitle("Decoder-Encoder attention")
    for i in range(3):
        for j in range(4):
            plt.subplot(3, 4, i*4+j+1)
            plt.imshow(decoder_src_atten[i].squeeze()[j][:len(tgt)-1, :len(src)], vmax=1, vmin=0, cmap="rainbow")
            plt.xticks(range(len(src)), src, fontsize=7)
            plt.yticks(range(len(tgt)-1), tgt[1:], rotation=45, fontsize=7)
            if j == 0:
                plt.ylabel("layer %i" % i)
            if i == 2:
                plt.xlabel("head %i" % j)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


# pad_mask(padded_id_seqs)
# output_mask(padded_id_seqs)
# position_embedding()
attention_matrix()
