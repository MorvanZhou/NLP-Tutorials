import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.pyplot import cm

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
    plt.savefig("transformer_pad_mask.png", dpi=200)
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
    plt.savefig("transformer_output_mask.png", dpi=200)
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
    plt.savefig("transformer_position_embedding.png", dpi=200)
    plt.show()


def attention_matrix():
    with open("./transformer_attention_matrix.pkl", "rb") as f:
        data = pickle.load(f)
    src = data["src"]
    tgt = data["tgt"]
    attentions = data["attentions"]

    encoder_atten = attentions["encoder"]
    decoder_tgt_atten = attentions["decoder"]["mh1"]
    decoder_src_atten = attentions["decoder"]["mh2"]
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
                plt.ylabel("layer %i" % (i+1))
            if i == 2:
                plt.xlabel("head %i" % (j+1))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("transformer_encoder_self_attention.png", dpi=200)
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
                plt.ylabel("layer %i" % (i+1))
            if i == 2:
                plt.xlabel("head %i" % (j+1))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("transformer_decoder_self_attention.png", dpi=200)
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
                plt.ylabel("layer %i" % (i+1))
            if i == 2:
                plt.xlabel("head %i" % (j+1))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("transformer_decoder_encoder_attention.png", dpi=200)
    plt.show()


def self_attention(bert_or_gpt="bert", case=0):
    with open(bert_or_gpt+"_attention_matrix.pkl", "rb") as f:
        data = pickle.load(f)
    src = data["src"]
    attentions = data["attentions"]

    encoder_atten = attentions["encoder"]
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    s_len = 0
    for s in src[case]:
        if s == "<SEP>":
            break
        s_len += 1

    plt.figure(0, (7, 28))
    for j in range(4):
        plt.subplot(4, 1, j + 1)
        img = encoder_atten[-1][case, j][:s_len-1, :s_len-1]
        plt.imshow(img, vmax=img.max(), vmin=0, cmap="rainbow")
        plt.xticks(range(s_len-1), src[case][:s_len-1], rotation=90, fontsize=9)
        plt.yticks(range(s_len-1), src[case][1:s_len], fontsize=9)
        plt.xlabel("head %i" % (j+1))
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()
    plt.savefig(bert_or_gpt+"_self_attention.png", dpi=500)
    # plt.show()


def self_attention_line(bert_or_gpt="bert", case=0):
    with open(bert_or_gpt+"_attention_matrix.pkl", "rb") as f:
        data = pickle.load(f)
    src = data["src"]
    attentions = data["attentions"]

    encoder_atten = attentions["encoder"]

    s_len = 0
    print(" ".join(src[case]))
    for s in src[case]:
        if s == "<SEP>":
            break
        s_len += 1
    y_label = src[case][:s_len][::-1]
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(7, 14))

    for i in range(2):
        for j in range(2):
            ax[i, j].set_yticks(np.arange(len(y_label)))
            ax[i, j].tick_params(labelright=True)
            ax[i, j].set_yticklabels(y_label, fontsize=9)     # input

            img = encoder_atten[-1][case, i+j][:s_len - 1, :s_len - 1]
            color = cm.rainbow(np.linspace(0, 1, img.shape[0]))
            for row, c in zip(range(img.shape[0]), color):
                for col in range(img.shape[1]):
                    alpha = (img[row, col] / img[row].max()) ** 7
                    ax[i, j].plot([1, 0], [img.shape[0]-row-1, img.shape[1]-col], alpha=alpha, c=c)
            ax[i, j].set_xticks(())
            ax[i, j].set_xlabel("head %i" % (j+1+i))
            ax[i, j].set_xlim(0, 1)
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()
    plt.savefig(bert_or_gpt+"_self_attention_line%i.png" % case, dpi=100)


# self_mask(padded_id_seqs)
# output_mask(padded_id_seqs)
# position_embedding()
# attention_matrix()
# self_attention("bert", case=2)
self_attention_line("bert", case=1)
# self_attention_line("gpt", case=4)
