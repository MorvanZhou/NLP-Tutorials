import matplotlib.pyplot as plt
import numpy as np
import pickle


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
                plt.ylabel("layer %i" % i)
            if i == 2:
                plt.xlabel("head %i" % j)
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
                plt.ylabel("layer %i" % i)
            if i == 2:
                plt.xlabel("head %i" % j)
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
                plt.ylabel("layer %i" % i)
            if i == 2:
                plt.xlabel("head %i" % j)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("transformer_decoder_encoder_attention.png", dpi=200)
    plt.show()


# pad_mask(padded_id_seqs)
# output_mask(padded_id_seqs)
# position_embedding()
attention_matrix()
