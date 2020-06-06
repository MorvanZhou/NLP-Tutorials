"""
reference:
https://zhuanlan.zhihu.com/p/49271699
https://jalammar.github.io/illustrated-bert/
"""

import tensorflow as tf
from tensorflow import keras
import utils
import time
from GPT import GPT
import pickle
import os

MODEL_DIM = 128
N_LAYER = 4
BATCH_SIZE = 12
LEARNING_RATE = 1e-4


class BERT(GPT):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, max_seg=3, drop_rate=0.1, padding_idx=0):
        super().__init__(model_dim, max_len, n_layer, n_head, n_vocab, max_seg, drop_rate, padding_idx)

    def mask(self, seqs):
        """
         abcd--
        b010011
        c001011
        d000111
        -000011
        -000001

        the mask shows below will see itself by embeddings in the encoder residual layer.
        # o1 = self.bn[0](attn + xz, training) #
         abcd--
        a100011
        b010011
        c001011
        d000101
        -000011
        -000001
        """
        eye = tf.eye(self.max_len+1, batch_shape=[len(seqs)], dtype=tf.float32)[:, 1:, :-1]
        pad = tf.math.equal(seqs, self.padding_idx)
        mask = tf.where(pad[:, tf.newaxis, tf.newaxis, :], 1, eye[:, tf.newaxis, :, :])
        return mask  # [n, 1, step, step]


def main():
    # get and process data
    data = utils.MRPCData("./MRPC", rows=2000)
    print("num word: ", data.num_word)
    model = BERT(
        model_dim=MODEL_DIM, max_len=data.max_len-1, n_layer=N_LAYER, n_head=4, n_vocab=data.num_word,
        max_seg=data.num_seg, drop_rate=0.1, padding_idx=data.pad_id)
    t0 = time.time()
    for t in range(5000):
        seqs, segs, xlen, nsp_labels = data.sample(BATCH_SIZE)
        loss, pred = model.step(seqs[:, :-1], segs[:, :-1], seqs[:, 1:], nsp_labels)
        if t % 50 == 0:
            pred = pred[0].numpy().argmax(axis=1)
            t1 = time.time()
            print(
                "\n\nstep: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.3f" % loss.numpy(),
                "\n| tgt: ", " ".join([data.i2v[i] for i in seqs[0, 1:][:xlen[0].sum()+1]]),
                "\n| prd: ", " ".join([data.i2v[i] for i in pred[:xlen[0].sum()+1]]),
                )
            t0 = t1
    os.makedirs("./visual_helper/bert", exist_ok=True)
    model.save_weights("./visual_helper/bert/model.ckpt")


def export_attention():
    data = utils.MRPCData("./MRPC", rows=2000)
    print("num word: ", data.num_word)
    model = BERT(
        model_dim=MODEL_DIM, max_len=data.max_len-1, n_layer=N_LAYER, n_head=4, n_vocab=data.num_word,
        max_seg=data.num_seg, drop_rate=0.1, padding_idx=data.pad_id)
    model.load_weights("./visual_helper/bert/model.ckpt")

    # save attention matrix for visualization
    seqs, segs, xlen, nsp_labels = data.sample(32)
    model(seqs[:, :-1], segs[:, :-1], False)
    data = {"src": [[data.i2v[i] for i in seqs[j]] for j in range(len(seqs))], "attentions": model.attentions}
    with open("./visual_helper/bert_attention_matrix.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
    export_attention()

