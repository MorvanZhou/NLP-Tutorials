"""
reference:
https://zhuanlan.zhihu.com/p/49271699
https://jalammar.github.io/illustrated-bert/
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import utils
import time
from transformer import Encoder
import pickle


tf.random.set_seed(1)
np.random.seed(1)

MODEL_DIM = 256
N_LAYER = 4
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
MASK_RATE = 0.15


class BERT(keras.Model):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, max_seg=3, drop_rate=0.1, padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        self.n_vocab = n_vocab
        self.max_len = max_len

        # I think task emb is not necessary for pretraining,
        # because the aim of all tasks is to train a universal sentence embedding
        # the body encoder is the same across all task, and the output layer defines each task.
        # finetuning replaces output layer and leaves the body encoder unchanged.

        # self.task_emb = keras.layers.Embedding(
        #     input_dim=n_task, output_dim=model_dim,  # [n_task, dim]
        #     embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        # )

        self.word_emb = keras.layers.Embedding(
            input_dim=n_vocab, output_dim=model_dim,  # [n_vocab, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )

        self.segment_emb = keras.layers.Embedding(
            input_dim=max_seg, output_dim=model_dim,  # [max_seg, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )
        self.position_emb = keras.layers.Embedding(
            input_dim=max_len, output_dim=model_dim,  # [step, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )
        self.position_emb = self.add_weight(
            name="pos", shape=[max_len, model_dim], dtype=tf.float32,
            initializer=keras.initializers.RandomNormal(0., 0.01))
        self.position_space = tf.ones((1, max_len, max_len))
        self.encoder = Encoder(n_head, model_dim, drop_rate, n_layer)
        self.o_mlm = keras.layers.Dense(n_vocab)
        self.o_nsp = keras.layers.Dense(2)

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self.opt = keras.optimizers.Adam(LEARNING_RATE)

    def __call__(self, seqs, segs, training=False):
        embed = self.input_emb(seqs, segs)  # [n, step, dim]
        z = self.encoder(embed, training=training, mask=self.pad_mask(seqs))
        mlm_logits = self.o_mlm(z)  # [n, step, n_vocab]
        nsp_logits = self.o_nsp(tf.reshape(z, [z.shape[0], -1]))  # [n, n_cls]
        return mlm_logits, nsp_logits

    def step(self, seqs, segs, seqs_, loss_mask, nsp_labels):
        with tf.GradientTape() as tape:
            mlm_logits, nsp_logits = self(seqs, segs, training=True)
            mlm_loss_batch = tf.boolean_mask(self.cross_entropy(seqs_, mlm_logits), loss_mask)
            mlm_loss = tf.reduce_mean(mlm_loss_batch)
            nsp_loss_batch = self.cross_entropy(nsp_labels, nsp_logits)
            nsp_loss = tf.reduce_mean(nsp_loss_batch)
            loss = mlm_loss + nsp_loss
            grads = tape.gradient(loss, self.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, mlm_logits

    def input_emb(self, seqs, segs):
        return self.word_emb(seqs) + self.segment_emb(segs) + tf.matmul(
            self.position_space, self.position_emb)  # [n, step, dim]

    def pad_mask(self, seqs):
        _seqs = tf.cast(tf.math.equal(seqs, self.padding_idx), tf.float32)    # seg = 2 is padding
        return _seqs[:, tf.newaxis, tf.newaxis, :]  # (n, 1, 1, step)

    @property
    def attentions(self):
        attentions = [l.mh.attention.numpy() for l in self.encoder.ls]
        return attentions


def _get_loss_mask(len_arange, seq, pad_id):
    rand_id = np.random.choice(len_arange, size=max(2, int(MASK_RATE * len(len_arange))), replace=False)
    loss_mask = np.full_like(seq, pad_id, dtype=np.bool)
    loss_mask[rand_id] = True
    return loss_mask[None, :], rand_id


def do_mask(seq, len_arange, pad_id, mask_id):
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    seq[rand_id] = mask_id
    return loss_mask


def do_replace(seq, len_arange, pad_id, word_ids):
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    seq[rand_id] = np.random.choice(word_ids, size=len(rand_id))
    return loss_mask


def do_nothing(seq, len_arange, pad_id):
    loss_mask, _ = _get_loss_mask(len_arange, seq, pad_id)
    return loss_mask


def random_mask_or_replace(data, arange):
    seqs, segs, xlen, nsp_labels = data.sample(BATCH_SIZE)
    seqs_ = seqs.copy()
    p = np.random.random()
    if p < 0.7:
        # mask
        loss_mask = np.concatenate(
            [do_mask(
                seqs[i],
                np.concatenate((arange[:xlen[i, 0]], arange[xlen[i, 0] + 1:xlen[i].sum() + 1])),
                data.pad_id,
                data.v2i["<MASK>"]) for i in range(len(seqs))], axis=0)
    elif p < 0.85:
        # do nothing
        loss_mask = np.concatenate(
            [do_nothing(
                seqs[i],
                np.concatenate((arange[:xlen[i, 0]], arange[xlen[i, 0] + 1:xlen[i].sum() + 1])),
                data.pad_id) for i in range(len(seqs))], axis=0)
    else:
        # replace
        loss_mask = np.concatenate(
            [do_replace(
                seqs[i],
                np.concatenate((arange[:xlen[i, 0]], arange[xlen[i, 0] + 1:xlen[i].sum() + 1])),
                data.pad_id,
                data.word_ids) for i in range(len(seqs))], axis=0)
    return seqs, segs, seqs_, loss_mask, xlen, nsp_labels


def main():
    # get and process data
    data = utils.MRPCData4BERT("./MRPC")
    print("num word: ", data.num_word)
    model = BERT(
        model_dim=MODEL_DIM, max_len=data.max_len, n_layer=N_LAYER, n_head=4, n_vocab=data.num_word,
        max_seg=data.num_seg, drop_rate=0.1, padding_idx=data.v2i["<PAD>"])
    t0 = time.time()
    arange = np.arange(0, data.max_len)
    for t in range(20):
        seqs, segs, seqs_, loss_mask, xlen, nsp_labels = random_mask_or_replace(data, arange)
        loss, pred = model.step(seqs, segs, seqs_, loss_mask, nsp_labels)
        if t % 20 == 0:
            pred = pred[0].numpy().argmax(axis=1)
            t1 = time.time()
            print(
                "\n\nstep: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.3f" % loss.numpy(),
                "\n| tgt: ", " ".join([data.i2v[i] for i in seqs[0][:xlen[0].sum()+1]]),
                "\n| prd: ", " ".join([data.i2v[i] for i in pred[:xlen[0].sum()+1]]),
                "\n| tgt word: ", [data.i2v[i] for i in seqs_[0]*loss_mask[0] if i != data.v2i["<PAD>"]],
                "\n| prd word: ", [data.i2v[i] for i in pred*loss_mask[0] if i != data.v2i["<PAD>"]],
                )
            t0 = t1

    model.save_weights("./visual_helper/bert/model.ckpt")
    # save attention matrix for visualization
    src_seq = "02-11-30"
    print("src: ", src_seq, "\nprediction: ", model.translate(src_seq, data.v2i, data.i2v))

    # save attention matrix for visualization
    test_seq = seqs[:1, segs[0] == 0]
    _ = model(test_seq, np.zeros_like(test_seq), training=False)

    data = {"src": [data.i2v[i] for i in data.x[0]], "tgt": [data.i2v[i] for i in data.y[0]],
            "attentions": model.attentions}
    with open("./visual_helper/bert_attention_matrix.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()


