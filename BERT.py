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

tf.random.set_seed(1)
np.random.seed(1)

MODEL_DIM = 128
N_LAYER = 6
BATCH_SIZE = 5
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
            mlm_loss_batch = self.cross_entropy(seqs_, mlm_logits) * loss_mask
            mlm_loss = tf.reduce_mean(mlm_loss_batch)
            nsp_loss_batch = self.cross_entropy(nsp_labels, nsp_logits)
            nsp_loss = tf.reduce_mean(nsp_loss_batch)
            loss = mlm_loss + nsp_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy(), mlm_logits.numpy().argmax(axis=2)

    def input_emb(self, seqs, segs):
        return self.word_emb(seqs) + self.segment_emb(segs) + tf.matmul(
            self.position_space, self.position_emb)  # [n, step, dim]

    def pad_mask(self, seqs):
        _seqs = tf.cast(tf.math.equal(seqs, self.padding_idx), tf.float32)    # seg = 2 is padding
        return _seqs[:, tf.newaxis, tf.newaxis, :]  # (n, 1, 1, step)

    @property
    def attentions(self):
        attentions = [l.multi_head.attention.numpy() for l in self.encoder.ls]
        return attentions


def do_mask(seq, len_arange, pad_id, mask_id):
    rand_id = np.random.choice(len_arange, size=max(1, int(MASK_RATE*len(len_arange))), replace=False)
    loss_mask = np.full_like(seq, pad_id)
    loss_mask[rand_id] = 1
    seq[rand_id] = mask_id
    return loss_mask


def do_replace(seq, len_arange, pad_id, word_ids):
    rand_id = np.random.choice(len_arange, size=max(1, int(MASK_RATE*len(len_arange))), replace=False)
    loss_mask = np.full_like(seq, pad_id)
    loss_mask[rand_id] = 1
    seq[rand_id] = np.random.choice(word_ids, size=len(rand_id))
    return loss_mask


def random_mask_or_replace(data):
    seqs, segs, xlen, nsp_labels = data.sample(BATCH_SIZE)
    seqs_ = seqs.copy()
    arange = np.arange(0, seqs.shape[1])
    if np.random.random() > 0.3:
        # mask
        loss_mask = np.vstack(
            [do_mask(
                seqs[i],
                np.concatenate((arange[:xlen[i, 0]], arange[xlen[i, 0] + 1:xlen[i].sum() + 1])),
                data.pad_id,
                data.v2i["<MASK>"]) for i in range(len(seqs))])
    else:
        # replace
        loss_mask = np.vstack(
            [do_replace(
                seqs[i],
                np.concatenate((arange[:xlen[i, 0]], arange[xlen[i, 0] + 1:xlen[i].sum() + 1])),
                data.pad_id,
                data.word_ids) for i in range(len(seqs))])
    return seqs, segs, seqs_, loss_mask, xlen, nsp_labels


def main():
    # get and process data
    data = utils.MRPCData4BERT("./MRPC")
    model = BERT(
        model_dim=MODEL_DIM, max_len=data.max_len, n_layer=N_LAYER, n_head=4, n_vocab=data.num_word,
        max_seg=data.num_seg, drop_rate=0.1, padding_idx=data.v2i["<PAD>"])
    t0 = time.time()
    for t in range(20000):
        seqs, segs, seqs_, loss_mask, xlen, nsp_labels = random_mask_or_replace(data)
        loss, pred = model.step(seqs, segs, seqs_, loss_mask, nsp_labels)
        if t % 20 == 0:
            t1 = time.time()
            print(
                "\n\nstep: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.3f" % loss,
                "\n| tgt: ", " ".join([data.i2v[i] for i in seqs[0][:xlen[0].sum()+1]]),
                "\n| prd: ", " ".join([data.i2v[i] for i in pred[0][:xlen[0].sum()+1]]),
                "\n| tgt word: ", [data.i2v[i] for i in seqs_[0]*loss_mask[0] if i != data.v2i["<PAD>"]],
                "\n| prd word: ", [data.i2v[i] for i in pred[0]*loss_mask[0] if i != data.v2i["<PAD>"]],
                )
            t0 = t1

    model.save_weights("./visual_helper/bert.ckpt")
    # save attention matrix for visualization
    # attentions, cls_logits_, seq_logits_ = model.sess.run([model.attentions, model.cls_logits, model.seq_logits], {model.tfx: bx, model.training: False})
    # data = {"src": bx, "label": cls_logits_.argmax(axis=1), "seq": seq_logits_.argmax(axis=-1), "attentions": attentions, "i2v": i2v}
    # import pickle
    # with open("./visual_helper/GPT_attention_matrix.pkl", "wb") as f:
    #     pickle.dump(data, f)


if __name__ == "__main__":
    main()


