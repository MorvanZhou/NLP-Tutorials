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


MODEL_DIM = 128
N_LAYER = 3
N_HEAD = 4
MAX_SEG = 3   # sentence 1, sentence 2, padding


class BERT(keras.Model):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, max_seg=3, drop_rate=0.1, padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        self.n_vocab = n_vocab

        self.word_emb = keras.layers.Embedding(
            input_dim=n_vocab, output_dim=model_dim,  # [n_vocab, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )
        # I think task emb is not necessary for pretraining,
        # because the aim of all tasks is to train a universal sentence embedding
        # the body encoder is the same across all task, and the output layer defines each task.
        # finetuning replaces output layer and leaves the body encoder unchanged.

        # self.task_emb = keras.layers.Embedding(
        #     input_dim=n_task, output_dim=model_dim,  # [n_task, dim]
        #     embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        # )
        self.segment_emb = keras.layers.Embedding(
            input_dim=max_seg, output_dim=model_dim,  # [max_seg, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )
        self.position_emb = keras.layers.Embedding(
            input_dim=max_len, output_dim=model_dim,  # [step, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )
        self.position_space = tf.expand_dims(tf.linspace(0., 1., max_len), axis=0)
        self.encoder = Encoder(n_head, model_dim, drop_rate, n_layer)

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.01)

    def get_embeddings(self, seg, seq):
        embed = self.word_emb(seq) + self.segment_emb(seg) + self.position_emb(self.position_space)    # [n, step, dim]
        return embed     # [n. step, dim]

    def random_mask(self, embed, mask_rate, seq):
        np_input_mask = np.random.rand(embed.shape[0], embed.shape[1]) > mask_rate  # [n, step]
        pad_filter = seq != self.padding_idx
        np_input_mask *= pad_filter
        np_target_mask = np.logical_not(np_input_mask)
        np_target_mask *= pad_filter
        return np_input_mask, np_target_mask

    def pad_mask(self, seqs):
        """
        pad mask will look like this:
        input|  T,1,2,P,P
        target|
        cls    [0,0,0,1,1]      - this line for predicting class
        1      [0,0,0,1,1]
        2      [0,0,0,1,1]
        P      [1,1,1,1,1]
        P      [1,1,1,1,1]
        """
        _seqs = tf.cast(tf.math.equal(seqs, self.padding_idx), tf.float32)    # seg = 2 is padding
        return _seqs[:, tf.newaxis, tf.newaxis, :]  # (n, 1, 1, step)

    @property
    def attentions(self):
        attentions = [l.multi_head.attention.numpy() for l in self.encoder.ls]
        return attentions


class BERTMLM(keras.Model):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.o_mlm = keras.layers.Dense(bert.n_vocab)

    def __call__(self, seg, seq, mask_rate=0.1, training=False):
        embed = self.bert.get_embeddings(seg, seq)   # [n. step, dim]

        # random mask
        np_input_mask, np_target_mask = self.bert.random_mask(embed, mask_rate, seq)
        input_mask = tf.expand_dims(tf.convert_to_tensor(np_input_mask.astype(np.float32)), axis=2)  # [n, step, 1]
        target_mask = tf.expand_dims(tf.convert_to_tensor(np_target_mask.astype(np.float32)), axis=2)  # [n, step, 1]

        masked_embed = embed * input_mask
        z = self.bert.encoder(masked_embed, training=training, mask=self.bert.pad_mask(seg))
        mlm_logits = self.o_mlm(z)      # [n, step, n_vocab]
        masked_logits = mlm_logits * target_mask
        inf = np.zeros(masked_logits.shape, dtype=np.float32)
        inf[:, :, 0] += 1e-9
        masked_logits += tf.convert_to_tensor(inf)         # make softmax to the first word index in vocab
        return masked_logits, np_target_mask

    def step(self, data, batch_size, mask_rate=0.1):
        seq, seg = data.sample_mlm(batch_size)
        with tf.GradientTape() as tape:
            mlm_logits, target_mask = self(seg, seq, mask_rate, training=True)
            target_words = seq * target_mask
            loss = self.bert.cross_entropy(target_words, mlm_logits)
        grads = tape.gradient(loss, self.trainable_variables)
        self.bert.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy(), mlm_logits.numpy().argmax(axis=2), target_words


class BERTNSP(keras.Model):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.o_nsp = keras.layers.Dense(2)

    def __call__(self, seg, seq, training=False):
        embed = self.bert.get_embeddings(seg, seq)  # [n. step, dim]
        z = self.bert.encoder(embed, training=training, mask=self.bert.pad_mask(seg))
        z = tf.reshape(z, shape=[z.shape[0], -1])
        logits = self.o_nsp(z)      # [n, n_cls]
        return logits

    def step(self, data, batch_size):
        seq, seg, label = data.sample_nsp(batch_size)
        with tf.GradientTape() as tape:
            nsp_logits = self(seg, seq, training=True)
            loss = self.bert.cross_entropy(label, nsp_logits)
        grads = tape.gradient(loss, self.trainable_variables)
        self.bert.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy()


def main():
    # get and process data
    data = utils.MRPCData4BERT("./MRPC")
    bert = BERT(
        model_dim=MODEL_DIM, max_len=data.max_len, n_layer=N_LAYER, n_head=N_HEAD, n_vocab=data.num_word,
        max_seg=MAX_SEG, drop_rate=0.1, padding_idx=data.v2i["<PAD>"])
    mlm_model = BERTMLM(bert)
    nsp_model = BERTNSP(bert)
    t0 = time.time()
    b_size = 64
    for t in range(20000):
        loss_mlm, masked_pred, masked_target = mlm_model.step(data, b_size, mask_rate=0.1)
        loss_nsp = nsp_model.step(data, b_size)
        if t % 10 == 0:
            t1 = time.time()
            print(
                    "\n\nstep: ", t,
                    "| time: %.2f" % (t1 - t0),
                    "| loss_mlm: %.3f | loss_nsp: %.3f" % (loss_mlm, loss_nsp),
                    "\n| mlm tgt: ", [data.i2v[i] for i in masked_target[0] if (i != data.v2i["<PAD>"]) and i != 0],
                    "\n| mlm prd: ", [data.i2v[i] for i in masked_pred[0] if (i != data.v2i["<PAD>"]) and i != 0],
                )
            t0 = t1

    # save attention matrix for visualization
    # attentions, cls_logits_, seq_logits_ = model.sess.run([model.attentions, model.cls_logits, model.seq_logits], {model.tfx: bx, model.training: False})
    # data = {"src": bx, "label": cls_logits_.argmax(axis=1), "seq": seq_logits_.argmax(axis=-1), "attentions": attentions, "i2v": i2v}
    # import pickle
    # with open("./visual_helper/GPT_attention_matrix.pkl", "wb") as f:
    #     pickle.dump(data, f)

if __name__ == "__main__":
    main()


