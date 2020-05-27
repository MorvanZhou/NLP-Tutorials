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
N_CLS = 2
MAX_SEG = 3   # sentence 1, sentence 2, padding
WORD_REPLACE_RATE = 0.15


class BERT(keras.Model):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, n_cls, max_seg=3, drop_rate=0.1, padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx

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
        self.o_mlm = keras.layers.Dense(n_vocab)
        self.o_cls = keras.layers.Dense(n_cls)

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.cross_entropy2 = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.001)
        self.opt2 = keras.optimizers.Adam(0.001)

    def get_embeddings(self, seg, seq):
        # position = tf.repeat(self.position_space, seg.shape[0], axis=0)
        # embed = tf.concat((
        #     self.task_emb(task),    # [n, 1, dim]
        #     self.word_emb(seq)     # [n, step, dim]
        # ), axis=1                           # [n, step+1, dim]
        # ) + self.segment_emb(seg) + self.position_emb(position)    # [n, step+1, dim]
        embed = self.word_emb(seq) + self.segment_emb(seg) + self.position_emb(self.position_space)    # [n, step, dim]
        return embed     # [n. step, dim]

    def __call__(self, seg, seq_replaced, task, training=None):
        embed = self.get_embeddings(seg, seq_replaced, task)
        z = self.encoder(embed, training=training, mask=self._pad_mask(seg))
        seq_logits = self.o_seq(z[:, 1:])  # [n, step, n_vocab]
        cls_logits = self.o_cls(z[:, 0])  # [n, n_cls]
        return seq_logits, cls_logits

    def mlm(self, seg, seq, mask_rate=0.1, training=False):
        embed = self.get_embeddings(seg, seq)   # [n. step, dim]

        # random mask
        np_input_mask = np.random.rand(embed.shape[0], embed.shape[1]) > mask_rate
        np_target_mask = np.logical_not(np_input_mask)
        input_mask = tf.expand_dims(tf.convert_to_tensor(np_input_mask.astype(np.float32)), axis=2)  # [n, step, 1]
        target_mask = tf.expand_dims(tf.convert_to_tensor(np_target_mask.astype(np.float32)), axis=2)   # [n, step, 1]

        masked_embed = embed * input_mask
        z = self.encoder(masked_embed, training=training, mask=self._pad_mask(seg))
        mlm_logits = self.o_mlm(z)
        masked_logits = mlm_logits * target_mask
        inf = np.zeros(masked_logits.shape, dtype=np.float32)
        inf[:, :, 0] += 1e-9
        masked_logits += tf.convert_to_tensor(inf)         # make softmax to the first word index in vocab
        return masked_logits, np_target_mask

    def mlm_step(self, seg, seq, mask_rate=0.1):
        with tf.GradientTape() as tape:
            mlm_logits, target_mask = self.mlm(seg, seq, mask_rate, training=True)
            target_words = seq * target_mask
            _loss = self.cross_entropy(target_words, mlm_logits)
        grads = tape.gradient(_loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return _loss.numpy(), mlm_logits.numpy().argmax(axis=2), target_words


    def step1(self, seg, seq_replaced, task, label_mask, label):
        with tf.GradientTape() as tape:
            seq_logits, _ = self(seg, seq_replaced, task, training=True)
            _loss = self.cross_entropy1(label, seq_logits*label_mask)
        grads = tape.gradient(_loss, self.trainable_variables)
        self.opt1.apply_gradients(zip(grads, self.trainable_variables))
        return _loss.numpy(), seq_logits.numpy()

    def step2(self, seg, seq_replaced, task, label):
        with tf.GradientTape() as tape:
            seq_logits, cls_logits = self(seg, seq_replaced, task, training=True)
            _loss = self.cross_entropy2(label, cls_logits)
        grads = tape.gradient(_loss, self.trainable_variables)
        self.opt2.apply_gradients(zip(grads, self.trainable_variables))
        return _loss.numpy(), seq_logits.numpy(), cls_logits.numpy()

    def _pad_mask(self, seqs):
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


def rand_replace(replace_tmp, bl):
    # The replace percentage in this method is different from which in original paper
    sizes = np.maximum(WORD_REPLACE_RATE * (bl-1), 1).astype(np.int)
    replace_change = np.random.rand(len(bl))
    label_rep = np.zeros_like(replace_tmp, dtype=np.bool)
    for i in range(len(replace_tmp)):
        if replace_change[i] < 0.9:     # replace with 90% chance
            if bl[i][0] == 0:   # single sentence
                label_rep[i, np.random.randint(low=1, high=bl[i][1]-1, size=sizes[i][1])] = True
            else:               # two sentence
                label_rep[i, np.random.randint(low=1, high=bl[i][0]-1, size=sizes[i][0])] = True
                label_rep[i, np.random.randint(low=bl[i][0]+1, high=sum(bl[i]), size=sizes[i][1])] = True

            if replace_change[i] < 0.8:     # use mask with 80% chance
                replace_tmp[i, label_rep[i]] = v2i["<MASK>"]
            else:                           # use replace with 10% chance
                replace_tmp[i, label_rep[i]] = [v2i[v] for v in
                                                   np.random.choice(normal_words, size=np.sum(label_rep[i]))]
        else:   # > 90% chance
            label_rep[i, :] = True

    return replace_tmp, label_rep


def train_mlm(data, model):
    bx, bs = data.sample_mlm(32)
    # replaced, label_rep = rand_replace(replace_tmp=bx.copy(), bl=bl)
    loss, masked_pred, masked_target = model.mlm_step(seg=bs, seq=bx, mask_rate=0.1)
    return loss, masked_pred, masked_target


def train_task2(x2, seg2, y2, task, slen):
    bi = np.random.randint(0, len(x2), size=b_size)
    bx, bs, by, bl = x2[bi], seg2[bi], y2[bi], slen[bi]
    replaced, label_rep = rand_replace(replace_tmp=bx.copy(), bl=bl)
    loss_, seq_logits_, cls_logits_ = model.step2(seg=bs, seq_replaced=replaced, task=task, label=by)
    # _, loss_, cls_logits_, seq_logits_ = model.sess.run([model.train2, model.loss2, model.cls_logits, model.seq_logits],
    #                                                     {
    #                                                         model.tfseq: bx, model.tfseq_replaced: replaced,
    #                                                         model.tflabel_rep: label_rep, model.tfseg: bs,
    #                                                         model.tfcls: by, model.tftask: task, model.training: True})
    return loss_, cls_logits_, by, replaced[0], seq_logits_[0], bx[0]


def main():
    # get and process data
    data = utils.MRPCData4BERT("./MRPC")
    model = BERT(
        model_dim=MODEL_DIM, max_len=data.max_len, n_layer=N_LAYER, n_head=N_HEAD, n_vocab=data.num_word, n_cls=N_CLS,
        max_seg=MAX_SEG, drop_rate=0.1, padding_idx=data.v2i["<PAD>"])

    t0 = time.time()
    b_size = 32
    for t in range(20000):
        loss, masked_pred, masked_target = train_mlm(data, model)
        print(
                "\n\nstep: ", t,
                # "| time: %.2f" % (t1 - t0),
                "| loss: %.3f" % loss,
                "\n| t1 tgt: ", u" ".join([data.i2v[i] for i in masked_target[0] if (i != data.v2i["<PAD>"]) and i != 0]).encode("utf-8"),
                # "\n| t1 rpl: ", u" ".join([data.i2v[i] for i in trpl1 if i != v2i["<PAD>"]]).encode("utf-8"),
                "\n| t1 prd: ", u" ".join([data.i2v[i] for i in masked_pred[0] if (i != data.v2i["<PAD>"]) and i != 0]).encode("utf-8"),
                # "\n| t2 acc: %.2f" % acc,
            )
        # if t % 10 == 0:     # task 2 in this case is easy to learn
        #     loss2, cls_logits_, by2, trpl2, t2logits, tx2 = train_task2(x2, seg2, y2, task2, len2)
        # if t % 50 == 0:
        #     correct = cls_logits_.argmax(axis=1) == by2
        #     acc = float(np.sum(correct) / len(cls_logits_))
        #     t1 = time.time()
        #     print(
        #         "\n\nstep: ", t,
        #         "| time: %.2f" % (t1 - t0),
        #         "| loss1: %.3f, loss2: %.3f" % (loss1, loss2),
        #         "\n| t1 tgt: ", u" ".join([i2v[i] for i in tx1 if i != v2i["<PAD>"]]).encode("utf-8"),
        #         "\n| t1 rpl: ", u" ".join([i2v[i] for i in trpl1 if i != v2i["<PAD>"]]).encode("utf-8"),
        #         "\n| t1 prd: ", u" ".join([i2v[i] for i in np.argmax(t1logits, axis=1) if i != v2i["<PAD>"]]).encode("utf-8"),
        #         "\n| t2 acc: %.2f" % acc,
        #     )
        #     t0 = t1

    # save attention matrix for visualization
    # attentions, cls_logits_, seq_logits_ = model.sess.run([model.attentions, model.cls_logits, model.seq_logits], {model.tfx: bx, model.training: False})
    # data = {"src": bx, "label": cls_logits_.argmax(axis=1), "seq": seq_logits_.argmax(axis=-1), "attentions": attentions, "i2v": i2v}
    # import pickle
    # with open("./visual_helper/GPT_attention_matrix.pkl", "wb") as f:
    #     pickle.dump(data, f)

if __name__ == "__main__":
    main()


