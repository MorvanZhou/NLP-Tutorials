import tensorflow as tf
from GPT import GPT, train, export_attention
import utils


class BERT(GPT):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, lr, max_seg=3, drop_rate=0.1, padding_idx=0):
        super().__init__(model_dim, max_len, n_layer, n_head, n_vocab, lr, max_seg, drop_rate, padding_idx)

    def mask(self, seqs):
        """
         abcd--
        a010011
        b001011
        c000111
        d000011
        -000001
        -000000

        a is a embedding for a-cd
        b is a embedding for ab-d
        c is a embedding for abc-
        later, b embedding will + another b embedding from previous residual input to predict c
        """
        eye = tf.eye(self.max_len+1, batch_shape=[len(seqs)], dtype=tf.float32)[:, 1:, :-1]
        pad = tf.math.equal(seqs, self.padding_idx)
        mask = tf.where(pad[:, tf.newaxis, tf.newaxis, :], 1, eye[:, tf.newaxis, :, :])
        return mask  # [n, 1, step, step]


if __name__ == "__main__":
    MODEL_DIM = 256
    N_LAYER = 4
    LEARNING_RATE = 1e-4
    d = utils.MRPCData("./MRPC", 2000)
    print("num word: ", d.num_word)
    m = BERT(
        model_dim=MODEL_DIM, max_len=d.max_len - 1, n_layer=N_LAYER, n_head=4, n_vocab=d.num_word,
        lr=LEARNING_RATE, max_seg=d.num_seg, drop_rate=0.2, padding_idx=d.pad_id)
    train(m, d, step=5000, name="bert_window_mask")
    export_attention(m, d, "bert_window_mask")

