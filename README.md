# Natural Language Processing Tutorial

Tutorial in Chinese can be found in [mofanpy.com](https://mofanpy.com/tutorials/machine-learning/nlp).

This repo includes many simple implementations of models in Neural Language Processing (NLP).

All code implementations in this tutorial are organized as following:

1. Search Engine
  - [TF-IDF numpy / TF-IDF skearn](#TF-IDF)
2. Understand Word (W2V)
  - [Continuous Bag of Words (CBOW)](#Word2Vec)
  - [Skip-Gram](#Word2Vec)
3. Understand Sentence (Seq2Seq)
  - [seq2seq](#Seq2Seq)
  - [CNN language model](#CNNLanguageModel)
4. All about Attention
  - [seq2seq with attention](#Seq2SeqAttention)
  - [Transformer](#Transformer)
5. Pretrained Models
  - [ELMo](#ELMO)
  - [GPT](#GPT)
  - [BERT](#BERT)
 
## TF-IDF

TF-IDF numpy [code](tf_idf.py)

TF-IDF short sklearn [code](tf_idf_sklearn.py)

![tf idf](img/tfidf_matrix.png)

## Word2Vec
[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)

Skip-Gram [code](skip-gram.py)

CBOW [code](CBOW.py)

![](img/cbow_illustration.png)
![](img/skip_gram_illustration.png)
![w2v](img/cbow.png)

## Seq2Seq
[Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

Seq2Seq [code](seq2seq.py)

![](img/seq2seq_illustration.png)

## CNNLanguageModel
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)

CNN language model [code](cnn-lm.py)

![](img/cnn-ml_sentence_embedding.png)

## Seq2SeqAttention
[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)

Seq2Seq Attention [code](seq2seq_attention.py)

![](img/luong_attention.png)
![seq2seq attention](img/seq2seq_attention.png)

## Transformer
[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

Transformer [code](transformer.py)

![](img/transformer_encoder_decoder.png)
![transformer attention line](img/transformer0_encoder_decoder_attention_line.png)

## ELMO
[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)

ELMO [code](ELMo.py)

![](img/elmo_training.png)
![](img/elmo_word_emb.png)

## GPT
[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

GPT [code](GPT.py)

![](img/gpt_structure.png)
![](img/gpt7_self_attention_line.png)

## BERT
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

BERT [code](BERT.py)

My new attempt [Bert with window mask](BERT_window_mask.py)

![](img/bert_gpt_comparison.png)
![bert](img/bert_self_mask2_self_attention_line.png)
