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

Thanks for the contribution made by [@W1Fl](https://github.com/W1Fl) with a simplified keras codes in [simple_realize](simple_realize)

## Dependencies

[requirements.txt](requirements.txt)

## TF-IDF

TF-IDF numpy [code](tf_idf.py)

TF-IDF short sklearn [code](tf_idf_sklearn.py)

![tf idf](https://mofanpy.com/static/results-small/nlp/tfidf_matrix.png)

## Word2Vec
[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)

Skip-Gram [code](skip-gram.py)

CBOW [code](CBOW.py)

![](https://mofanpy.com/static/results-small/nlp/cbow_illustration.png)
![](https://mofanpy.com/static/results-small/nlp/skip_gram_illustration.png)
![w2v](https://mofanpy.com/static/results/nlp/cbow_code_result.png)

## Seq2Seq
[Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

Seq2Seq [code](seq2seq.py)

![](https://mofanpy.com/static/results-small/nlp/seq2seq_illustration.png)

## CNNLanguageModel
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)

CNN language model [code](cnn-lm.py)

![](https://mofanpy.com/static/results-small/nlp/cnn-ml_sentence_embedding.png)

## Seq2SeqAttention
[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)

Seq2Seq Attention [code](seq2seq_attention.py)

![](https://mofanpy.com/static/results-small/nlp/luong_attention.png)
![seq2seq attention](https://mofanpy.com/static/results-small/nlp/seq2seq_attention_res.png)

## Transformer
[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

Transformer [code](transformer.py)

![](https://mofanpy.com/static/results-small/nlp/transformer_encoder_decoder.png)
![transformer attention](img/transformer0_decoder_encoder_attention.png)
![transformer attention line](img/transformer0_encoder_decoder_attention_line.png)

## ELMO
[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)

ELMO [code](ELMo.py)

![](https://mofanpy.com/static/results-small/nlp/elmo_training.png)
![](https://mofanpy.com/static/results-small/nlp/elmo_word_emb.png)

## GPT
[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

GPT [code](GPT.py)

![](https://mofanpy.com/static/results-small/nlp/gpt_structure.png)
![](https://mofanpy.com/static/results-small/nlp/gpt7_self_attention_line.png)
![](https://mofanpy.com/static/results-small/nlp/gpt7_self_attention.png)

## BERT
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

BERT [code](BERT.py)

My new attempt [Bert with window mask](BERT_window_mask.py)

![](https://mofanpy.com/static/results-small/nlp/bert_gpt_comparison.png)
![bert](https://mofanpy.com/static/results-small/nlp/bert_self_mask4_self_attention_line.png)
