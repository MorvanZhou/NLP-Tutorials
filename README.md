# Natural Language Processing Tutorial

Tutorial in Chinese can be found in [mofanpy.com](https://mofanpy.com/tutorials/machine-learning/nlp/).

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

Thanks for the contribution made by [@W1Fl](https://github.com/W1Fl) with a simplified keras codes in [simple_realize](simple_realize).
And the a [pytorch version of this NLP](/pytorch) tutorial made by [@ruifanxu](https://github.com/ruifan831).

## Installation

```shell script
$ git clone https://github.com/MorvanZhou/NLP-Tutorials
$ cd NLP-Tutorials/
$ sudo pip3 install -r requirements.txt
```


## TF-IDF

TF-IDF numpy [code](tf_idf.py)

TF-IDF short sklearn [code](tf_idf_sklearn.py)

<a target="_blank" href="https://mofanpy.com/static/results/nlp/tfidf_matrix.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/tfidf_matrix.png" height="250px" alt="image">
</a>


## Word2Vec
[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)

Skip-Gram [code](skip-gram.py)

CBOW [code](CBOW.py)

<a target="_blank" href="https://mofanpy.com/static/results/nlp/cbow_illustration.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/cbow_illustration.png" height="250px" alt="image">
</a>

<a target="_blank" href="https://mofanpy.com/static/results/nlp/skip_gram_illustration.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/skip_gram_illustration.png" height="250px" alt="image">
</a>

<a target="_blank" href="https://mofanpy.com/static/results/nlp/cbow_code_result.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/cbow_code_result.png" height="250px" alt="image">
</a>


## Seq2Seq
[Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

Seq2Seq [code](seq2seq.py)

<a target="_blank" href="https://mofanpy.com/static/results/nlp/seq2seq_illustration.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/seq2seq_illustration.png" height="250px" alt="image">
</a>

## CNNLanguageModel
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)

CNN language model [code](cnn-lm.py)

<a target="_blank" href="https://mofanpy.com/static/results/nlp/cnn-ml_sentence_embedding.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/cnn-ml_sentence_embedding.png" height="250px" alt="image">
</a>


## Seq2SeqAttention
[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)

Seq2Seq Attention [code](seq2seq_attention.py)

<a target="_blank" href="https://mofanpy.com/static/results/nlp/luong_attention.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/luong_attention.png" height="250px" alt="image">
</a>
<a target="_blank" href="https://mofanpy.com/static/results/nlp/seq2seq_attention_res.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/seq2seq_attention_res.png" height="250px" alt="image">
</a>



## Transformer
[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

Transformer [code](transformer.py)

<a target="_blank" href="https://mofanpy.com/static/results/nlp/transformer_encoder_decoder.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/transformer_encoder_decoder.png" height="250px" alt="image">
</a>
<a target="_blank" href="https://mofanpy.com/static/results/nlp/transformer0_decoder_encoder_attention.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/transformer0_decoder_encoder_attention.png" height="250px" alt="image">
</a>
<a target="_blank" href="https://mofanpy.com/static/results/nlp/transformer0_encoder_decoder_attention_line.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/transformer0_encoder_decoder_attention_line.png" height="250px" alt="image">
</a>


## ELMO
[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)

ELMO [code](ELMo.py)

<a target="_blank" href="https://mofanpy.com/static/results/nlp/elmo_training.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/elmo_training.png" height="250px" alt="image">
</a>
<a target="_blank" href="https://mofanpy.com/static/results/nlp/elmo_word_emb.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/elmo_word_emb.png" height="250px" alt="image">
</a>


## GPT
[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

GPT [code](GPT.py)

<a target="_blank" href="https://mofanpy.com/static/results/nlp/gpt_structure.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/gpt_structure.png" height="250px" alt="image">
</a>
<a target="_blank" href="https://mofanpy.com/static/results/nlp/gpt7_self_attention_line.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/gpt7_self_attention_line.png" height="250px" alt="image">
</a>


## BERT
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

BERT [code](BERT.py)

My new attempt [Bert with window mask](BERT_window_mask.py)

<a target="_blank" href="https://mofanpy.com/static/results/nlp/bert_gpt_comparison.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/bert_gpt_comparison.png" height="250px" alt="image">
</a>
<a target="_blank" href="https://mofanpy.com/static/results/nlp/bert_self_mask4_self_attention_line.png" style="text-align: center">
<img src="https://mofanpy.com/static/results/nlp/bert_self_mask4_self_attention_line.png" height="250px" alt="image">
</a>

