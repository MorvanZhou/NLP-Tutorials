# Natural Language Processing Tutorial

Tutorial in Chinese can be found in [mofanpy.com](https://mofanpy.com/tutorials/machine-learning/nlp).

This repo includes many simple implementations of models in Neural Language Processing (NLP).

All code implementations in this tutorial are organized as following:

1. Search Engine
  - [TF-IDF numpy](/tf_idf.py) / [TF-IDF skearn](/tf_idf_sklearn.py)
2. Understand Word
  - [Continuous Bag of Words (CBOW)](/CBOW.py)
  - [Skip-Gram](/skip-gram.py)
3. Understand Sentence
  - [seq2seq](/seq2seq.py)
  - [CNN language model](/cnn-lm.py)
4. All about Attention
  - [seq2seq with attention](/seq2seq_attention.py)
  - [Transformer](/transformer.py)
5. Pretrained Models
  - [ELMo](/ELMO.py)
  - [GPT](/GPT.py)
  - [BERT](/BERT.py) or my new attempt [Bert with self mask](/BERT_self_mask.py)
 
## some of the results

TF-IDF visualization

![tf idf](img/tfidf_matrix.png)

word2vec visualization

![w2v](img/cbow.png)

seq2seq attention visualization

![seq2seq attention](img/seq2seq_attention.png)

transformer encoder-decoder attention visualization

![transformer attention](img/transformer0_decoder_encoder_attention.png)
![transformer attention line](img/transformer0_encoder_decoder_attention_line.png)

bert self attention visualization

![bert](img/bert_self_mask2_self_attention_line.png)