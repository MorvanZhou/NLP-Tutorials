import numpy as np
import datetime
import os
import requests
import pandas as pd
import re
import itertools
import matplotlib.pyplot as plt

PAD_ID = 0


class DateData:
    def __init__(self, n):
        self.date_cn = []
        self.date_en = []
        for timestamp in np.random.randint(143835585, 2043835585, n):
            date = datetime.datetime.fromtimestamp(timestamp)
            self.date_cn.append(date.strftime("%y-%m-%d"))
            self.date_en.append(date.strftime("%d/%b/%Y"))
        self.vocab = set(
            [str(i) for i in range(0, 10)] + ["-", "/", "<GO>", "<EOS>"] + [
                i.split("/")[1] for i in self.date_en])
        self.v2i = {v: i for i, v in enumerate(self.vocab, start=1)}
        self.v2i["<PAD>"] = PAD_ID
        self.vocab.add("<PAD>")
        self.i2v = {i: v for v, i in self.v2i.items()}
        self.x, self.y = [], []
        for cn, en in zip(self.date_cn, self.date_en):
            self.x.append([self.v2i[v] for v in cn])
            self.y.append(
                [self.v2i["<GO>"], ] + [self.v2i[v] for v in en[:3]] + [
                    self.v2i[en[3:6]], ] + [self.v2i[v] for v in en[6:]] + [
                    self.v2i["<EOS>"], ])
        self.x, self.y = np.array(self.x), np.array(self.y)
        self.start_token = self.v2i["<GO>"]
        self.end_token = self.v2i["<EOS>"]

    def sample(self, n=64):
        bi = np.random.randint(0, len(self.x), size=n)
        bx, by = self.x[bi], self.y[bi]
        decoder_len = np.full((len(bx),), by.shape[1] - 1, dtype=np.int32)
        return bx, by, decoder_len

    def idx2str(self, idx):
        x = []
        for i in idx:
            x.append(self.i2v[i])
            if i == self.end_token:
                break
        return "".join(x)

    @property
    def num_word(self):
        return len(self.vocab)


def pad_zero(seqs, max_len):
    padded = np.full((len(seqs), max_len), fill_value=PAD_ID, dtype=np.int32)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded


def maybe_download_mrpc(save_dir="./MRPC/", proxy=None):
    train_url = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt'
    test_url = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'
    os.makedirs(save_dir, exist_ok=True)
    proxies = {"http": proxy, "https": proxy}
    for url in [train_url, test_url]:
        raw_path = os.path.join(save_dir, url.split("/")[-1])
        if not os.path.isfile(raw_path):
            print("downloading from %s" % url)
            r = requests.get(url, proxies=proxies)
            with open(raw_path, "w") as f:
                f.write(r.text.replace('"', "<QUOTE>"))
                print("completed")


def _text_standardize(text):
    text = re.sub(r'—', '-', text)
    text = re.sub(r'–', '-', text)
    text = re.sub(r'―', '-', text)
    text = re.sub(r" \d+(,\d+)?(\.\d+)? ", " <NUM> ", text)
    text = re.sub(r" \d+-+?\d*", " <NUM>-", text)
    return text.strip()


def _process_mrpc(dir="./MRPC"):
    data = {"train": None, "test": None}
    files = os.listdir(dir)
    top_n = 10
    for f in files:
        df = pd.read_csv(os.path.join(dir, f), sep='\t', nrows=top_n)
        k = "train" if "train" in f else "test"
        data[k] = {"is_same": df.iloc[:, 0].values, "s1": df["#1 String"].values, "s2": df["#2 String"].values}
    vocab = set()
    for n in ["train", "test"]:
        for m in ["s1", "s2"]:
            for i in range(len(data[n][m])):
                data[n][m][i] = _text_standardize(data[n][m][i].lower())
                cs = data[n][m][i].split(" ")
                vocab.update(set(cs))
    v2i = {v: i for i, v in enumerate(vocab, start=1)}
    v2i["<PAD>"] = PAD_ID
    v2i["<MASK>"] = len(v2i)
    v2i["<SEP>"] = len(v2i)
    i2v = {i: v for v, i in v2i.items()}
    for n in ["train", "test"]:
        for m in ["s1", "s2"]:
            data[n][m+"id"] = [[v2i[v] for v in c.split(" ")] for c in data[n][m]]
    max_len = max(
        [len(s1) + len(s2) + 1 for s1, s2 in zip(data["train"]["s1id"] + data["test"]["s1id"], data["train"]["s2id"]+ data["test"]["s2id"])])
    return data, v2i, i2v, max_len


class MRPCData4BERT:
    num_seg = 3
    pad_id = PAD_ID

    def __init__(self, data_dir="./MRPC/", proxy=None):
        maybe_download_mrpc(save_dir=data_dir, proxy=proxy)
        data, self.v2i, self.i2v, self.max_len = _process_mrpc(data_dir)

        self.xlen = np.array([
            [
                len(data["train"]["s1id"][i]), len(data["train"]["s2id"][i])
             ] for i in range(len(data["train"]["s1id"]))], dtype=int)
        x = [
            data["train"]["s1id"][i] + [self.v2i["<SEP>"]] + data["train"]["s2id"][i]
            for i in range(len(self.xlen))
        ]
        self.x = pad_zero(x, max_len=self.max_len)
        self.nsp_y = data["train"]["is_same"][:, None]

        self.seg = np.full(self.x.shape, self.num_seg-1, np.int32)
        for i in range(len(x)):
            si = self.xlen[i][0] + 1
            self.seg[i, :si] = 0
            si_ = si + self.xlen[i][1]
            self.seg[i, si:si_] = 1

        self.word_ids = np.array(list(set(self.i2v.keys()).difference(
            [self.v2i[v] for v in ["<PAD>", "<MASK>", "<SEP>"]])))

    def sample(self, n):
        bi = np.random.randint(0, self.x.shape[0], size=n)
        bx, bs, bl, by = self.x[bi], self.seg[bi], self.xlen[bi], self.nsp_y[bi]
        return bx, bs, bl, by

    @property
    def num_word(self):
        return len(self.v2i)


class MRPCData4GPT:
    def __init__(self, data_dir="./MRPC/", proxy=None):
        maybe_download_mrpc(save_dir="./MRPC/", proxy=proxy)
        data, v2i, i2v, max_len = _process_mrpc(dir, go="<GO> ", end=" <EOS>")
        max_len = max_len * 2 + 1
        unsupervised_x = data["train"]["s1id"] + data["train"]["s2id"]
        unsupervised_x = pad_zero(unsupervised_x, max_len=max_len)
        supervised_x = [data["train"]["s1id"][i] + data["train"]["s2id"][i] for i in range(len(data["train"]["s1id"]))]
        supervised_x = pad_zero(supervised_x, max_len=max_len)
        supervised_label = data["train"]["is_same"]
        print("task1 example: ", data["train"]["s1"][0])
        print("task2 example: ", data["train"]["s1"][0] + " " + data["train"]["s2"][0])
        return v2i, i2v, max_len, unsupervised_x, supervised_x, supervised_label


class Dataset:
    def __init__(self, x, y, v2i, i2v):
        self.x, self.y = x, y
        self.v2i, self.i2v = v2i, i2v
        self.vocab = v2i.keys()

    def sample(self, n):
        b_idx = np.random.randint(0, len(self.x), n)
        bx, by = self.x[b_idx], self.y[b_idx]
        return bx, by

    @property
    def num_word(self):
        return len(self.v2i)


def process_w2v_data(corpus, skip_window=2, method="skip_gram"):
    all_words = [sentence.split(" ") for sentence in corpus]
    all_words = np.array(list(itertools.chain(*all_words)))
    # vocab sort by decreasing frequency for the negative sampling below (nce_loss).
    vocab, v_count = np.unique(all_words, return_counts=True)
    vocab = vocab[np.argsort(v_count)[::-1]]

    print("all vocabularies sorted from more frequent to less frequent:\n", vocab)
    v2i = {v: i for i, v in enumerate(vocab)}
    i2v = {i: v for v, i in v2i.items()}

    # pair data
    pairs = []
    js = [i for i in range(-skip_window, skip_window + 1) if i != 0]

    for c in corpus:
        words = c.split(" ")
        w_idx = [v2i[w] for w in words]
        if method == "skip_gram":
            for i in range(len(w_idx)):
                for j in js:
                    if i + j < 0 or i + j >= len(w_idx):
                        continue
                    pairs.append((w_idx[i], w_idx[i + j]))  # (center, context) or (feature, target)
        elif method.lower() == "cbow":
            for i in range(skip_window, len(w_idx) - skip_window):
                context = []
                for j in js:
                    context.append(w_idx[i + j])
                pairs.append(context + [w_idx[i]])  # (contexts, center) or (feature, target)
        else:
            raise ValueError
    pairs = np.array(pairs)
    print("5 example pairs:\n", pairs[:5])
    if method.lower() == "skip_gram":
        x, y = pairs[:, 0], pairs[:, 1]
    elif method.lower() == "cbow":
        x, y = pairs[:, :-1], pairs[:, -1]
    else:
        raise ValueError
    return Dataset(x, y, v2i, i2v)


def show_w2v_word_embedding(model, data: Dataset, path):
    word_emb = model.embeddings.get_weights()[0]
    for i in range(data.num_word):
        c = "blue"
        try:
            int(data.i2v[i])
        except ValueError:
            c = "red"
        plt.text(word_emb[i, 0], word_emb[i, 1], s=data.i2v[i], color=c, weight="bold")
    plt.xlim(word_emb[:, 0].min() - .5, word_emb[:, 0].max() + .5)
    plt.ylim(word_emb[:, 1].min() - .5, word_emb[:, 1].max() + .5)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel("embedding dim1")
    plt.ylabel("embedding dim2")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, format="png")
    plt.show()