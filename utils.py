import numpy as np
import datetime
import os
import requests
import pandas as pd
import re


def sub_sampling(int_words, threshold=1e-5):
    # drop_p = 1 - sqrt(threshold/f(w))
    vocab, counts = np.unique(int_words, return_counts=True)
    total_count = len(int_words)
    frequency = counts / total_count
    p_keep = {v: np.sqrt(threshold / frequency[i]) for i, v in enumerate(vocab)}    # p_drop = 1 - p_keep
    kept_words = [word for word in int_words if np.random.random() < p_keep[word]]
    return kept_words


def get_date_data(n=5000):
    date_cn = []
    date_en = []
    for timestamp in np.random.randint(143835585, 2043835585, n):
        date = datetime.datetime.fromtimestamp(timestamp)
        date_cn.append(date.strftime("%y-%m-%d"))
        date_en.append(date.strftime("%d/%b/%Y"))
    vocab = set([str(i) for i in range(0, 10)] + ["-", "/", "<GO>", "<EOS>"] + [i.split("/")[1] for i in date_en])
    v2i = {v: i for i, v in enumerate(vocab, start=1)}
    v2i["<PAD>"] = 0
    vocab.add("<PAD>")
    i2v = {i: v for v, i in v2i.items()}
    x, y = [], []
    for cn, en in zip(date_cn, date_en):
        x.append([v2i[v] for v in cn])
        y.append(
            [v2i["<GO>"], ] + [v2i[v] for v in en[:3]] + [v2i[en[3:6]], ] + [v2i[v] for v in en[6:]] + [v2i["<EOS>"], ])
    x, y = np.array(x), np.array(y)
    return vocab, x, y, v2i, i2v, date_cn, date_en


def pad_zero(seqs, max_len):
    padded = np.zeros((len(seqs), max_len), dtype=np.int32)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded


def rotate_data(x, r):
    # x shape [n, 2]
    rotation_matrix = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return rotation_matrix.dot(x.T).T


def generate_dist_data():
    emb_dist1 = np.concatenate((np.random.normal(-.5, .3, (50, 2)), np.random.normal(0, .2, (100, 2))), axis=0)
    np.random.shuffle(emb_dist1)
    infrequent_words = np.arange(len(emb_dist1)-10)  # fake infrequent words
    frequent_words = np.arange(len(emb_dist1)-10, len(emb_dist1))  # the rest is frequent words
    emb_dist2 = rotate_data(emb_dist1, r=-np.pi / 3)
    emb_dist2[infrequent_words] += np.random.normal(0, 0.01, size=(
        len(infrequent_words), 2))  # large disturb infrequent word emb
    emb_dist2[frequent_words] += np.random.normal(0, 0.005,
                                                  size=(len(frequent_words), 2))  # small disturb frequent word emb
    emb_dist2 += np.array([[1, 1]])  # shift
    return emb_dist1, emb_dist2, frequent_words, infrequent_words


def maybe_download_mrpc(save_dir="./MRPC/", proxy=None):
    train_url = 'https://s3.amazonaws.com/senteval/senteval_data/msr_paraphrase_train.txt'
    test_url = 'https://s3.amazonaws.com/senteval/senteval_data/msr_paraphrase_test.txt'
    os.makedirs(save_dir, exist_ok=True)
    proxies = {"http": proxy, "https": proxy}
    for url in [train_url, test_url]:
        raw_path = os.path.join(save_dir, url.split("/")[-1])
        if not os.path.isfile(raw_path):
            print("downloading from %s" % url)
            r = requests.get(url, proxies=proxies)
            with open(raw_path, "wb") as f:
                f.write(r.content.replace('"', "<QUOTE>"))
                print("completed")


def text_standardize(text):
    text = re.sub(r'—', '-', text)
    text = re.sub(r'–', '-', text)
    text = re.sub(r'―', '-', text)
    text = re.sub(r" \d+(,\d+)?(\.\d+)? ", " <NUM> ", text)
    text = re.sub(r" \d+-", " <NUM>-", text)
    return text.strip()


def process_mrpc(dir="./MRPC/", go="<GO> ", end=" <EOS>"):
    data = {}
    files = os.listdir(dir)
    for f in files:
        df = pd.read_csv(os.path.join(dir, f), sep='\t')
        k = "train" if "train" in f else "test"
        data[k] = {"is_same": df["Quality"].values, "s1": df["#1 String"].values, "s2": df["#2 String"].values}
    vocab = set()
    for n in ["train", "test"]:
        for m in ["s1", "s2"]:
            for i in range(len(data[n][m])):
                data[n][m][i] = go + text_standardize(data[n][m][i].lower()) + end
                cs = data[n][m][i].split(" ")
                vocab.update(set(cs))
    v2i = {v: i for i, v in enumerate(vocab, start=1)}
    v2i["<PAD>"] = 0
    vocab.update(["<PAD>"])
    i2v = {i: v for v, i in v2i.items()}
    for n in ["train", "test"]:
        for m in ["s1", "s2"]:
            data[n][m+"id"] = [[v2i[v] for v in c.split(" ")] for c in data[n][m]]
    max_len = max(
        [len(s) for s in data["train"]["s1id"] + data["train"]["s2id"] + data["test"]["s1id"] + data["test"]["s2id"]])
    return data, v2i, i2v, max_len


def gpt_mrpc(dir="./MRPC", proxy=None):
    maybe_download_mrpc(save_dir="./MRPC/", proxy=proxy)
    data, v2i, i2v, max_len = process_mrpc(dir, go="<GO> ", end=" <EOS>")
    max_len = max_len * 2 + 1
    unsupervised_x = data["train"]["s1id"] + data["train"]["s2id"]
    unsupervised_x = pad_zero(unsupervised_x, max_len=max_len)
    supervised_x = [data["train"]["s1id"][i] + data["train"]["s2id"][i] for i in range(len(data["train"]["s1id"]))]
    supervised_x = pad_zero(supervised_x, max_len=max_len)
    supervised_label = data["train"]["is_same"]
    print("task1 example: ", data["train"]["s1"][0])
    print("task2 example: ", data["train"]["s1"][0] + " " + data["train"]["s2"][0])
    return v2i, i2v, max_len, unsupervised_x, supervised_x, supervised_label


def bert_mrpc(dir="./MRPC/", proxy=None):
    maybe_download_mrpc(save_dir="./MRPC/", proxy=proxy)
    data, v2i, i2v, max_len = process_mrpc(dir, go="", end=" <SEP>")
    v2i["<MASK>"] = len(v2i)        # add mask token
    i2v[v2i["<MASK>"]] = "<MASK>"

    max_len = max_len * 2 + 1
    x1 = data["train"]["s1id"] + data["train"]["s2id"]
    len1 = np.array([[0, len(s)] for s in x1])
    x1 = pad_zero(x1, max_len=max_len)
    x2 = [data["train"]["s1id"][i] + data["train"]["s2id"][i] for i in range(len(data["train"]["s1id"]))]
    x2 = pad_zero(x2, max_len=max_len)
    len2 = np.array([[len(s1), len(s2)] for s1, s2 in zip(data["train"]["s1id"], data["train"]["s2id"])])
    y2 = data["train"]["is_same"]
    print("task1 example: ", data["train"]["s1"][0])
    print("task2 example: ", data["train"]["s1"][0] + " " + data["train"]["s2"][0])

    seg1 = np.full((len(x1), max_len + 1), 2, np.int32)
    for i in range(len(data["train"]["s1id"])):
        si = len(data["train"]["s1id"][i]) + 1  # add 1 task seg
        seg1[i, :si] = 0
        si = len(data["train"]["s2id"][i]) + 1  # add 1 task seg
        seg1[i + len(data["train"]["s1id"]), :si] = 0

    seg2 = np.full((len(data["train"]["s1id"]), max_len + 1), 2, np.int32)
    for i in range(len(data["train"]["s1id"])):
        si = 1  # task seg
        si += len(data["train"]["s1id"][i])
        seg2[i, :si] = 0
        si_ = si + len(data["train"]["s2id"][i])
        seg2[i, si:si_] = 1
    print("segment1 example: ", seg1[1])
    print("segment2 example: ", seg2[1])
    normal_words = set(v2i.keys()).difference(["<SEP>", "<MASK>", "<PAD>"])
    return x1, x2, y2, seg1, seg2, max_len, v2i, i2v, len1, len2, list(normal_words)