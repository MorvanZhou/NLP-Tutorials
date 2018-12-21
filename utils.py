import numpy as np
import datetime
import os
import requests
import pandas as pd


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


def download_mrpc(save_dir="./MRPC/", proxy=None):
    # based on this script: https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
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
                f.write(r.content.replace(b'"', b"<DQUOTE>"))
                print("completed")


def process_mrpc(dir="./MRPC/"):
    data = {}
    files = os.listdir(dir)
    for f in files:
        df = pd.read_csv(os.path.join(dir, f), sep='\t')
        k = "train" if "train" in f else "test"
        data[k] = {"is_same": df["Quality"].values, "s1": df["#1 String"].values, "s2": df["#2 String"].values}
    all_croups = np.concatenate((data["train"]["s1"], data["train"]["s2"], data["test"]["s1"], data["test"]["s2"]))
    vocab = set()
    croups_lsit = []
    for c in all_croups:
        cs = c.split(" ")
        croups_lsit.append(cs)
        vocab.update(set(cs))
    v2i = {v: i for i, v in enumerate(vocab)}
    i2v = {i: v for v, i in v2i.items()}
    data["train"]["s1id"] = [[v2i[v] for v in c.split(" ")] for c in data["train"]["s1"]]
    data["train"]["s2id"] = [[v2i[v] for v in c.split(" ")] for c in data["train"]["s2"]]
    data["test"]["s1id"] = [[v2i[v] for v in c.split(" ")] for c in data["test"]["s1"]]
    data["test"]["s2id"] = [[v2i[v] for v in c.split(" ")] for c in data["test"]["s2"]]
    return data, vocab, v2i, i2v


data, vocab, v2i, i2v = process_mrpc()
print(data["train"]["s1id"][:3])
print(data["train"]["s1"][:3])
