from pickle import load
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy,softmax, relu

import utils
from GPT import GPT
import os
import pickle

MASK_RATE = 0.15

class BERT(GPT):

    def __init__(
        self, model_dim, max_len, num_layer, num_head, n_vocab, lr,
        max_seg=3, drop_rate=0.2, padding_idx=0) -> None:
        super().__init__(model_dim, max_len, num_layer, num_head, n_vocab, lr, max_seg, drop_rate, padding_idx)
    
    def step(self,seqs,segs,seqs_, loss_mask,nsp_labels):
        device = next(self.parameters()).device
        self.opt.zero_grad()
        mlm_logits, nsp_logits = self(seqs, segs, training=True)    # [n, step, n_vocab], [n, n_cls]
        mlm_loss = cross_entropy(
            torch.masked_select(mlm_logits,loss_mask).reshape(-1,mlm_logits.shape[2]),
            torch.masked_select(seqs_,loss_mask.squeeze(2))
            )
        nsp_loss = cross_entropy(nsp_logits,nsp_labels.reshape(-1))
        loss = mlm_loss + 0.2 * nsp_loss
        loss.backward()
        self.opt.step()
        return loss.cpu().data.numpy(),mlm_logits

    def mask(self, seqs):
        mask = torch.eq(seqs,self.padding_idx)
        return mask[:, None, None, :]

def _get_loss_mask(len_arange, seq, pad_id):
    rand_id = np.random.choice(len_arange, size=max(2, int(MASK_RATE * len(len_arange))), replace=False)
    loss_mask = np.full_like(seq, pad_id, dtype=np.bool)
    loss_mask[rand_id] = True
    return loss_mask[None, :], rand_id

def do_mask(seq, len_arange, pad_id, mask_id):
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    seq[rand_id] = mask_id
    return loss_mask

def do_replace(seq, len_arange, pad_id, word_ids):
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    seq[rand_id] = torch.from_numpy(np.random.choice(word_ids, size=len(rand_id))).type(torch.IntTensor)
    return loss_mask

def do_nothing(seq, len_arange, pad_id):
    loss_mask, _ = _get_loss_mask(len_arange, seq, pad_id)
    return loss_mask

def random_mask_or_replace(data,arange,dataset):
    seqs, segs,xlen,nsp_labels = data
    seqs_ = seqs.data.clone()
    p = np.random.random()
    if p < 0.7:
        # mask
        loss_mask = np.concatenate([
            do_mask(
                seqs[i],
                np.concatenate((arange[:xlen[i,0]],arange[xlen[i,0]+1:xlen[i].sum()+1])),
                dataset.pad_id,
                dataset.mask_id
                )
                for i in range(len(seqs))], axis=0)
    elif p < 0.85:
        # do nothing
        loss_mask = np.concatenate([
            do_nothing(
                seqs[i],
                np.concatenate((arange[:xlen[i,0]],arange[xlen[i,0]+1:xlen[i].sum()+1])),
                dataset.pad_id
                )
                for i in range(len(seqs))],  axis=0)
    else:
        # replace
        loss_mask = np.concatenate([
            do_replace(
                seqs[i],
                np.concatenate((arange[:xlen[i,0]],arange[xlen[i,0]+1:xlen[i].sum()+1])),
                dataset.pad_id,
                dataset.word_ids
                )
                for i in range(len(seqs))],  axis=0)
    loss_mask = torch.from_numpy(loss_mask).unsqueeze(2)
    return seqs, segs, seqs_, loss_mask, xlen, nsp_labels

def train():
    MODEL_DIM = 256
    N_LAYER = 4
    LEARNING_RATE = 1e-4
    dataset = utils.MRPCData("./MRPC",2000)
    print("num word: ",dataset.num_word)
    model = BERT(
        model_dim=MODEL_DIM, max_len=dataset.max_len, num_layer=N_LAYER, num_head=4, n_vocab=dataset.num_word,
        lr=LEARNING_RATE, max_seg=dataset.num_seg, drop_rate=0.2, padding_idx=dataset.pad_id
    )
    if torch.cuda.is_available():
        print("GPU train avaliable")
        device =torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")
        model = model.cpu()
    
    loader = DataLoader(dataset,batch_size=32,shuffle=True)
    arange = np.arange(0,dataset.max_len)
    for epoch in range(500):
        for batch_idx, batch in enumerate(loader):
            seqs, segs, seqs_, loss_mask, xlen, nsp_labels = random_mask_or_replace(batch,arange,dataset)
            seqs, segs, seqs_, nsp_labels, loss_mask = seqs.type(torch.LongTensor).to(device), segs.type(torch.LongTensor).to(device),seqs_.type(torch.LongTensor).to(device),nsp_labels.to(device),loss_mask.to(device)
            loss, pred = model.step(seqs, segs, seqs_, loss_mask, nsp_labels)
            if batch_idx % 100 == 0:
                pred = pred[0].cpu().data.numpy().argmax(axis=1)
                print(
                "\n\nEpoch: ",epoch,
                "|batch: ", batch_idx,
                "| loss: %.3f" % loss,
                "\n| tgt: ", " ".join([dataset.i2v[i] for i in seqs[0].cpu().data.numpy()[:xlen[0].sum()+1]]),
                "\n| prd: ", " ".join([dataset.i2v[i] for i in pred[:xlen[0].sum()+1]]),
                "\n| tgt word: ", [dataset.i2v[i] for i in (seqs_[0]*loss_mask[0].view(-1)).cpu().data.numpy() if i != dataset.v2i["<PAD>"]],
                "\n| prd word: ", [dataset.i2v[i] for i in pred*(loss_mask[0].view(-1).cpu().data.numpy()) if i != dataset.v2i["<PAD>"]],
                )
    os.makedirs("./visual/models/bert",exist_ok=True)
    torch.save(model.state_dict(),"./visual/models/bert/model.pth")
    export_attention(model,device,dataset)

def export_attention(model,device,data,name="bert"):
    model.load_state_dict(torch.load("./visual/models/bert/model.pth",map_location=device))
    seqs, segs,xlen,nsp_labels = data[:32]
    seqs, segs,xlen,nsp_labels = torch.from_numpy(seqs),torch.from_numpy(segs),torch.from_numpy(xlen),torch.from_numpy(nsp_labels)
    seqs, segs,nsp_labels = seqs.type(torch.LongTensor).to(device), segs.type(torch.LongTensor).to(device),nsp_labels.to(device)
    model(seqs,segs,False)
    seqs = seqs.cpu().data.numpy()
    data = {"src": [[data.i2v[i] for i in seqs[j]] for j in range(len(seqs))], "attentions": model.attentions}
    path = "./visual/tmp/%s_attention_matrix.pkl" % name
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
if __name__ == "__main__":
    train()