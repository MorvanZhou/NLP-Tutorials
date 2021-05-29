from transformer import Encoder
from torch import nn,optim
from torch.nn.functional import cross_entropy,softmax, relu
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import torch
import utils
import os
import pickle

class GPT(nn.Module):

    def __init__(self, model_dim, max_len, num_layer, num_head, n_vocab, lr, max_seg=3, drop_rate=0.2,padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        self.n_vocab = n_vocab
        self.max_len = max_len
        
        self.word_emb = nn.Embedding(n_vocab,model_dim)
        self.word_emb.weight.data.normal_(0,0.1)

        self.segment_emb = nn.Embedding(num_embeddings= max_seg, embedding_dim=model_dim)
        self.segment_emb.weight.data.normal_(0,0.1)
        self.position_emb = torch.empty(1,max_len,model_dim)
        nn.init.kaiming_normal_(self.position_emb,mode='fan_out', nonlinearity='relu')
        self.position_emb = nn.Parameter(self.position_emb)


        self.encoder = Encoder(n_head=num_head, emb_dim=model_dim, drop_rate=drop_rate, n_layer=num_layer)
        self.task_mlm = nn.Linear(in_features=model_dim, out_features=n_vocab)
        self.task_nsp = nn.Linear(in_features=model_dim*self.max_len, out_features=2)

        self.opt = optim.Adam(self.parameters(),lr)
    
    def forward(self,seqs, segs, training=False):
        embed = self.input_emb(seqs, segs)
        z = self.encoder(embed, training, mask = self.mask(seqs))   # [n, step, model_dim]
        mlm_logits = self.task_mlm(z)   # [n, step, n_vocab]
        nsp_logits = self.task_nsp(z.reshape(z.shape[0],-1))    # [n, n_cls]
        return mlm_logits, nsp_logits
    
    def step(self, seqs, segs, seqs_, nsp_labels):
        self.opt.zero_grad()
        mlm_logits, nsp_logits = self(seqs, segs, training=True)
        pred_loss = cross_entropy(mlm_logits.reshape(-1,self.n_vocab),seqs_.reshape(-1))
        nsp_loss = cross_entropy(nsp_logits,nsp_labels.reshape(-1))
        loss = pred_loss + 0.2 * nsp_loss
        loss.backward()
        self.opt.step()
        return loss.cpu().data.numpy(), mlm_logits
    
    def input_emb(self,seqs, segs):
        # device = next(self.parameters()).device
        # self.position_emb = self.position_emb.to(device)
        return self.word_emb(seqs) + self.segment_emb(segs) + self.position_emb
    
    def mask(self, seqs):
        device = next(self.parameters()).device
        batch_size, seq_len = seqs.shape
        mask = torch.triu(torch.ones((seq_len,seq_len), dtype=torch.long), diagonal=1).to(device)  # [seq_len ,seq_len]
        pad = torch.eq(seqs,self.padding_idx)   # [n, seq_len]
        mask = torch.where(pad[:,None,None,:],1,mask[None,None,:,:]).to(device)   # [n, 1, seq_len, seq_len]
        return mask>0   # [n, 1, seq_len, seq_len]
    
    @property
    def attentions(self):
        attentions = {
            "encoder": [l.mh.attention.cpu().data.numpy() for l in self.encoder.encoder_layers]
        }
        return attentions

def train():
    MODEL_DIM = 512
    N_LAYER = 8
    LEARNING_RATE = 1e-4
    dataset = utils.MRPCData("./MRPC",2000)
    print("num word: ",dataset.num_word)
    model = GPT(
        model_dim=MODEL_DIM, max_len=dataset.max_len-1, num_layer=N_LAYER, num_head=4, n_vocab=dataset.num_word,
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

    for epoch in range(10):
        for batch_idx, batch in enumerate(loader):
            seqs, segs,xlen,nsp_labels = batch
            seqs, segs,nsp_labels = seqs.type(torch.LongTensor).to(device), segs.type(torch.LongTensor).to(device),nsp_labels.to(device)
            # pred: [n, step, n_vocab]
            loss,pred = model.step(seqs=seqs[:,:-1], segs= segs[:,:-1], seqs_=seqs[:,1:], nsp_labels=nsp_labels)
            if batch_idx %100 == 0:
                pred = pred[0].cpu().data.numpy().argmax(axis = 1) # [step]
                print(
                    "Epoch: ",epoch,
                "|batch: ", batch_idx,
                "| loss: %.3f" % loss,
                "\n| tgt: ", " ".join([dataset.i2v[i] for i in seqs[0, 1:].cpu().data.numpy()[:xlen[0].sum()+1]]),
                "\n| prd: ", " ".join([dataset.i2v[i] for i in pred[:xlen[0].sum()+1]]),
                )
    os.makedirs("./visual/models/gpt",exist_ok=True)
    torch.save(model.state_dict(),"./visual/models/gpt/model.pth")
    export_attention(model,device,dataset)

def export_attention(model,device,data,name="gpt"):
    model.load_state_dict(torch.load("./visual/models/gpt/model.pth",map_location=device))
    seqs, segs,xlen,nsp_labels = data[:32]
    seqs, segs,xlen,nsp_labels = torch.from_numpy(seqs),torch.from_numpy(segs),torch.from_numpy(xlen),torch.from_numpy(nsp_labels)
    seqs, segs,nsp_labels = seqs.type(torch.LongTensor).to(device), segs.type(torch.LongTensor).to(device),nsp_labels.to(device)
    model(seqs[:,:-1],segs[:,:-1],False)
    seqs = seqs.cpu().data.numpy()
    data = {"src": [[data.i2v[i] for i in seqs[j]] for j in range(len(seqs))], "attentions": model.attentions}
    path = "./visual/tmp/%s_attention_matrix.pkl" % name
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
if __name__ == "__main__":
    train()
            



