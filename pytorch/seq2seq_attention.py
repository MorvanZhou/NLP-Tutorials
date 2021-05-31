from torch import nn
import torch
import numpy as np
import utils
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy,softmax

class Seq2Seq(nn.Module):
    def __init__(self,enc_v_dim, dec_v_dim, emb_dim, units, max_pred_len, start_token, end_token):
        super().__init__()
        self.units = units
        self.dec_v_dim = dec_v_dim

        # encoder
        self.enc_embeddings = nn.Embedding(enc_v_dim,emb_dim)
        self.enc_embeddings.weight.data.normal_(0,0.1)
        self.encoder = nn.LSTM(emb_dim,units,1,batch_first=True)

        # decoder
        self.dec_embeddings = nn.Embedding(dec_v_dim,emb_dim)
        self.attn = nn.Linear(units,units)
        self.decoder_cell = nn.LSTMCell(emb_dim,units)
        self.decoder_dense = nn.Linear(units*2,dec_v_dim)

        self.opt = torch.optim.Adam(self.parameters(),lr=0.001)
        self.max_pred_len = max_pred_len
        self.start_token = start_token
        self.end_token = end_token
    
    def encode(self,x):
        embedded = self.enc_embeddings(x)   # [n, step, emb]
        hidden = (torch.zeros(1,x.shape[0],self.units),torch.zeros(1,x.shape[0],self.units))
        o,(h,c) = self.encoder(embedded,hidden) # [n, step, units], [num_layers * num_directions, n, units]
        return o,h,c
    
    def inference(self,x,return_align=False):
        self.eval()
        o,hx,cx = self.encode(x)    # [n, step, units], [num_layers * num_directions, n, units] * 2
        hx,cx = hx[0],cx[0]         # [n, units]
        start = torch.ones(x.shape[0],1)    # [n, 1]
        start[:,0] = torch.tensor(self.start_token)
        start= start.type(torch.LongTensor)
        dec_emb_in = self.dec_embeddings(start) # [n, 1, emb_dim]
        dec_emb_in = dec_emb_in.permute(1,0,2)  # [1, n, emb_dim]
        dec_in = dec_emb_in[0]                  # [n, emb_dim]
        output = []
        for i in range(self.max_pred_len):
            attn_prod = torch.matmul(self.attn(hx.unsqueeze(1)),o.permute(0,2,1)) # [n, 1, step]
            att_weight = softmax(attn_prod, dim=2)  # [n, 1, step]
            context = torch.matmul(att_weight,o)    # [n, 1, units]
            # attn_prod = torch.matmul(self.attn(o),hx.unsqueeze(2))  # [n, step, 1]
            # attn_weight = softmax(attn_prod,dim=1)                  # [n, step, 1]
            # context = torch.matmul(o.permute(0,2,1),attn_weight)    # [n, units, 1]
            hx, cx = self.decoder_cell(dec_in, (hx, cx))
            hc = torch.cat([context.squeeze(1),hx],dim=1)           # [n, units *2]
            # hc = torch.cat([context.squeeze(2),hx],dim=1)           # [n, units *2]
            result = self.decoder_dense(hc)
            result = result.argmax(dim=1).view(-1,1)
            dec_in=self.dec_embeddings(result).permute(1,0,2)[0]
            output.append(result)
        output = torch.stack(output,dim=0)
        self.train()

        return output.permute(1,0,2).view(-1,self.max_pred_len)
    
    def train_logit(self,x,y):
        o,hx,cx = self.encode(x)    # [n, step, units], [num_layers * num_directions, n, units] * 2
        hx,cx = hx[0],cx[0]         # [n, units]
        dec_in = y[:,:-1]           # [n, step]
        dec_emb_in = self.dec_embeddings(dec_in)    # [n, step, emb_dim]
        dec_emb_in = dec_emb_in.permute(1,0,2)      # [step, n, emb_dim]
        output = []
        for i in range(dec_emb_in.shape[0]):
            # General Attention:
            # score(ht,hs) = (ht^T)(Wa)hs
            # hs is the output from encoder
            # ht is the previous hidden state from decoder
            # self.attn(o): [n, step, units]
            attn_prod = torch.matmul(self.attn(hx.unsqueeze(1)),o.permute(0,2,1)) # [n, 1, step]
            att_weight = softmax(attn_prod, dim=2)  # [n, 1, step]
            context = torch.matmul(att_weight,o)    # [n, 1, units]
            # attn_prod = torch.matmul(self.attn(o),hx.unsqueeze(2))  # [n, step, 1]
            # attn_weight = softmax(attn_prod,dim=1)                  # [n, step, 1]
            # context = torch.matmul(o.permute(0,2,1),attn_weight)    # [n, units, 1]
            hx, cx = self.decoder_cell(dec_emb_in[i], (hx, cx))     # [n, units]
            hc = torch.cat([context.squeeze(1),hx],dim=1)           # [n, units *2]
            # hc = torch.cat([context.squeeze(2),hx],dim=1)           # [n, units *2]
            result = self.decoder_dense(hc)                              # [n, dec_v_dim]
            output.append(result)
        output = torch.stack(output,dim=0)  # [step, n, dec_v_dim]
        return output.permute(1,0,2)        # [n, step, dec_v_dim]
    
    def step(self,x,y):
        self.opt.zero_grad()
        batch_size = x.shape[0]
        logit = self.train_logit(x,y)    
        dec_out = y[:,1:]
        loss = cross_entropy(logit.reshape(-1,self.dec_v_dim),dec_out.reshape(-1))
        loss.backward()
        self.opt.step()
        return loss.detach().numpy()


def train():
    dataset = utils.DateData(4000)
    print("Chinese time order: yy/mm/dd ",dataset.date_cn[:3],"\nEnglish time order: dd/M/yyyy", dataset.date_en[:3])
    print("Vocabularies: ", dataset.vocab)
    print(f"x index sample:  \n{dataset.idx2str(dataset.x[0])}\n{dataset.x[0]}",
    f"\ny index sample:  \n{dataset.idx2str(dataset.y[0])}\n{dataset.y[0]}")
    loader = DataLoader(dataset,batch_size=32,shuffle=True)
    model = Seq2Seq(dataset.num_word,dataset.num_word,emb_dim=16,units=32,max_pred_len=11,start_token=dataset.start_token,end_token=dataset.end_token)
    for i in range(100):
        for batch_idx , batch in enumerate(loader):
            bx, by, decoder_len = batch
            loss = model.step(bx,by)
            if batch_idx % 70 == 0:
                target = dataset.idx2str(by[0, 1:-1].data.numpy())
                pred = model.inference(bx[0:1])
                res = dataset.idx2str(pred[0].data.numpy())
                src = dataset.idx2str(bx[0].data.numpy())
                print(
                    "Epoch: ",i,
                    "| t: ", batch_idx,
                    "| loss: %.3f" % loss,
                    "| input: ", src,
                    "| target: ", target,
                    "| inference: ", res,
                )
    # pkl_data = {"i2v": dataset.i2v, "x": dataset.x[:6], "y": dataset.y[:6], "align": model.inference(dataset.x[:6], return_align=True)}

    # with open("./visual/tmp/attention_align.pkl", "wb") as f:
    #     pickle.dump(pkl_data, f)

if __name__ == "__main__":
    train()