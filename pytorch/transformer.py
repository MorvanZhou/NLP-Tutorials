import torch.nn as nn
from torch.nn.functional import cross_entropy,softmax, relu
import numpy as np
import torch
from torch.utils import data
import utils
from torch.utils.data import DataLoader
import argparse

MAX_LEN = 11

class MultiHead(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.head_dim = model_dim // n_head
        self.n_head = n_head
        self.model_dim = model_dim
        self.wq = nn.Linear(model_dim, n_head * self.head_dim)
        self.wk = nn.Linear(model_dim, n_head * self.head_dim)
        self.wv = nn.Linear(model_dim, n_head * self.head_dim)

        self.o_dense = nn.Linear(model_dim, model_dim)
        self.o_drop = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self,q,k,v,mask,training):
        # residual connect
        residual = q
        dim_per_head= self.head_dim
        num_heads = self.n_head
        batch_size = q.size(0)

        # linear projection
        key = self.wk(k)    # [n, step, num_heads * head_dim]
        value = self.wv(v)  # [n, step, num_heads * head_dim]
        query = self.wq(q)  # [n, step, num_heads * head_dim]

        # split by head
        query = self.split_heads(query)       # [n, n_head, q_step, h_dim]
        key = self.split_heads(key)
        value = self.split_heads(value)  # [n, h, step, h_dim]
        context = self.scaled_dot_product_attention(query,key, value, mask)    # [n, q_step, h*dv]
        o = self.o_dense(context)   # [n, step, dim]
        o = self.o_drop(o)

        o = self.layer_norm(residual+o)
        return o

    def split_heads(self, x):
        x = torch.reshape(x,(x.shape[0], x.shape[1], self.n_head, self.head_dim))
        return x.permute(0,2,1,3)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = torch.tensor(k.shape[-1]).type(torch.float)
        score = torch.matmul(q,k.permute(0,1,3,2)) / (torch.sqrt(dk) + 1e-8)    # [n, n_head, step, step]
        if mask is not None:
            # change the value at masked position to negative infinity,
            # so the attention score at these positions after softmax will close to 0.
            score = score.masked_fill_(mask,-np.inf)
        self.attention = softmax(score,dim=-1)
        context = torch.matmul(self.attention,v)    # [n, num_head, step, head_dim]
        context = context.permute(0,2,1,3)          # [n, step, num_head, head_dim]
        context = context.reshape((context.shape[0], context.shape[1],-1))  
        return context  # [n, step, model_dim]

class PositionWiseFFN(nn.Module):
    def __init__(self,model_dim, dropout = 0.0):
        super().__init__()
        dff = model_dim*4
        self.l = nn.Linear(model_dim,dff)
        self.o = nn.Linear(dff,model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self,x):
        o = relu(self.l(x))
        o = self.o(o)
        o = self.dropout(o)

        o = self.layer_norm(x + o)
        return o    # [n, step, dim]



class EncoderLayer(nn.Module):

    def __init__(self, n_head, emb_dim, drop_rate):
        super().__init__()
        self.mh = MultiHead(n_head, emb_dim, drop_rate)
        self.ffn = PositionWiseFFN(emb_dim,drop_rate)
    
    def forward(self, xz, training, mask):
        # xz: [n, step, emb_dim]
        context = self.mh(xz, xz, xz, mask, training)  # [n, step, emb_dim]
        o = self.ffn(context)
        return o

class Encoder(nn.Module):
    def __init__(self, n_head, emb_dim, drop_rate, n_layer):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(n_head, emb_dim, drop_rate) for _ in range(n_layer)]
        )    
    def forward(self, xz, training, mask):

        for encoder in self.encoder_layers:
            xz = encoder(xz,training,mask)
        return xz       # [n, step, emb_dim]

class DecoderLayer(nn.Module):
    def __init__(self,n_head,model_dim,drop_rate):
        super().__init__()
        self.mh = nn.ModuleList([MultiHead(n_head, model_dim, drop_rate) for _ in range(2)])
        self.ffn = PositionWiseFFN(model_dim,drop_rate)
    
    def forward(self,yz, xz, training, yz_look_ahead_mask,xz_pad_mask):
        dec_output = self.mh[0](yz, yz, yz, yz_look_ahead_mask, training)   # [n, step, model_dim]
        
        dec_output = self.mh[1](dec_output, xz, xz, xz_pad_mask, training)  # [n, step, model_dim]

        dec_output = self.ffn(dec_output)   # [n, step, model_dim]

        return dec_output
    
class Decoder(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()

        self.num_layers = n_layer

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]
        )
    
    def forward(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        for decoder in self.decoder_layers:
            yz = decoder(yz, xz, training, yz_look_ahead_mask, xz_pad_mask)
        return yz   # [n, step, model_dim]

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, emb_dim, n_vocab):
        super().__init__()
        pos = np.expand_dims(np.arange(max_len),1)  # [max_len, 1]
        pe = pos / np.power(1000, 2*np.expand_dims(np.arange(emb_dim),0)/emb_dim)  # [max_len, emb_dim]
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = np.expand_dims(pe,0) # [1, max_len, emb_dim]
        self.pe = torch.from_numpy(pe).type(torch.float32)
        self.embeddings = nn.Embedding(n_vocab,emb_dim)
        self.embeddings.weight.data.normal_(0,0.1)
        
    def forward(self, x):
        device = self.embeddings.weight.device
        self.pe = self.pe.to(device)    
        x_embed = self.embeddings(x) + self.pe  # [n, step, emb_dim]
        return x_embed  # [n, step, emb_dim]

class Transformer(nn.Module):
    def __init__(self, n_vocab, max_len, n_layer = 6, emb_dim=512, n_head = 8, drop_rate=0.1, padding_idx=0):
        super().__init__()
        self.max_len = max_len
        self.padding_idx = torch.tensor(padding_idx)
        self.dec_v_emb = n_vocab 

        self.embed = PositionEmbedding(max_len, emb_dim, n_vocab)
        self.encoder = Encoder(n_head, emb_dim, drop_rate, n_layer)
        self.decoder = Decoder(n_head, emb_dim, drop_rate, n_layer)
        self.o = nn.Linear(emb_dim,n_vocab)
        self.opt = torch.optim.Adam(self.parameters(),lr=0.002)
    
    def forward(self,x,y,training= None):
        x_embed, y_embed = self.embed(x), self.embed(y) # [n, step, emb_dim] * 2
        pad_mask = self._pad_mask(x)    # [n, 1, step, step]
        encoded_z = self.encoder(x_embed,training,pad_mask) # [n, step, emb_dim]
        yz_look_ahead_mask = self._look_ahead_mask(y)   # [n, 1, step, step]
        decoded_z = self.decoder(y_embed,encoded_z, training, yz_look_ahead_mask, pad_mask) # [n, step, emb_dim]
        o = self.o(decoded_z)   # [n, step, n_vocab]
        return o
    
    def step(self, x, y):
        self.opt.zero_grad()
        logits = self(x,y[:, :-1],training=True)
        pad_mask = ~torch.eq(y[:,1:],self.padding_idx)  # [n, seq_len]
        loss = cross_entropy(logits.reshape(-1, self.dec_v_emb),y[:,1:].reshape(-1))
        loss.backward()
        self.opt.step()
        return loss.cpu().data.numpy(), logits

    def _pad_bool(self, seqs):
        o = torch.eq(seqs,self.padding_idx) # [n, step]
        return o
    def _pad_mask(self, seqs):
        len_q = seqs.size(1)
        mask = self._pad_bool(seqs).unsqueeze(1).expand(-1,len_q,-1)    # [n, len_q, step]
        return mask.unsqueeze(1)    # [n, 1, len_q, step]
    
    def _look_ahead_mask(self,seqs):
        device = next(self.parameters()).device
        batch_size, seq_len = seqs.shape
        mask = torch.triu(torch.ones((seq_len,seq_len), dtype=torch.long), diagonal=1).to(device)  # [seq_len ,seq_len]
        mask = torch.where(self._pad_bool(seqs)[:,None,None,:],1,mask[None,None,:,:]).to(device)   # [n, 1, seq_len, seq_len]
        return mask>0   # [n, 1, seq_len, seq_len]
    
    def translate(self, src, v2i, i2v):
        self.eval()
        device = next(self.parameters()).device
        src_pad = src
        # Initialize Decoder input by constructing a matrix M([n, self.max_len+1]) with initial value:
        # M[n,0] = start token id
        # M[n,:] = 0
        target = torch.from_numpy(utils.pad_zero(np.array([[v2i["<GO>"], ] for _ in range(len(src))]), self.max_len+1)).to(device)
        x_embed = self.embed(src_pad)
        encoded_z = self.encoder(x_embed,False,mask=self._pad_mask(src_pad))
        for i in range(0,self.max_len):
            y = target[:,:-1]
            y_embed = self.embed(y)
            decoded_z = self.decoder(y_embed,encoded_z,False,self._look_ahead_mask(y),self._pad_mask(src_pad))
            o = self.o(decoded_z)[:,i,:]
            idx = o.argmax(dim = 1).detach()
            # Update the Decoder input, to predict for the next position.
            target[:,i+1] = idx
        self.train()
        return target




def train(emb_dim=32,n_layer=3,n_head=4):
    
    dataset = utils.DateData(4000)
    print("Chinese time order: yy/mm/dd ",dataset.date_cn[:3],"\nEnglish time order: dd/M/yyyy", dataset.date_en[:3])
    print("Vocabularies: ", dataset.vocab)
    print(f"x index sample:  \n{dataset.idx2str(dataset.x[0])}\n{dataset.x[0]}",
    f"\ny index sample:  \n{dataset.idx2str(dataset.y[0])}\n{dataset.y[0]}")
    loader = DataLoader(dataset,batch_size=32,shuffle=True)
    model = Transformer(n_vocab=dataset.num_word, max_len=MAX_LEN, n_layer = n_layer, emb_dim=emb_dim, n_head = n_head, drop_rate=0.1, padding_idx=0)
    if torch.cuda.is_available():
        print("GPU train avaliable")
        device =torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")
        model = model.cpu()
    for i in range(100):
        for batch_idx , batch in enumerate(loader):
            bx, by, decoder_len = batch
            bx, by = torch.from_numpy(utils.pad_zero(bx,max_len = MAX_LEN)).type(torch.LongTensor).to(device), torch.from_numpy(utils.pad_zero(by,MAX_LEN+1)).type(torch.LongTensor).to(device)
            loss, logits = model.step(bx,by)
            if batch_idx%50 == 0:
                target = dataset.idx2str(by[0, 1:-1].cpu().data.numpy())
                pred = model.translate(bx[0:1],dataset.v2i,dataset.i2v)
                res = dataset.idx2str(pred[0].cpu().data.numpy())
                src = dataset.idx2str(bx[0].cpu().data.numpy())
                print(
                    "Epoch: ",i,
                    "| t: ", batch_idx,
                    "| loss: %.3f" % loss,
                    "| input: ", src,
                    "| target: ", target,
                    "| inference: ", res,
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim",type=int, help="change the model dimension")
    parser.add_argument("--n_layer",type=int, help="change the number of layers in Encoder and Decoder")
    parser.add_argument("--n_head",type=int, help="change the number of heads in MultiHeadAttention")

    args = parser.parse_args()
    args = dict(filter(lambda x: x[1],vars(args).items()))
    train(**args)