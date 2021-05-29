from torch import nn,optim
import torch
from torch.nn.functional import cross_entropy,softmax
import utils
from torch.utils.data import DataLoader
import os


class ELMo(nn.Module):

    def __init__(self, v_dim, emb_dim, units, n_layers, lr):
        super().__init__()
        self.n_layers = n_layers
        self.units = units
        self.v_dim = v_dim

        # encoder
        self.word_embed = nn.Embedding(num_embeddings= v_dim, embedding_dim= emb_dim,padding_idx=0)
        self.word_embed.weight.data.normal_(0,0.1)

        # forward LSTM
        self.fs = nn.ModuleList(
            [nn.LSTM(input_size = emb_dim, hidden_size = units, batch_first=True) if i==0 else nn.LSTM(input_size = units, hidden_size = units, batch_first=True) for i in range(n_layers)])
        self.f_logits = nn.Linear(in_features=units, out_features=v_dim)

        # backward LSTM
        self.bs = nn.ModuleList(
            [nn.LSTM(input_size = emb_dim, hidden_size = units, batch_first=True) if i==0 else nn.LSTM(input_size = units, hidden_size = units, batch_first=True) for i in range(n_layers)])
        self.b_logits = nn.Linear(in_features=units, out_features=v_dim)

        self.opt = optim.Adam(self.parameters(),lr = lr)

    def forward(self,seqs):
        embedded = self.word_embed(seqs)    # [n, step, emb_dim]
        fxs = [embedded[:, :-1, :]]         # [n, step-1, emb_dim]
        bxs = [embedded[:, 1:, :]]          # [n, step-1, emb_dim]
        for fl,bl in zip(self.fs,self.bs):
            hidden_f = (torch.zeros(1,seqs.shape[0],self.units),torch.zeros(1,seqs.shape[0],self.units))
            output_f,(h_f,c_f) = fl(fxs[-1], hidden_f)   # [n, step-1, units], [1, n, units]
            fxs.append(output_f)
            
            hidden_b = (torch.zeros(1,seqs.shape[0],self.units),torch.zeros(1,seqs.shape[0],self.units))
            output_b,(h_b,c_b) = bl(torch.flip(bxs[-1],dims=[1,]), hidden_b) # [n, step-1, units], [1, n, units]
            bxs.append(torch.flip(output_b,dims=(1,)))
        return fxs,bxs

    def step(self,seqs):
        self.opt.zero_grad()
        fo,bo = self(seqs)
        fo = self.f_logits(fo[-1])  # [n, step-1, v_dim]
        bo = self.b_logits(bo[-1])  # [n, step-1, v_dim]
        loss = (
            cross_entropy(fo.reshape(-1,self.v_dim),seqs[:,1:].reshape(-1)) +
            cross_entropy(bo.reshape(-1,self.v_dim),seqs[:,:-1].reshape(-1)))/2
        loss.backward()
        self.opt.step()
        return loss.detach().numpy(), (fo,bo)
    
    def get_emb(self,seqs):
        fxs,bxs = self(seqs)
        xs = [
            torch.cat((fxs[0][:,1:,:],bxs[0][:,:-1,:]),dim=2).cpu().data.numpy()
        ] + [
            torch.cat((f[:,1:,:],b[:,:-1,:]),dim=2).cpu().data.numpy() for f,b in zip(fxs[1:],bxs[1:])
        ]
        for x in xs:
            print("layers shape=",x.shape)
        return xs



def train():
    dataset = utils.MRPCSingle("./MRPC",rows=2000)
    UNITS = 256
    N_LAYERS = 2
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-3
    print('num word: ',dataset.num_word)
    model = ELMo(v_dim = dataset.num_word,emb_dim = UNITS, units=UNITS, n_layers=N_LAYERS,lr=LEARNING_RATE)
    if torch.cuda.is_available():
        print("GPU train avaliable")
        device =torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")
        model = model.cpu()
    loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
    for i in range(1):
        for batch_idx , batch in enumerate(loader):
            batch = batch.type(torch.LongTensor).to(device)
            loss, (fo,bo) = model.step(batch)
            if batch_idx % 20 ==0:
                fp = fo[0].cpu().data.numpy().argmax(axis=1)
                bp = bo[0].cpu().data.numpy().argmax(axis=1)
                print("\n\nEpoch: ", i,
                "| batch: ", batch_idx,
                "| loss: %.3f" % loss,
                "\n| tgt: ", " ".join([dataset.i2v[i] for i in batch[0].cpu().data.numpy() if i != dataset.pad_id]),
                "\n| f_prd: ", " ".join([dataset.i2v[i] for i in fp if i != dataset.pad_id]),
                "\n| b_prd: ", " ".join([dataset.i2v[i] for i in bp if i != dataset.pad_id]),
                )
    os.makedirs("./visual/models/elmo",exist_ok=True)
    torch.save(model.state_dict(),"./visual/models/elmo/model.pth")
    export_w2v(model,batch[:4],device)

def export_w2v(model,data,device):
    model.load_state_dict(torch.load("./visual/models/elmo/model.pth",map_location=device))
    emb = model.get_emb(data)
    print(emb)
if __name__ == "__main__":
    train()
