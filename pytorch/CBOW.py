from torch import nn
import torch
from torch.nn.functional import cross_entropy,softmax
from utils import Dataset,process_w2v_data
from visual import show_w2v_word_embedding

corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]

class CBOW(nn.Module):
    def __init__(self,v_dim,emb_dim):
        super().__init__()
        self.v_dim = v_dim
        self.embeddings = nn.Embedding(v_dim,emb_dim)
        self.embeddings.weight.data.normal_(0,0.1)

        # self.opt = torch.optim.Adam(0.01)
        self.hidden_out = nn.Linear(emb_dim,v_dim)
        self.opt = torch.optim.SGD(self.parameters(),momentum=0.9,lr=0.01)
    
    def forward(self,x,training=None, mask=None):
        # x.shape = [n,skip_window*2]
        o = self.embeddings(x)  # [n, skip_window*2, emb_dim]
        o = torch.mean(o,dim=1) # [n, emb_dim]
        return o
    
    def loss(self, x, y, training=None):
        embedded = self(x,training)
        pred= self.hidden_out(embedded)
        return cross_entropy(pred,y)
    
    def step(self,x,y):
        self.opt.zero_grad()
        loss = self.loss(x,y,True)
        loss.backward()
        self.opt.step()
        return loss.detach().numpy()

def train(model,data):
    if torch.cuda.is_available():
        print("GPU train avaliable")
        device =torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")
        model = model.cpu()
    for t in range(2500):
        bx,by = data.sample(16)
        bx,by = torch.from_numpy(bx).to(device), torch.from_numpy(by).to(device)
        loss = model.step(bx,by)
        if t%200 == 0:
            print(f"step: {t}  |  loss: {loss}")

if __name__ == "__main__":
    d = process_w2v_data(corpus,skip_window=2, method="cbow")
    m = CBOW(d.num_word, 2)
    train(m,d)

    show_w2v_word_embedding(m,d,"./visual/results/cbow.png")