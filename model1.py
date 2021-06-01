import torch
import torch.nn as nn
import math


class LearnedPositionalEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 500):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TemproalEmbedding(nn.Module):
    def __init__(self, in_dim, layers=1, dropout = .1):
        super(TemproalEmbedding, self).__init__()
        self.rnn = nn.LSTM(input_size=in_dim,hidden_size=in_dim,num_layers=layers,dropout=dropout)

    def forward(self, input):
        ori_shape = input.shape
        x = input.permute(3, 0, 2, 1)
        x = x.reshape(ori_shape[3], ori_shape[0] * ori_shape[2], ori_shape[1])
        x,_ = self.rnn(x)
        x = x.reshape(ori_shape[3], ori_shape[0], ori_shape[2], ori_shape[1])
        x = x.permute(1, 3, 2, 0)
        return x

class TrafficTransformer(nn.Module):
    def __init__(self,in_dim,layers=1,dropout=.1,heads=8):
        super().__init__()
        self.heads = heads
        self.pos = PositionalEncoding(in_dim,dropout=dropout)
        self.lpos = LearnedPositionalEncoding(in_dim, dropout=dropout)
        self.trans = nn.Transformer(in_dim, heads, layers, layers, in_dim*4, dropout=dropout)

    def forward(self,input, mask):
        x = input.permute(1,0,2)
        x = self.pos(x)
        x = self.lpos(x)
        x = self.trans(x,x,tgt_mask=mask)
        return x.permute(1,0,2)

    def _gen_mask(self,input):
        l = input.shape[1]
        mask = torch.eye(l)
        mask = mask.bool()
        return mask

class ttnet(nn.Module):
    def __init__(self, dropout=0.1, supports=None, in_dim=2, out_dim=12, hid_dim=32, layers=6):
        super(ttnet, self).__init__()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=hid_dim,
                                    kernel_size=(1, 1))
        self.start_embedding = TemproalEmbedding(hid_dim, layers=3, dropout=dropout)
        self.end_conv = nn.Linear(hid_dim, out_dim)
        self.network = TrafficTransformer(in_dim=hid_dim, layers=layers, dropout=dropout)

        mask0 = supports[0].detach()
        mask1 = supports[1].detach()
        mask = mask0 + mask1
        out = 0
        for i in range(1, 7):
            out += mask ** i
        self.mask = out == 0


    def forward(self, input):
        x = self.start_conv(input)
        x = self.start_embedding(x)[..., -1]
        x = x.transpose(1, 2)
        x = self.network(x, self.mask)
        x = self.end_conv(x)
        return x.transpose(1,2).unsqueeze(-1)
