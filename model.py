import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# The encoder layers of generative and prediction models
class ConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, model='Gen'):
        super(ConvLayer, self).__init__()
        self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs * 2, kernel_size,
                              stride=stride, padding=padding, dilation=dilation))
        self.dilation = dilation
        self.padding = padding
        self.model = model
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.trans = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)
        if self.trans is not None:
            self.trans.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = self.conv(x)
        out = self.glu(y[:,:,:-self.padding].contiguous()) if self.model=='Gen' else self.glu(y)
        out = self.dropout(out)
        if self.trans is not None:
            x = self.trans(x)
        return out + x


class Encoder(nn.Module):
    def __init__(self, input_size, hid_size, n_levels, kernel_size=3, dropout=0.2, model='Gen'):
        super(Encoder, self).__init__()
        layers = []
        for i in range(n_levels):
            dilation_size = 2 ** i
            padding = (kernel_size-1)*dilation_size if model=='Gen' else dilation_size

            if i==0:
                layers += [ConvLayer(input_size, hid_size, kernel_size, stride=1,
                                     dilation=dilation_size, padding=padding, dropout=dropout, model=model)]
            else:
                layers += [ConvLayer(hid_size, hid_size, kernel_size, stride=1,
                                     dilation=dilation_size, padding=padding, dropout=dropout, model=model)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Generative model
class GEN(nn.Module):
    def __init__(self, emb_size, dic_size, hid_size, n_levels, kernel_size=3, emb_dropout=0.1, dropout=0.2):
        super(GEN, self).__init__()
        self.emb = nn.Embedding(dic_size, emb_size, padding_idx=0)
        self.drop = nn.Dropout(emb_dropout)
        self.encoder = Encoder(emb_size, hid_size, n_levels, kernel_size, dropout=dropout, model='Gen')
        self.decoder = nn.Linear(hid_size, dic_size)

    def forward(self, input):
        emb = self.drop(self.emb(input))
        y = self.encoder(emb.transpose(1, 2))
        o = self.decoder(y.transpose(1, 2))
        return o.contiguous()

# The decoder of prediction model
class NNet(nn.Module):
    def __init__(self, n_in, n_out, hide=(64, 64, 8)):
        super(NNet, self).__init__()
        self.n_hide = len(hide)
        self.fcs = nn.ModuleList([weight_norm(nn.Linear(n_in, hide[i] * 2)) if i == 0 else
                                  weight_norm(nn.Linear(hide[i - 1], n_out)) if i == self.n_hide else
                                  weight_norm(nn.Linear(hide[i - 1], hide[i] * 2)) for i in range(self.n_hide + 1)])
        self.init_weights()

    def init_weights(self):
        for i in range(self.n_hide + 1):
            self.fcs[i].weight.data.normal_(0, 0.01)

    def forward(self, x):
        for i in range(self.n_hide):
            x = F.glu(self.fcs[i](x))
        x = self.fcs[-1](x)
        return x


class PreDecoder(nn.Module):
    def __init__(self, emb_size, hid_size, output_size):
        super(PreDecoder, self).__init__()
        self.linear0 = NNet(n_in=emb_size, n_out=output_size, hide=(hid_size * 2, hid_size * 2, hid_size))
        self.linear1 = weight_norm(nn.Linear(hid_size, output_size))
        self.softmax = nn.Softmax(dim=1)
        #self.atten = None
        self.init_weights()

    def init_weights(self):
        self.linear1.weight.data.normal_(0, 0.01)

    def forward(self, emb, v):
        h = self.linear0(emb)
        v = self.linear1(v)
        a = self.softmax((v * h).masked_fill(emb[:,:,:1] == 0, float('-inf')))
        #self.atten = a
        out = torch.sum(a * h, 1)
        return out

# The prediction model
class PRE(nn.Module):
    def __init__(self, emb_size, dic_size, output_size, hid_size, n_levels, kernel_size=3, emb_dropout=0.1, dropout=0.2):
        super(PRE, self).__init__()
        self.emb = nn.Embedding(dic_size, emb_size, padding_idx=0)
        self.drop = nn.Dropout(emb_dropout)
        self.encoder = Encoder(emb_size, hid_size, n_levels, kernel_size, dropout=dropout, model='Pre')
        self.decoder = PreDecoder(emb_size, hid_size, output_size)

    def forward(self, input):
        emb = self.drop(self.emb(input))
        v = self.encoder(emb.transpose(1, 2))
        o = self.decoder(emb, v.transpose(1, 2))
        return o.contiguous()
