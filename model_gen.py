import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class ConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(ConvLayer, self).__init__()
        self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs * 2, kernel_size,
                              stride=stride, padding=padding, dilation=dilation))
        self.padding = padding
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
        out = self.glu(y[:,:,:-self.padding].contiguous())
        out = self.dropout(out)
        if self.trans is not None:
            x = self.trans(x)
        return out + x


class ConvNet(nn.Module):
    def __init__(self, input_size,hid_size, n_levels,kernel_size=3, dropout=0.2):
        super(ConvNet, self).__init__()
        layers = []
        for i in range(n_levels):
            dilation_size = 2 ** i
            if i==0:
                layers += [ConvLayer(input_size, hid_size, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            else:
                layers += [ConvLayer(hid_size, hid_size, kernel_size, stride=1, dilation=dilation_size,
                                 padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GEN(nn.Module):
    def __init__(self, input_size, dic_size, output_size, hid_size=256, n_levels=5, kernel_size=3, emb_dropout=0.1, dropout=0.2):
        super(GEN, self).__init__()
        self.emb = nn.Embedding(dic_size, input_size)
        self.drop = nn.Dropout(emb_dropout)
        self.linear = nn.Linear(input_size, hid_size,bias=False)
        self.encoder = ConvNet( input_size,hid_size, n_levels, kernel_size, dropout=dropout)
        self.decoder = nn.Linear(hid_size, output_size)
        self.init_weights()

    def init_weights(self):
        self.emb.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        emb = self.drop(self.emb(input))
        y = self.encoder(emb.transpose(1, 2))
        o = self.decoder(y.transpose(1, 2))
        return o.contiguous()
