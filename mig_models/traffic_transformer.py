import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsl.nn.models.base_model import BaseModel
from tsl.nn.layers.graph_convs.diff_conv import DiffConv


class MultiGraphConv(nn.Module):
    """
    Multi-layer GCN block: applies a sequence of GraphConv layers
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 k: int,
                 n_layers: int = 2,
                 root_weight: bool = True,
                 add_backward: bool = True,
                 bias: bool = True,
                 activation: str = None):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(n_layers):
            if i == 0:
                in_channels = in_channels
            else:
                in_channels = out_channels

            layer = DiffConv(in_channels, out_channels, k=k,
                            root_weight=root_weight,
                            add_backward=add_backward,
                            bias=bias,
                            activation=activation)
            self.layers.append(layer)

    def forward(self, x, edge_index,
                edge_weight = None, cache_support: bool = False):
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight=edge_weight,
                      cache_support=cache_support)
            
        return x


# class MultiGraphConv(nn.Module):
#     """
#     Multi-layer GCN block: applies a sequence of GraphConv layers
#     """
#     def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, bias=True, activation="relu"):
#         super(MultiGraphConv, self).__init__()
#         layers = []
#         if num_layers == 1:
#             layers.append(nn.Linear(in_feats, out_feats, bias=bias))
#         else:
#             # first layer
#             layers.append(nn.Linear(in_feats, hidden_feats, bias=bias))
#             # intermediate
#             for _ in range(num_layers - 2):
#                 layers.append(nn.Linear(hidden_feats, hidden_feats, bias=bias))
#             # final
#             layers.append(nn.Linear(hidden_feats, out_feats, bias=bias))
#         self.layers = nn.ModuleList(layers)
#         self.fn_act = getattr(F, activation) 

#     def forward(self, x, A_hat):
#         # x: (B, N, F)
#         h = x
#         for i, lin in enumerate(self.layers):
#             h = torch.einsum('ij,bjf->bif', A_hat, h)
#             h = lin(h)
#             if i < len(self.layers) - 1:
#                 h = self.fn_act(h)
#         return h

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T].unsqueeze(2)

class PeriodicEncoding(nn.Module):
    def __init__(self, d_model, period, max_len=5000):
        super(PeriodicEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term * (2*math.pi/period))
        pe[:, 1::2] = torch.cos(position * div_term * (2*math.pi/period))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T].unsqueeze(2)
    
class TrafficTransformerEncoder(nn.Module):
    def __init__(self, in_feat, out_feat, hid_feat,
                 n_heads, dropout=0.1):
        super(TrafficTransformerEncoder, self).__init__()
        self.in_fc = nn.Linear(in_feat, hid_feat)
        self.GCN = MultiGraphConv(hid_feat, hid_feat, hid_feat,
                                 num_layers=2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_feat, nhead=n_heads,
            dropout=dropout, dim_feedforward=hid_feat*4)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=1)
        self.out_fc = nn.Linear(hid_feat, out_feat)

    def forward(self, x):
        # x: (B, T, N, F) -> (T, B*N, F)
        B, T, N, F = x.shape
        h = self.in_fc(x).permute(1,0,2,3).contiguous().view(T, B*N, -1)
        h = self.transformer(h)
        h = h.view(T, B, N, -1).permute(1,0,2,3)
        return self.out_fc(h)

class TrafficTransformer(BaseModel):
    def __init__(self, 
                    input_size,
                    output_size,
                    hidden_size,
                    n_heads,
                    n_layers,
                    exog_size=0,
                    kernel_size=2,
                    dropout=0.1,
                    num_nodes=None,
                    activation="relu"):
        
        super(TrafficTransformer, self).__init__()
        self.in_fc = nn.Linear(input_size + exog_size, hidden_size)
        # multilayer graph conv
        self.graph_conv = MultiGraphConv(hidden_size, hidden_size, hidden_size,
                                         n_layers=kernel_size, activation=activation)
        
        self.pos_enc = PositionalEncoding(hidden_size)
        self.daily_enc = PeriodicEncoding(hidden_size, period=24)
        self.weekly_enc = PeriodicEncoding(hidden_size, period=168)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads,
            dropout=dropout, dim_feedforward=hidden_size*4)
        
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=n_layers)
        
        self.out_fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, x, u=None, edge_index=None,
                edge_weight=None, cache_support=False):
        if u is not None:
            # u: (B, T, N, F)
            # x: (B, T, N, F) -> (B*T, N, F)
            x = torch.cat([x, u], dim=-1)
        # x: (B, T, N, F)
        B, T, N, F = x.shape
        h = self.in_fc(x)
        # graph conv per timestep
        h = h.view(B*T, N, -1)
        h = self.graph_conv(h, edge_index, edge_weight)
        h = h.view(B, T, N, -1)
        # add positional & periodic encodings
        h = self.pos_enc(h)
        h = self.daily_enc(h)
        h = self.weekly_enc(h)
        # transformer expects (T, B*N, hid)
        h = h.permute(1,0,2,3).contiguous().view(T, B*N, -1)
        h = self.transformer(h)
        h = h.view(T, B, N, -1).permute(1,0,2,3)
        return self.out_fc(h)

