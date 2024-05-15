import math
from typing import Union, Sequence, Optional, List, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Parameter
try:
    from torch_geometric.nn import global_add_pool, GINConv
except ImportError:
    print("[Info]: torch_geometric is not installed.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dropout=0.1, nonlinearity=None):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.nonlinearity = getattr(nn, nonlinearity)() if nonlinearity else None
        self.reset_parameters()

    def forward(self, x):
        x = self.dropout(self.fc(x))
        if self.nonlinearity is not None:
            x = self.nonlinearity(x)
        return x

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight)


ModuleType = Type[nn.Module]


def miniblock(
        input_size: int,
        output_size: int = 0,
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = None,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and \
    activation."""
    layers: List[nn.Module] = [nn.Linear(input_size, output_size)]
    if norm_layer is not None:
        layers += [norm_layer(output_size)]  # type: ignore
    if activation is not None:
        layers += [activation()]
    return layers


class MLP(nn.Module):
    """Simple MLP backbone.

    Create a MLP of size input_dim * hidden_sizes[0] * hidden_sizes[1] * ...
    * hidden_sizes[-1] * output_dim

    :param int input_dim: dimension of the input vector.
    :param int output_dim: dimension of the output vector. If set to 0, there
        is no final linear layer.
    :param hidden_sizes: shape of MLP passed in as a list, not incluing
        input_dim and output_dim.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same actvition for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int = 0,
            hidden_sizes: Sequence[int] = (),
            norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
            activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
            device: Optional[Union[str, int, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.device = device
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes)
                norm_layer_list = norm_layer
            else:
                norm_layer_list = [
                    norm_layer for _ in range(len(hidden_sizes))]
        else:
            norm_layer_list = [None] * len(hidden_sizes)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
            else:
                activation_list = [
                    activation for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)
        hidden_sizes = [input_dim] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, norm, activ in zip(
                hidden_sizes[:-1], hidden_sizes[1:],
                norm_layer_list, activation_list):
            model += miniblock(in_dim, out_dim, norm, activ)
        if output_dim > 0:
            model += [nn.Linear(hidden_sizes[-1], output_dim)]
        self.output_dim = output_dim or hidden_sizes[-1]
        self.model = nn.Sequential(*model)

    def forward(
            self, x: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        x = torch.as_tensor(
            x, device=self.device, dtype=torch.float32)  # type: ignore
        return self.model(x)


class CNNBlock(nn.Module):
    def __init__(self, inp_dim, hid_dim, kernel_sizes):
        super(CNNBlock, self).__init__()
        modules = []
        in_channels = inp_dim
        assert len(hid_dim) == len(kernel_sizes), "len(hid_dim) != len(kernel_sizes)"
        # Build Encoder
        for h_dim, ks in zip(hid_dim, kernel_sizes):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size=ks, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.encoder(x)
        out = out.flatten(1)
        out = torch.mean(out, dim=1)
        return out.squeeze()


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        h = torch.matmul(inp, self.W)  # inp.shape: (B, N, in_features), h.shape: # [B, N, out_features]
        N = h.size()[1]  # number of nodes
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features),
                             h.repeat(1, N, 1)], dim=-1).view(-1, N, N, 2 * self.out_features)
        # [B, N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim, dropout=0.2, alpha=1e-2, heads_num=3):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(inp_dim, hid_dim, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(heads_num)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(hid_dim * heads_num, out_dim, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = torch.as_tensor(
            x, device=self.device, dtype=torch.float32)  # type: ignore
        adj = torch.as_tensor(
            adj, device=self.device, dtype=torch.int32)  # type: ignore
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x.mean(-2)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inp, adj):
        support = torch.mm(inp, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """
    https://github.com/tkipf/pygcn/blob/master/pygcn/models.py
    """

    def __init__(self, inp_dim, hid_dim, out_dim, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(inp_dim, hid_dim)
        self.gc2 = GraphConvolution(hid_dim, hid_dim)
        self.gc2 = GraphConvolution(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class GraphSAGELayer(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(GraphSAGELayer, self).__init__()
        self.inp_dim = inp_dim
        self.W1 = nn.Parameter(torch.zeros(size=(inp_dim, hid_dim)))
        self.W2 = nn.Parameter(torch.zeros(size=(2 * hid_dim, out_dim)))
        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_dim)

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, inp, adj):
        inp = F.leaky_relu(torch.matmul(inp, self.W1))
        h1 = torch.matmul(adj, inp)
        degree = adj.sum(dim=-1, keepdim=True) + 1e-6
        h1 = torch.div(h1, degree)
        h1 = torch.cat([inp, h1], dim=-1)
        h1 = F.leaky_relu(torch.matmul(h1, self.W2))
        h1 = self.bn(h1.transpose(1, 2)).transpose(1, 2)
        return h1


class Graphsage(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(Graphsage, self).__init__()
        self.sage1 = GraphSAGELayer(inp_dim, hid_dim, hid_dim)
        self.sage2 = GraphSAGELayer(hid_dim, hid_dim, hid_dim)
        self.sage3 = GraphSAGELayer(hid_dim, hid_dim, out_dim)
        # self.out = nn.Linear(3 * hid_dim, out_dim)
        self.out = FF(3 * hid_dim)

    def forward(self, inp, adj):
        adj = adj.float()
        hid1 = self.sage1(inp, adj)
        hid2 = self.sage2(hid1, adj)
        hid3 = self.sage3(hid2, adj)
        out = self.out(torch.cat([hid1, hid2, hid3], dim=-1))
        return out.mean(dim=1)


class GraphSAGE(nn.Module):
    """
    https://github.com/davide-belli/graphsage-diffpool-classifier/blob/master/source/models/graphsage.py
    """

    def __init__(self, input_feat, output_feat, device="cuda:0", normalize=True):
        super(GraphSAGE, self).__init__()
        self.device = device
        self.normalize = normalize
        self.input_feat = input_feat
        self.output_feat = output_feat
        self.linear = nn.Linear(self.input_feat, self.output_feat)
        self.layer_norm = nn.LayerNorm(self.output_feat)  # elementwise_affine=False
        nn.init.xavier_uniform_(self.linear.weight)

    def aggregate_convolutional(self, x, a):
        eye = torch.eye(a.shape[0], dtype=torch.float, device=self.device)
        a = a + eye
        h_hat = a @ x

        return h_hat

    def forward(self, x, a):
        h_hat = self.aggregate_convolutional(x, a)
        h = F.relu(self.linear(h_hat))
        if self.normalize:
            # h = F.normalize(h, p=2, dim=1)  # Normalize edge embeddings
            h = self.layer_norm(h)  # Normalize layerwise (mean=0, std=1)

        return h


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)
            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch, num_graphs):
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        # return x.view(num_graphs, -1, x.shape[-1]).mean(dim=1), x
        return None, x

    def get_embeddings(self, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1), device=device)
                x, _ = self.forward(x, edge_index, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class InfoGraph(nn.Module):
    def __init__(self, inp_dim, hid_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(InfoGraph, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.embedding_dim = mi_units = hid_dim * num_gc_layers
        self.encoder = Encoder(inp_dim, hid_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        # self.global_d = FF(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):
        y, M = self.encoder(x, edge_index, batch, num_graphs)

        # g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        return l_enc
