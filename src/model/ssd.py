import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import ReLU, Sequential, Linear

try:
    from torch_geometric.nn import Batch, Data
except ImportError:
    print("[Info]: torch_geometric is not installed.")

from src.model.actor_critic import SIGMA_MIN, SIGMA_MAX
from src.model.modules import InfoGraph, FF
from src.model.sampler import get_mask_from_sequence_lengths, batched_index_select, sample_subG
from src.utils.config_parser import config
from src.utils.torch_utils import local_global_loss_, info_to_device


class GINLayer(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim, eps=1.0):
        super(GINLayer, self).__init__()
        self.inp_dim = inp_dim
        self.W = nn.Parameter(torch.zeros(size=(inp_dim, hid_dim)))
        self.reset_parameters()
        self.out = Sequential(Linear(hid_dim, hid_dim), ReLU(), Linear(hid_dim, out_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([eps]))

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, inp, adj):
        inp = F.relu(torch.matmul(inp, self.W))
        h1 = torch.matmul(adj, inp)
        degree = adj.sum(dim=-1, keepdim=True) + 1e-6
        h1 = torch.div(h1, degree)
        h1 += self.eps * inp
        h1 = self.out(h1)
        return h1


class GIN(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(GIN, self).__init__()
        self.in_ff = Sequential(Linear(inp_dim, hid_dim), ReLU(), Linear(hid_dim, hid_dim))
        self.sage1 = GINLayer(hid_dim, hid_dim, hid_dim)
        self.bn1 = torch.nn.BatchNorm1d(hid_dim)
        self.sage2 = GINLayer(hid_dim, hid_dim, hid_dim)
        self.bn2 = torch.nn.BatchNorm1d(hid_dim)
        self.sage3 = GINLayer(hid_dim, hid_dim, out_dim)
        self.bn3 = torch.nn.BatchNorm1d(hid_dim)
        self.ff = FF(4 * out_dim)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, inp, adj, g=False):
        adj = adj.float()
        bsz = adj.shape[0]
        inp = self.in_ff(inp)
        hid1 = F.relu(self.sage1(inp, adj))
        hid1 = F.dropout(hid1)
        hid2 = F.relu(self.sage2(hid1, adj))
        hid2 = F.dropout(hid2)
        hid3 = F.relu(self.sage3(hid2, adj))
        hid3 = F.dropout(hid3)
        hid = torch.cat([inp, hid1, hid2, hid3], dim=-1)
        if not g:
            hid = hid.view(bsz, config.t_k, (1 + config.s_k), -1)[:, :, 0, :]
        return self.ff(hid).mean(dim=1)


class BaseNet(nn.Module):
    def __init__(self, inp_dim, hid_dim, args):
        super(BaseNet, self).__init__()
        self.args = args
        self.inp_bn = nn.BatchNorm2d(hid_dim)
        self.feat_W = nn.Linear(inp_dim, hid_dim)
        num_gc_layers = 3
        self.distributed = args.local_rank is not None
        self.use_gat = args.use_gat
        self.gnn = InfoGraph(hid_dim, hid_dim, num_gc_layers).to(args.device)
        self.output_dim = hid_dim * num_gc_layers
        # self.ln = nn.LayerNorm(hid_dim)
        # self.st_att = self.transformer.encoder.layer_stack[-1].slf_attn
        self.mu = nn.Linear(hid_dim, 1)
        self.logvar = nn.Linear(hid_dim, 1)
        # self.dropout = nn.Dropout(0.2)
        self.mu_bn = nn.BatchNorm2d(config.max_task)
        self.logvar_bn = nn.BatchNorm2d(config.max_task)

        self.random_sample = args.random_sample
        self.no_mi = args.no_mi
        self.kld_w = args.kld_w
        self.nl_w = args.nl_w
        self.mi_w = args.mi_w

        # predefined variables, to avoid repeat created
        T, N = config.obs_len, config.max_task
        self.nei_offset = torch.arange(0, T * N, step=N, device=args.device).unsqueeze(0).view(1, T, 1, 1)

        skip_adj = torch.zeros(N, dtype=torch.bool, device=args.device)
        skip_adj[0] = 1
        skip_adj = skip_adj.repeat(T - 1).flatten()
        self.skip_adj = torch.diag_embed(skip_adj, offset=N)

        self.prior_mean = torch.zeros((1, N, T, (1 + config.max_neis)), device=args.device)
        self.prior_var = torch.empty_like(self.prior_mean, device=args.device).fill_(0.995)
        self.prior_logvar = torch.empty_like(self.prior_mean, device=args.device).fill_(np.log(0.995))

        self.cnt = 1

    def forward(self, s, state=None, info=None, with_loss=False):
        # bsz * OBS_LEN * MAX_TASK * FEAT
        bsz, T, N, D = s.shape
        MN = config.max_neis
        s = self.inp_bn(self.feat_W(s).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # s = self.dropout(s)

        adj_m, task_nums, k_nei, n_nei, workload = info_to_device(info, s.device)
        task_mask = get_mask_from_sequence_lengths(task_nums, N)
        nei_mask = get_mask_from_sequence_lengths(n_nei.flatten(), max_length=MN)

        t_feat = s.transpose(1, 2)

        k_neis = k_nei.unsqueeze(1).repeat(1, T, 1, 1)
        # nei_offset = torch.arange(0, T * N, step=N).unsqueeze(0).view(1, T, 1, 1).repeat(bsz, 1, N, MN)
        s_feat = batched_index_select(s.flatten(1, 2), k_neis + self.nei_offset)

        # bsz * N * T * (1+MN) * D
        feat = torch.cat([t_feat.unsqueeze(3), s_feat.transpose(1, 2)], dim=3)
        mu = torch.tanh(self.mu_bn(self.mu(feat).squeeze(-1)))
        logvar = torch.clamp(self.logvar_bn(self.logvar(feat).squeeze(-1)), min=SIGMA_MIN, max=SIGMA_MAX)
        var = logvar.exp()
        eps = torch.empty_like(mu).normal_()
        dis = eps * var + mu

        self.cnt += 1

        sampled_data = sample_subG(dis[..., 0], dis[..., 1:], adj_m, k_nei, n_nei, s, self.use_gat,
                                   t_k=config.t_k, s_k=config.s_k)
        if self.use_gat:
            st_adj, sub_feat, t_index, s_index = sampled_data
            st_adj, sub_feat = st_adj.flatten(0, 1), sub_feat.flatten(0, 1)
            l_enc = self.gnn(sub_feat, st_adj)

        else:
            graph_list, t_index, s_index = sampled_data

            graph_list = Batch.from_data_list(graph_list)
            l_enc = self.gnn(graph_list.x, graph_list.edge_index, graph_list.batch, bsz)

        l_enc = l_enc.view(bsz, N, -1)

        if with_loss:
            G_adj_m = adj_m.unsqueeze(1).expand(-1, T, -1, -1).bool()
            G_adj_m = torch.stack([torch.block_diag(*G_adj_m[b]) for b in range(bsz)])
            G_adj_m = G_adj_m | self.skip_adj
            if self.use_gat:
                g_enc = self.gnn(s.flatten(1, 2), G_adj_m, g=True)
            else:
                G = Batch.from_data_list(
                    [Data(x=s.flatten(1, 2)[b, ...], edge_index=G_adj_m[b, ...].to_sparse().indices()) for b in
                        range(bsz)])

                g_enc = self.gnn(G.x, G.edge_index, G.batch, bsz)
            mi_loss = local_global_loss_(l_enc, g_enc, measure='JSD')
            loss_dict = self.loss(mu, logvar, var, mi_loss, dis, t_index, s_index, task_mask, nei_mask)

            return l_enc, state, loss_dict
        else:
            return l_enc, state

    def loss(self, posterior_mu, posterior_logvar, posterior_var, mi_loss, dis, t_index, s_index, task_mask, nei_mask):
        bias = dis - posterior_mu
        gaussian_loss = 0.5 * (posterior_logvar + bias * bias / posterior_var).mean()
        t_ce = (t_index * (F.softmax(dis[..., 0], dim=-1) + 1e-9).log())[task_mask].sum() / task_mask.sum()
        s_ce = (s_index * (F.softmax(dis[..., 1:][t_index].view_as(s_index), dim=-1) + 1e-9).log())
        nei_mask = nei_mask.view(dis.shape[0], config.max_task, 1, -1).expand_as(s_ce)
        s_ce = s_ce[nei_mask].sum() / nei_mask.sum()
        NL = gaussian_loss - t_ce - s_ce

        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017,
        # https://arxiv.org/pdf/1703.01488.pdf
        var_division = posterior_var / self.prior_var
        diff = posterior_mu - self.prior_mean
        diff_term = diff * diff / self.prior_var
        logvar_division = self.prior_logvar - posterior_logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum())

        loss = self.kld_w * KLD + self.nl_w * NL + self.mi_w * mi_loss
        return {
            "KLD": self.kld_w * KLD,
            "NL": self.nl_w * NL,
            "MI": self.mi_w * mi_loss,
            "total_loss": loss
        }
