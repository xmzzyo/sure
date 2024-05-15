import warnings
from typing import Optional

import torch
from torch import Tensor
from torch.overrides import has_torch_function, handle_torch_function
try:
    from torch_geometric.nn import Data
except ImportError:
    print("[Info]: torch_geometric is not installed.")

from src.utils.config_parser import config
from src.utils.torch_utils import device


def gumbel_kmax(logits, k=5, mask=None, tau=1, hard=False, eps=1e-10, dim=-1):
    if not torch.jit.is_scripting():
        if type(logits) is not Tensor and has_torch_function((logits,)):
            return handle_torch_function(
                gumbel_kmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format,
                                device=device).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)

    if mask is None:
        y_soft = gumbels.softmax(dim)
    else:
        # https://github.com/allenai/allennlp/blob/5b1da908238794efb318da3327271cc945c6ba46/allennlp/nn/util.py#L243
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        y_soft = torch.nn.functional.softmax(gumbels * mask, dim=dim)
        y_soft = y_soft * mask
        y_soft = y_soft / (y_soft.sum(dim=dim, keepdim=True) + 1e-13)

    # resolve nan
    # y_soft = torch.where(torch.isnan(y_soft), torch.full_like(y_soft, 0), y_soft)

    if hard:
        # Straight through.
        index = y_soft.topk(k, dim)[1]
        # print("top_k_index ", index.shape)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format, device=device).scatter_(dim,
                                                                                                                index,
                                                                                                                1.0)
        ret = (y_hard - y_soft.detach() + y_soft)
        # shape = ret.shape
        # for i in range(shape[0]):
        #     for j in range(shape[1]):
        #         if len(shape) == 4:
        #             for l in range(shape[2]):
        #                 if ret[i, j, l, :].sum() != k:
        #                     print(y_soft[i, j, l, :], ret[i, j, l, :].sum(), k)
        #         else:
        #             if ret[i, j, :].sum() != k:
        #                 print(y_soft[i, j, :], ret[i, j, :].sum(), k)
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret.bool()


def get_mask_from_sequence_lengths(
        sequence_lengths: torch.Tensor, max_length: int
) -> torch.BoolTensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of each batch
    element, this function returns a `(batch_size, max_length)` mask variable.  For example, if
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.

    We require `max_length` here instead of just computing it from the input `sequence_lengths`
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return (sequence_lengths.unsqueeze(1) >= range_tensor).to(device)


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def flatten_and_batch_shift_indices(indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for [`batched_index_select`](./util.md#batched_index_select).
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into dimension 2 of a
    target tensor, which has size `(batch_size, sequence_length, embedding_size)`. This
    function returns a vector that correctly indexes into the flattened target. The sequence
    length of the target must be provided to compute the appropriate offsets.

    ```python
        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]
    ```

    # Parameters

    indices : `torch.LongTensor`, required.
    sequence_length : `int`, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.

    # Returns

    offset_indices : `torch.LongTensor`
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ValueError(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(
        target: torch.Tensor,
        indices: torch.LongTensor,
        flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.

    This function returns selected values in the target with respect to the provided indices, which
    have size `(batch_size, d_1, ..., d_n, embedding_size)`. This can use the optionally
    precomputed `flattened_indices` with size `(batch_size * d_1 * ... * d_n)` if given.

    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    [CoreferenceResolver](https://docs.allennlp.org/models/main/models/coref/models/coref/)
    model to select contextual word representations corresponding to the start and end indices of
    mentions.

    The key reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).

    # Parameters

    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A tensor of shape (batch_size, ...), where each element is an index into the
        `sequence_length` dimension of the `target` tensor.
    flattened_indices : `Optional[torch.Tensor]`, optional (default = `None`)
        An optional tensor representing the result of calling `flatten_and_batch_shift_indices`
        on `indices`. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.

    # Returns

    selected_targets : `torch.Tensor`
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.reshape(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


# constant
# anchor + neighbour
skip_adj = torch.zeros((config.s_k + 1), dtype=torch.bool, device=device)
skip_adj[0] = 1
skip_adj = skip_adj.repeat(config.t_k - 1).flatten()
skip_adj = torch.diag_embed(skip_adj, offset=config.s_k + 1)


def sample_subG(t_dis, s_dis, adj, k_nei, n_nei, feat, dense, t_k=2, s_k=2):
    """
    :param t_dis:
    :param s_dis:
    :param adj: bsz * max_task * max_task
    :param k_nei:
    :param n_nei:
    :param feat: bsz * T * N * FEAT
    :param dense
    :param t_k:
    :param s_k:
    :return:
    """
    bsz, N, T = t_dis.shape
    MN = s_dis.shape[-1]

    anchor_id = torch.arange(T * N, device=device).view(1, T, N).transpose(1, 2).expand(bsz, -1, -1)  # bsz * N * T
    # bsz * N * T
    t_index = gumbel_kmax(t_dis, k=t_k, hard=True)
    # bsz * N * t_k
    selected_t = anchor_id[t_index].view(bsz, N, t_k, 1)

    # bsz * T * N * max_nei -> bsz * N * T * max_nei
    ss_dis = s_dis[t_index].view(bsz, N, t_k, MN)
    nei_mask = get_mask_from_sequence_lengths(n_nei.flatten(), max_length=MN).view(bsz, N, 1, -1)
    s_index = gumbel_kmax(ss_dis, k=s_k, mask=nei_mask, hard=True)
    # bsz * N * max_nei -> bsz * N * t_k * max_nei
    k_neis = k_nei.unsqueeze(2).repeat(1, 1, t_k, 1)
    t_offset = torch.arange(0, T * N, step=N, device=device).view(1, 1, -1).expand(bsz, N, -1)
    offset = t_offset[t_index].view(bsz, N, t_k, 1)
    k_neis += offset
    selected_s = k_neis[s_index].view(bsz, N, t_k, s_k)

    # bsz * N * t_k * (s_k + 1)
    selected_n = torch.cat([selected_t, selected_s], dim=-1)
    origin_edge = selected_n % N

    adj_x = batched_index_select(adj, origin_edge).bool()
    # bsz * N * t_k * (s_k + 1) * (s_k + 1)
    adj_xy = batched_index_select(adj_x.transpose(3, 4).contiguous(), origin_edge)
    st_adj = torch.stack([torch.block_diag(*aa) for aa in adj_xy.flatten(0, 1)]).view(bsz, N, t_k * (s_k + 1),
                                                                                      t_k * (s_k + 1))
    st_adj |= skip_adj
    feat = feat.flatten(1, 2)
    sub_feat = batched_index_select(feat, selected_n.flatten(2, 3))

    if dense:
        return st_adj, sub_feat, t_index, s_index
    else:
        data_list = [Data(x=sub_feat[b, n, ...], edge_index=st_adj[b, n, ...].to_sparse().indices(), bid=b, nid=n) for n
                     in range(N) for b in range(bsz)]
        return data_list, t_index, s_index


# @torch.jit.script
def edges_by_id(N, bsz, coo_xy, feat):
    for i in range(bsz):
        print("job ", i)
        job_edges = coo_xy[i * N: (i + 1) * N]
        for n in range(N):
            edges = job_edges[n]
            nid, inverse_indices = edges.flatten().unique(return_inverse=True)
            sub_adj = inverse_indices.view(-1, 2).T
            sub_feat = feat[0, nid, :]

def sample_thread(Q, start_node, sampled_len):
    cur = start_node
    sampled_nodes = [cur]
    while len(sampled_nodes) < sampled_len:
        if Q[cur].sum() == 0:
            break
        cur = torch.multinomial(Q[cur], 1)[0]
        sampled_nodes.append(cur)
    mask = [1] * len(sampled_nodes) + [0] * (sampled_len - len(sampled_nodes))
    sampled_nodes.extend([0] * (sampled_len - len(sampled_nodes)))
    return torch.tensor(sampled_nodes), torch.tensor(mask)


def sort_st():
    pass


def sampler(batch_Q, T, N, task_nums, sampled_len):
    bsz = batch_Q.shape[0]
    sampled_seqs = []
    sampled_mask = []
    for b, tn in zip(range(bsz), task_nums):
        task_sampled = []
        task_mask = []
        # nodes at the last time step are start/target nodes respectively
        for sn in range((T - 1) * N, (T - 1) * N + tn):
            seqs, mask = sample_thread(batch_Q[b], sn, sampled_len)
            task_sampled.append(seqs)
            task_mask.append(mask)
        sampled_seqs.append(torch.stack(task_sampled))
        sampled_mask.append(torch.stack(task_mask))
    return torch.cat(sampled_seqs), torch.cat(sampled_mask)


def pad_sampled_seq():
    sort_st()
