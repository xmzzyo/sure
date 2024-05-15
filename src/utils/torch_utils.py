import math
from typing import Sequence, Iterable, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import rnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def info_to_device(info, device):
    adj_m, task_nums, k_nei, n_nei, workload = info
    adj_m = torch.tensor(adj_m, dtype=torch.float, device=device)
    task_nums = torch.tensor(task_nums, dtype=torch.int, device=device)
    k_nei = torch.tensor(k_nei, dtype=torch.int, device=device)
    n_nei = torch.tensor(n_nei, dtype=torch.int, device=device)
    return adj_m, task_nums, k_nei, n_nei, workload


def to_tensor(
        obj: Union[torch.Tensor, np.ndarray, Iterable, Sequence, int, float],
        dtype: torch.dtype = torch.float,
        **kwargs) -> torch.Tensor:
    # torch.as_tensor()
    if torch.is_tensor(obj):
        return obj.to(dtype=dtype, device=device, **kwargs)

    if isinstance(obj, np.ndarray):
        if torch.is_tensor(obj[0]) and len(obj[0].size()) > 0:
            return torch.stack(obj.tolist()).to(dtype=dtype, device=device, **kwargs)
        return torch.from_numpy(obj).to(dtype=dtype, device=device, **kwargs)

    if not isinstance(obj, Sequence):
        if isinstance(obj, set):
            obj = [*obj]
        else:
            obj = [obj]
    elif not isinstance(obj, list) and isinstance(obj, Iterable):
        obj = [*obj]

    if isinstance(obj, list):
        if torch.is_tensor(obj[0]) and len(obj[0].size()) > 0:
            return torch.stack(obj).to(dtype=dtype, device=device, **kwargs)
        elif isinstance(obj[0], list):
            obj = [to_tensor(o) for o in obj]
            return torch.stack(obj).to(dtype=dtype, device=device, **kwargs)

    return torch.tensor(obj, dtype=dtype, device=device, **kwargs)


def int_to_device(device: Union[int, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device < 0:
        return torch.device("cpu")
    return torch.device(device)


def move_to_device(obj, device: Union[torch.device, int]):
    """
    Given a structure (possibly) containing Tensors,
    move all the Tensors to the specified device (or do nothing, if they are already on
    the target device).
    """
    device = int_to_device(device)

    if isinstance(obj, torch.Tensor):
        # You may be wondering why we don't just always call `obj.to(device)` since that would
        # be a no-op anyway if `obj` is already on `device`. Well that works fine except
        # when PyTorch is not compiled with CUDA support, in which case even calling
        # `obj.to(torch.device("cpu"))` would result in an error.
        return obj if obj.device == device else obj.to(device=device)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = move_to_device(value, device)
        return obj
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = move_to_device(item, device)
        return obj
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*(move_to_device(item, device) for item in obj))
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    else:
        return obj


def init_param(x, ini_type='uniform'):
    for layer in x._all_weights:
        for w in layer:
            if 'weight' in w:
                if ini_type == 'xavier':
                    init.xavier_normal_(getattr(x, w))
                elif ini_type == 'uniform':
                    stdv = 1.0 / math.sqrt(x.hidden_size)
                    init.uniform_(getattr(x, w), -stdv, stdv)
                elif ini_type == 'normal':
                    stdv = 1.0 / math.sqrt(x.hidden_size)
                    init.normal_(getattr(x, w), .0, stdv)
                else:
                    raise ValueError("init by one of xavier, uniform, normal")


def pad_seqs(seqs, seqs_ids=None, mode="left"):
    lens = np.array([x.shape[-2] for x in seqs])
    idx = np.argsort(-1 * lens)
    seqs = [seqs[ix] for ix in idx]
    lens = lens[idx]
    seqs = [to_tensor(s) for s in seqs]
    lens = torch.tensor(lens)
    seqs = rnn.pad_sequence(seqs, batch_first=True)
    if seqs_ids is not None:
        seqs_ids = [to_tensor(seqs_ids[ix]) for ix in idx]
        seqs_ids = rnn.pad_sequence(seqs_ids, batch_first=True)
        seqs_ids = to_tensor(seqs_ids)
        return seqs, lens, seqs_ids
    else:
        return seqs, lens


def log_sum_exp(x, axis=None):
    """Log sum exp function

    Args:
        x: Input.
        axis: Axis over which to perform sum.

    Returns:
        torch.Tensor: log sum exp

    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))


def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.

    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq


def local_global_loss_(l_enc, g_enc, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0] * l_enc.shape[1]
    N = int(num_nodes / num_graphs)

    # pos_mask = torch.zeros((num_nodes, num_graphs))
    # neg_mask = torch.ones((num_nodes, num_graphs))
    # for nodeidx, graphidx in enumerate(batch):
    #     pos_mask[nodeidx][graphidx] = 1.
    #     neg_mask[nodeidx][graphidx] = 0.
    pos_mask = torch.block_diag(*[torch.ones((N, 1), device=device) for _ in range(num_graphs)]).T
    neg_mask = 1 - pos_mask

    res = torch.matmul(l_enc, g_enc.t())

    E_pos = get_positive_expectation(torch.matmul(res, pos_mask), measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(torch.matmul(res, neg_mask), measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos
