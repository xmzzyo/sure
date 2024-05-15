from typing import Any, Dict, Tuple, Union, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from src.model.modules import MLP
from src.utils.utils import timethis

SIGMA_MIN = -20
SIGMA_MAX = 2


class DAGDataset(Dataset):
    def __init__(self, state, adj_matrix, task_num, k_nei, n_nei, v_s, returns):
        self.state = state
        self.adj_matrix = adj_matrix
        self.task_num = task_num
        self.k_nei = k_nei
        self.n_nei = n_nei
        self.v_s = v_s
        self.returns = returns

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, idx):
        data = (
            self.state[idx], self.adj_matrix[idx], self.task_num[idx], self.k_nei[idx], self.n_nei[idx], self.v_s[idx],
            self.returns[idx])
        return data


class Actor(nn.Module):
    """Simple actor network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
            self,
            preprocess_net: nn.Module,
            action_shape: Sequence[int],
            hidden_sizes: Sequence[int] = (),
            max_action: float = 1.0,
            device: Union[str, int, torch.device] = "cpu",
            preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim",
                            preprocess_net_output_dim)
        self.last = MLP(input_dim, self.output_dim,
                        hidden_sizes, device=self.device)
        self._max = max_action

    def forward(
            self,
            s: Union[np.ndarray, torch.Tensor],
            state: Any = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> logits -> action."""
        info = (s["adj_matrix"], s["task_num"], s["k_nei"], s["n_nei"], s["workload"])
        s = torch.as_tensor(
            s["next_state"], device=self.device, dtype=torch.float32  # type: ignore
        )
        logits, h = self.preprocess(s, state, info=info)[:2]
        logits = self._max * torch.tanh(self.last(logits))
        logits = F.softmax(logits, dim=-1)
        return logits, h

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class CriticWrapper:
    def __init__(self, critic):
        self.critic = critic

    def forward(self, s: Union[np.ndarray, torch.Tensor],
                a: Optional[Union[np.ndarray, torch.Tensor]] = None,
                info: Dict[str, Any] = {},
                with_loss=False, ):
        return self.critic(s, a, info, with_loss)

    @timethis
    def train_ray(self, data, ent_loss, clip_loss, actor, _grad_norm, _value_clip, _eps_clip, _weight_vf, scaler,
                  optim):
        return self.critic.train_ray(data, ent_loss, clip_loss, actor, _grad_norm, _value_clip, _eps_clip, _weight_vf,
                                     scaler,
                                     optim)

    def get_weights(self):
        return self.critic.state_dict()

    def set_weights(self, weights):
        self.critic.load_state_dict(weights)


class Critic(nn.Module):
    """Simple critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
            self,
            preprocess_net: nn.Module,
            hidden_sizes: Sequence[int] = (),
            device: Union[str, int, torch.device] = "cpu",
            preprocess_net_output_dim: Optional[int] = None,
            local_rank=-1
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = getattr(preprocess_net, "output_dim",
                            preprocess_net_output_dim)
        self.last = MLP(input_dim, 1, hidden_sizes, device=self.device)
        # self.cnn = CNNBlock(inp_dim=1, hid_dim=[16, 32, 12], kernel_sizes=[2, 3, 4])
        self.dist = local_rank >= 0

    def forward(
            self,
            s: Union[np.ndarray, torch.Tensor],
            a: Optional[Union[np.ndarray, torch.Tensor]] = None,
            info: Dict[str, Any] = {},
            with_loss=False,
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        info = (s["adj_matrix"], s["task_num"], s["k_nei"], s["n_nei"], s["workload"])
        s = torch.as_tensor(
            s["next_state"], device=self.device, dtype=torch.float32  # type: ignore
        )
        # if a is not None:
        #     a = torch.as_tensor(
        #         a, device=self.device, dtype=torch.float32  # type: ignore
        #     )
        #     s = torch.cat([s, a], dim=1)
        out = self.preprocess(s, state=None, info=info, with_loss=with_loss)
        loss_dict = None
        if len(out) == 2:
            logits, h = out
        else:
            logits, h, loss_dict = out
        logits = self.last(logits)
        # logits = self.cnn(logits)
        logits = logits.mean(1)
        if len(out) == 2:
            return F.leaky_relu(logits)
        else:
            return F.leaky_relu(logits), loss_dict

    def train_ray(self, data, ent_loss, clip_loss, actor, _grad_norm, _value_clip, _eps_clip, _weight_vf, scaler,
                  optim):
        # s = {"next_state": data.state, "adj_matrix": data.adj_matrix, "task_num": data.task_num,
        #      "k_nei": data.k_nei, "n_nei": data.n_nei}
        # calculate loss for critic
        # value, loss_dict = self(data.obs, a=None, info=None, with_loss=True)
        value, loss_dict = self(data, a=None, info=None, with_loss=True)
        value = value.flatten()
        if _value_clip:
            v_clip = data.v_s + (value - data.v_s).clamp(
                -_eps_clip, _eps_clip)
            vf1 = (data.returns - value).pow(2)
            vf2 = (data.returns - v_clip).pow(2)
            vf_loss = torch.max(vf1, vf2).mean()
        else:
            vf_loss = (data.returns - value).pow(2).mean()
        # calculate regularization and overall loss
        loss = clip_loss + _weight_vf * vf_loss + ent_loss + loss_dict["total_loss"]
        optim.zero_grad()
        # with torch.autograd.detect_anomaly():
        scaler.scale(loss).backward()

        if _grad_norm:  # clip large gradient
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(self.parameters()),
                max_norm=_grad_norm)
        scaler.step(optim)

        scaler.update()
        loss_dict["loss"] = loss
        loss_dict["vf_loss"] = vf_loss
        return loss_dict

    def test_ray(self, data):
        return data

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class ActorProb(nn.Module):
    """Simple actor network (output with a Gauss distribution).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param bool unbounded: whether to apply tanh activation on final logits.
        Default to False.
    :param bool conditioned_sigma: True when sigma is calculated from the
        input, False when sigma is an independent parameter. Default to False.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
            self,
            preprocess_net: nn.Module,
            action_shape: Sequence[int],
            hidden_sizes: Sequence[int] = (),
            max_action: float = 1.0,
            device: Union[str, int, torch.device] = "cpu",
            unbounded: bool = False,
            conditioned_sigma: bool = False,
            preprocess_net_output_dim: Optional[int] = None,
            local_rank=-1,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.device = device
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim",
                            preprocess_net_output_dim)
        self.mu = MLP(input_dim, self.output_dim,
                      hidden_sizes, device=self.device)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(input_dim, self.output_dim,
                             hidden_sizes, device=self.device)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))
        self._max = max_action
        self._unbounded = unbounded
        self.dist = local_rank >= 0

    def forward(
            self,
            s: Union[np.ndarray, torch.Tensor],
            state: Any = None,
            info: Dict[str, Any] = {},
            with_loss=False,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: s -> logits -> (mu, sigma)."""
        info = (s["adj_matrix"], s["task_num"], s["k_nei"], s["n_nei"], s["workload"])
        s = torch.as_tensor(
            s["next_state"], device=self.device, dtype=torch.float32  # type: ignore
        )
        out = self.preprocess(s, state, info=info, with_loss=with_loss)
        loss_dict = None
        if len(out) == 2:
            logits, h = out
        else:
            logits, h, loss_dict = out
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(
                self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX
            ).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()

        if len(out) == 2:
            return (mu, sigma), state
        else:
            return (mu, sigma), state, loss_dict

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class RecurrentActorProb(nn.Module):
    """Recurrent version of ActorProb.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            layer_num: int,
            state_shape: Sequence[int],
            action_shape: Sequence[int],
            hidden_layer_size: int = 128,
            max_action: float = 1.0,
            device: Union[str, int, torch.device] = "cpu",
            unbounded: bool = False,
            conditioned_sigma: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=int(np.prod(state_shape)),
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        output_dim = int(np.prod(action_shape))
        self.mu = nn.Linear(hidden_layer_size, output_dim)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(hidden_layer_size, output_dim)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self._max = max_action
        self._unbounded = unbounded

    def forward(
            self,
            s: Union[np.ndarray, torch.Tensor],
            state: Optional[Dict[str, torch.Tensor]] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        s = torch.as_tensor(
            s, device=self.device, dtype=torch.float32)  # type: ignore
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            s = s.unsqueeze(-2)
        self.nn.flatten_parameters()
        if state is None:
            s, (h, c) = self.nn(s)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            s, (h, c) = self.nn(s, (state["h"].transpose(0, 1).contiguous(),
                                    state["c"].transpose(0, 1).contiguous()))
        logits = s[:, -1]
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(
                self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX
            ).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        # please ensure the first dim is batch size: [bsz, len, ...]
        return (mu, sigma), {"h": h.transpose(0, 1).detach(),
                             "c": c.transpose(0, 1).detach()}


class RecurrentCritic(nn.Module):
    """Recurrent version of Critic.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            layer_num: int,
            state_shape: Sequence[int],
            action_shape: Sequence[int] = [0],
            device: Union[str, int, torch.device] = "cpu",
            hidden_layer_size: int = 128,
    ) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(
            input_size=int(np.prod(state_shape)),
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.fc2 = nn.Linear(hidden_layer_size + int(np.prod(action_shape)), 1)

    def forward(
            self,
            s: Union[np.ndarray, torch.Tensor],
            a: Optional[Union[np.ndarray, torch.Tensor]] = None,
            info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        s = torch.as_tensor(
            s, device=self.device, dtype=torch.float32)  # type: ignore
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        assert len(s.shape) == 3
        self.nn.flatten_parameters()
        s, (h, c) = self.nn(s)
        s = s[:, -1]
        if a is not None:
            a = torch.as_tensor(
                a, device=self.device, dtype=torch.float32)  # type: ignore
            s = torch.cat([s, a], dim=1)
        s = self.fc2(s)
        return s
