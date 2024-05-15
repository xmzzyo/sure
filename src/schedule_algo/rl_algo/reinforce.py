from typing import Any, Dict, List, Type, Union, Optional

import numpy as np
import torch
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import BasePolicy
from tianshou.utils import RunningMeanStd


class PGPolicy(BasePolicy):
    """Implementation of REINFORCE algorithm.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            optim: torch.optim.Optimizer,
            dist_fn: Type[torch.distributions.Distribution],
            discount_factor: float = 0.99,
            reward_normalization: bool = False,
            action_scaling: bool = True,
            action_bound_method: str = "clip",
            max_grad_norm: float = None,
            lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(action_scaling=action_scaling,
                         action_bound_method=action_bound_method, **kwargs)
        self.actor = model
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.dist_fn = dist_fn
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self._grad_norm = max_grad_norm
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8

    def process_fn(
            self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        r"""Compute the discounted returns for each transition.

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.
        """
        v_s_ = np.full(indice.shape, self.ret_rms.mean)
        unnormalized_returns, _ = self.compute_episodic_return(
            batch, buffer, indice, v_s_=v_s_, gamma=self._gamma, gae_lambda=1.0)
        if self._rew_norm:
            batch.returns = (unnormalized_returns - self.ret_rms.mean) / \
                            np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        return batch

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            with_loss=False,
            **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        obs = batch.obs
        with_loss = with_loss & self.training
        out = self.actor(obs, state=state, info={}, with_loss=with_loss)
        if with_loss:
            logits, h, loss_dict = out
        else:
            logits, h = out
        dist = self.dist_fn(*logits)
        act = dist.sample()
        if with_loss:
            return Batch(logits=logits, act=act, state=h, dist=dist), loss_dict
        else:
            return Batch(logits=logits, act=act, state=h, dist=dist)

    def learn(  # type: ignore
            self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, kld_losses, nl_losses, mi_losses = [], [], [], []
        for _ in range(repeat):
            for b in batch.split(batch_size, merge_last=True):
                self.optim.zero_grad()
                out = self(b, with_loss=True)
                if len(out) == 2:
                    result, loss_dict = out
                else:
                    result = out
                dist = result.dist
                a = to_torch_as(b.act, result.act)
                ret = to_torch_as(b.returns, result.act)
                log_prob = dist.log_prob(a).reshape(len(ret), -1).transpose(0, 1)
                loss = -(log_prob * ret).mean()
                loss += loss_dict["total_loss"]
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self._grad_norm)
                self.optim.step()
                kld_losses.append(loss_dict["KLD"].item())
                nl_losses.append(loss_dict["NL"].item())
                mi_losses.append(loss_dict["MI"].item())
                losses.append(loss.item())
        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "loss": losses,
            "loss/kld": kld_losses,
            "loss/nl": nl_losses,
            "loss/mi": mi_losses,
        }
