from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.func import functional_call, stack_module_state, vmap

LOG_2 = math.log(2.0)
ACTION_EPS = 1e-6


def _orth(layer: nn.Linear, std: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(in_dim: int, out_dim: int, hidden: int, std_out: float) -> nn.Sequential:
    """H-H tanh MLP with orthogonal init."""
    return nn.Sequential(
        _orth(nn.Linear(in_dim, hidden)),
        nn.Tanh(),
        _orth(nn.Linear(hidden, hidden)),
        nn.Tanh(),
        _orth(nn.Linear(hidden, out_dim), std=std_out),
    )


def _squashed_log_prob(dist: Normal, pre_tanh: torch.Tensor) -> torch.Tensor:
    correction = 2.0 * (LOG_2 - pre_tanh - F.softplus(-2.0 * pre_tanh))
    return (dist.log_prob(pre_tanh) - correction).sum(-1)


class MultiHeadAgent(nn.Module):
    """Continuous-action agent with task and (optional) descriptor critics."""

    discrete = False

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        descriptor_dim: int = 0,
        hidden_dim: int = 64,
        logstd_init: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.descriptor_dim = descriptor_dim
        self.actor_mean = _mlp(obs_dim, act_dim, hidden_dim, std_out=0.01)
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), float(logstd_init)))
        self.value_r = _mlp(obs_dim, 1, hidden_dim, std_out=1.0)
        self.value_d = (
            _mlp(obs_dim, descriptor_dim, hidden_dim, std_out=1.0) if descriptor_dim > 0 else None
        )

    def ctor_kwargs(self) -> dict:
        return dict(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            descriptor_dim=self.descriptor_dim,
            hidden_dim=self.hidden_dim,
        )

    def get_values(self, x: torch.Tensor) -> torch.Tensor:
        """All value heads concatenated. x: (B, D) -> (B, 1+K)."""
        if self.value_d is None:
            return self.value_r(x)
        return torch.cat([self.value_r(x), self.value_d(x)], dim=-1)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) tanh-squashed action and return policy stats."""
        mean = self.actor_mean(x)
        std = self.actor_logstd.expand_as(mean).exp()
        dist = Normal(mean, std, validate_args=False)
        if action is None:
            pre_tanh = dist.sample()
            action = torch.tanh(pre_tanh)
        else:
            action = action.clamp(-1.0 + ACTION_EPS, 1.0 - ACTION_EPS)
            pre_tanh = torch.atanh(action)
        return (
            action,
            _squashed_log_prob(dist, pre_tanh),
            dist.entropy().sum(-1),
            self.get_values(x),
        )

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        mode: str = "action_value",
    ):
        if mode == "values":
            return self.get_values(x)
        if mode == "deterministic":
            return torch.tanh(self.actor_mean(x))
        if mode != "action_value":
            raise ValueError(f"Unknown MultiHeadAgent forward mode: {mode}")
        return self.get_action_and_value(x, action)

    @torch.no_grad()
    def act_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Mean squashed action. ``x: (B, D) -> (B, A)``."""
        return torch.tanh(self.actor_mean(x))


class CategoricalAgent(nn.Module):
    """Discrete-action agent with task and (optional) descriptor critics.

    ``act_dim`` is the number of discrete actions. Actions cross module
    boundaries as float index tensors of shape ``(B, 1)`` so rollout buffers
    and PPO batch reshapes are identical to the continuous case. Sampling
    uses the Gumbel-max trick so it works under ``vmap(randomness="different")``.
    """

    discrete = True

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        descriptor_dim: int = 0,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.descriptor_dim = descriptor_dim
        self.actor_logits = _mlp(obs_dim, act_dim, hidden_dim, std_out=0.01)
        self.value_r = _mlp(obs_dim, 1, hidden_dim, std_out=1.0)
        self.value_d = (
            _mlp(obs_dim, descriptor_dim, hidden_dim, std_out=1.0) if descriptor_dim > 0 else None
        )

    def ctor_kwargs(self) -> dict:
        return dict(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            descriptor_dim=self.descriptor_dim,
            hidden_dim=self.hidden_dim,
        )

    def get_values(self, x: torch.Tensor) -> torch.Tensor:
        """All value heads concatenated. x: (B, D) -> (B, 1+K)."""
        if self.value_d is None:
            return self.value_r(x)
        return torch.cat([self.value_r(x), self.value_d(x)], dim=-1)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) a discrete action and return policy stats."""
        logits = self.actor_logits(x)
        log_probs = F.log_softmax(logits, dim=-1)
        if action is None:
            u = torch.rand_like(logits).clamp_min(1e-10)
            idx = (logits - torch.log(-torch.log(u))).argmax(dim=-1)
        else:
            idx = action.squeeze(-1).long()
        logprob = log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs * log_probs.exp()).sum(-1)
        return idx.unsqueeze(-1).to(x.dtype), logprob, entropy, self.get_values(x)

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        mode: str = "action_value",
    ):
        if mode == "values":
            return self.get_values(x)
        if mode == "deterministic":
            return self.actor_logits(x).argmax(dim=-1, keepdim=True).to(x.dtype)
        if mode != "action_value":
            raise ValueError(f"Unknown CategoricalAgent forward mode: {mode}")
        return self.get_action_and_value(x, action)

    @torch.no_grad()
    def act_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Argmax action index as float. ``x: (B, D) -> (B, 1)``."""
        return self.actor_logits(x).argmax(dim=-1, keepdim=True).to(x.dtype)


def make_agent(
    obs_dim: int,
    act_dim: int,
    *,
    descriptor_dim: int,
    hidden_dim: int,
    discrete: bool,
    logstd_init: float = 0.0,
    spec: dict | None = None,
) -> nn.Module:
    """``act_dim`` is the number of actions when discrete, action dim otherwise.

    ``spec`` selects a pixel architecture (``{"arch": ..., "image_shape": ...,
    ...}``); ``None`` or ``{"arch": "mlp"}`` gives the state-based MLP agents.
    """
    arch = (spec or {}).get("arch", "mlp")
    if arch == "cleanrl_atari":
        from src.agents_pixels import AtariAgent

        return AtariAgent(obs_dim, act_dim, descriptor_dim=descriptor_dim, hidden_dim=hidden_dim)
    if arch != "mlp":
        if discrete:
            raise NotImplementedError("Pixel agents only support continuous actions")
        from src.agents_pixels import PixelMultiHeadAgent

        return PixelMultiHeadAgent(
            obs_dim,
            act_dim,
            descriptor_dim=descriptor_dim,
            hidden_dim=hidden_dim,
            logstd_init=logstd_init,
            image_shape=tuple(spec["image_shape"]),
            arch=arch,
        )
    if discrete:
        return CategoricalAgent(
            obs_dim, act_dim, descriptor_dim=descriptor_dim, hidden_dim=hidden_dim
        )
    return MultiHeadAgent(
        obs_dim,
        act_dim,
        descriptor_dim=descriptor_dim,
        hidden_dim=hidden_dim,
        logstd_init=logstd_init,
    )


class FunctionalPolicyPopulation(nn.Module):
    """Functional wrapper around a stack of identically shaped agent policies."""

    def __init__(self, agents: list[nn.Module]):
        super().__init__()
        if not agents:
            raise ValueError("FunctionalPolicyPopulation requires at least one policy")

        first = agents[0]
        self.N = len(agents)
        self.obs_dim = first.obs_dim
        self.act_dim = first.act_dim
        self.hidden_dim = first.hidden_dim
        self.descriptor_dim = first.descriptor_dim
        self.discrete = first.discrete
        agent_cls = type(first)
        # Plain dict (not a submodule/buffer), used to rebuild single agents.
        self._agent_ctor_kwargs = dict(first.ctor_kwargs())

        for agent in agents:
            if type(agent) is not agent_cls or agent.ctor_kwargs() != self._agent_ctor_kwargs:
                raise ValueError("All policies in a population must have identical shapes")

        params, buffers = stack_module_state(agents)
        self.param_names = tuple(params.keys())
        self.params = nn.ParameterList([nn.Parameter(p.detach().clone()) for p in params.values()])
        self.buffer_names = tuple(buffers.keys())
        for i, buffer in enumerate(buffers.values()):
            self.register_buffer(f"_stacked_buffer_{i}", buffer.detach().clone())

        template = agent_cls(**self._agent_ctor_kwargs)
        # We don't do self._template = template since that would register template as a
        # (trainable) submodule.
        object.__setattr__(self, "_template", template)

    def _param_dict(self) -> dict[str, torch.Tensor]:
        return dict(zip(self.param_names, self.params, strict=True))

    def _buffer_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, f"_stacked_buffer_{i}") for i, name in enumerate(self.buffer_names)
        }

    def get_values(self, x: torch.Tensor) -> torch.Tensor:
        """All value heads. x: (N, B, D) -> (N, B, 1+K)."""

        def call(params, buffers, obs):
            return functional_call(
                self._template,
                (params, buffers),
                (obs,),
                {"mode": "values"},
            )

        return vmap(call)(self._param_dict(), self._buffer_dict(), x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate actions for all policies in parallel."""
        if action is None:

            def call(params, buffers, obs):
                return functional_call(self._template, (params, buffers), (obs,))

            return vmap(call, randomness="different")(self._param_dict(), self._buffer_dict(), x)

        def call(params, buffers, obs, act):
            return functional_call(self._template, (params, buffers), (obs, act))

        return vmap(call)(self._param_dict(), self._buffer_dict(), x, action)

    @torch.no_grad()
    def act_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Mean actions. x: (N, B, D) -> (N, B, A)."""

        def call(params, buffers, obs):
            return functional_call(
                self._template,
                (params, buffers),
                (obs,),
                {"mode": "deterministic"},
            )

        return vmap(call)(self._param_dict(), self._buffer_dict(), x)

    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        """Clip gradients independently for each policy."""
        total_sq = None
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad.detach().reshape(self.N, -1)
            sq = grad.pow(2).sum(dim=1)
            total_sq = sq if total_sq is None else total_sq + sq

        if total_sq is None:
            return torch.zeros(self.N, device=self.params[0].device)

        total_norm = total_sq.sqrt()
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
        for param in self.params:
            if param.grad is not None:
                param.grad.mul_(clip_coef.view(self.N, *([1] * (param.grad.dim() - 1))))
        return total_norm

    def num_params_per_policy(self) -> int:
        return sum(param[0].numel() for param in self.params)

    def policy_state_dict(self, n: int) -> OrderedDict[str, torch.Tensor]:
        state = OrderedDict(
            (name, param[n].detach().cpu().clone())
            for name, param in zip(self.param_names, self.params, strict=True)
        )
        for i, name in enumerate(self.buffer_names):
            state[name] = getattr(self, f"_stacked_buffer_{i}")[n].detach().cpu().clone()
        return state

    def policy_state_dicts(self) -> list[OrderedDict[str, torch.Tensor]]:
        return [self.policy_state_dict(n) for n in range(self.N)]
