"""Vision agent for pixel observations.

Observations cross module boundaries exactly like the state-based case: flat
float32 tensors of shape ``(B, obs_dim)``, holding raw pixel values in
``[0, 255]`` (``normalize_obs`` must be off for pixel envs). The agent
reshapes to ``(C, H, W)`` and scales by 1/255 internally, so collectors and
the PPO update treat pixel and state observations identically.

The encoder is the Nature CNN (Mnih et al. 2015) with orthogonal
initialization; it has no BatchNorm, so it is safe under the vmapped
population (``FunctionalPolicyPopulation`` stacks buffers across policies).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from src.agents import ACTION_EPS, _orth, _squashed_log_prob


def _orth_conv(layer: nn.Conv2d, std: float = math.sqrt(2.0)) -> nn.Conv2d:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


def _head(hidden_dim: int, out_dim: int, std_out: float) -> nn.Sequential:
    return nn.Sequential(
        _orth(nn.Linear(hidden_dim, hidden_dim)),
        nn.ReLU(),
        _orth(nn.Linear(hidden_dim, out_dim), std=std_out),
    )


class NatureCNN(nn.Module):
    """Mnih et al. (2015) DQN encoder."""

    def __init__(self, image_shape: tuple[int, int, int], hidden_dim: int):
        super().__init__()
        c, h, w = image_shape
        self.conv = nn.Sequential(
            _orth_conv(nn.Conv2d(c, 32, 8, stride=4)),
            nn.ReLU(),
            _orth_conv(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            _orth_conv(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.fc = nn.Sequential(_orth(nn.Linear(flat, hidden_dim)), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class PixelMultiHeadAgent(nn.Module):
    """Pixel-input agent: shared Nature CNN, 2-layer MLP actor/task/descriptor heads.

    Drop-in interface match for ``MultiHeadAgent``; ``x`` is the flat raw
    pixel observation ``(B, obs_dim)`` in ``[0, 255]``.
    """

    discrete = False

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        descriptor_dim: int = 0,
        hidden_dim: int = 512,
        logstd_init: float = 0.0,
        image_shape: tuple[int, int, int] = (3, 84, 84),
        arch: str = "nature",
    ):
        super().__init__()
        if arch != "nature":
            raise ValueError(f"Unknown pixel encoder arch: {arch}")
        image_shape = tuple(int(s) for s in image_shape)
        if math.prod(image_shape) != obs_dim:
            raise ValueError(f"image_shape {image_shape} does not match obs_dim {obs_dim}")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.descriptor_dim = descriptor_dim
        self.image_shape = image_shape
        self.arch = arch
        self.logstd_init = float(logstd_init)

        self.encoder = NatureCNN(image_shape, hidden_dim)
        self.actor_mean = _head(hidden_dim, act_dim, std_out=0.01)
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), float(logstd_init)))
        self.value_r = _head(hidden_dim, 1, std_out=1.0)
        self.value_d = (
            _head(hidden_dim, descriptor_dim, std_out=1.0) if descriptor_dim > 0 else None
        )

    def ctor_kwargs(self) -> dict:
        return dict(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            descriptor_dim=self.descriptor_dim,
            hidden_dim=self.hidden_dim,
            logstd_init=self.logstd_init,
            image_shape=self.image_shape,
            arch=self.arch,
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        img = x.reshape(-1, *self.image_shape) / 255.0
        return self.encoder(img)

    def _values_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        if self.value_d is None:
            return self.value_r(feat)
        return torch.cat([self.value_r(feat), self.value_d(feat)], dim=-1)

    def get_values(self, x: torch.Tensor) -> torch.Tensor:
        """All value heads concatenated. x: (B, D) -> (B, 1+K)."""
        return self._values_from_features(self._features(x))

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) tanh-squashed action and return policy stats."""
        feat = self._features(x)
        mean = self.actor_mean(feat)
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
            self._values_from_features(feat),
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
            return torch.tanh(self.actor_mean(self._features(x)))
        if mode != "action_value":
            raise ValueError(f"Unknown PixelMultiHeadAgent forward mode: {mode}")
        return self.get_action_and_value(x, action)

    @torch.no_grad()
    def act_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Mean squashed action. ``x: (B, D) -> (B, A)``."""
        return torch.tanh(self.actor_mean(self._features(x)))


class AtariAgent(nn.Module):
    """CleanRL ``ppo_atari_envpool.py`` agent plus linear descriptor critics on the shared trunk.

    ``x`` is the flat uint8-valued frame stack ``(B, 4 * 84 * 84)``; module names match
    CleanRL's ``Agent`` (``network``, ``actor``, ``critic``) so its state dicts load directly.
    Sampling uses the Gumbel-max trick so it works under ``vmap(randomness="different")``.
    """

    discrete = True
    image_shape = (4, 84, 84)

    def __init__(self, obs_dim: int, act_dim: int, descriptor_dim: int = 0, hidden_dim: int = 512):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.descriptor_dim = descriptor_dim
        cnn = NatureCNN(self.image_shape, hidden_dim)
        self.network = nn.Sequential(*cnn.conv, *cnn.fc)
        self.actor = _orth(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.critic = _orth(nn.Linear(hidden_dim, 1), std=1.0)
        self.critic_d = (
            _orth(nn.Linear(hidden_dim, descriptor_dim), std=1.0) if descriptor_dim > 0 else None
        )

    def ctor_kwargs(self) -> dict:
        return dict(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            descriptor_dim=self.descriptor_dim,
            hidden_dim=self.hidden_dim,
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.reshape(-1, *self.image_shape) / 255.0)

    def _values_from_features(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.critic_d is None:
            return self.critic(hidden)
        return torch.cat([self.critic(hidden), self.critic_d(hidden)], dim=-1)

    def get_values(self, x: torch.Tensor) -> torch.Tensor:
        """All value heads concatenated. x: (B, D) -> (B, 1+K)."""
        return self._values_from_features(self._features(x))

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) a discrete action and return policy stats."""
        hidden = self._features(x)
        log_probs = F.log_softmax(self.actor(hidden), dim=-1)
        if action is None:
            u = torch.rand_like(log_probs).clamp_min(1e-10)
            idx = (log_probs - torch.log(-torch.log(u))).argmax(dim=-1)
        else:
            idx = action.squeeze(-1).long()
        logprob = log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs * log_probs.exp()).sum(-1)
        return (
            idx.unsqueeze(-1).to(x.dtype),
            logprob,
            entropy,
            self._values_from_features(hidden),
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
            return self.actor(self._features(x)).argmax(dim=-1, keepdim=True).to(x.dtype)
        if mode != "action_value":
            raise ValueError(f"Unknown AtariAgent forward mode: {mode}")
        return self.get_action_and_value(x, action)
