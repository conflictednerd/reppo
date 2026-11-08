"""RePPO: N PPO policies trained jointly, each with a repulsion-shaped advantage.

Per-policy ``obs_rms`` is owned by the rollout collector so it is saved with
the checkpoint and frozen at evaluation. ``force_c_zero=True`` gives
Independent PPO.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.agents import FunctionalPolicyPopulation, make_agent
from src.envs import (
    descriptor_dim,
    descriptor_names,
    make_rollout_collector,
)
from src.utils import (
    MetricLogger,
    PopulationState,
    compute_diversity_coefficients,
    compute_gae,
    vendi_score,
)


def build_agent_spec(cfg: DictConfig, env_cfg: DictConfig) -> dict | None:
    """Pixel-architecture spec from cfg.policy + env_cfg.pixels; None for MLP."""
    policy = cfg.get("policy", None)
    if policy is None or str(policy.get("arch", "mlp")) == "mlp":
        return None
    if str(policy.arch) == "cleanrl_atari":
        return {"arch": "cleanrl_atari"}

    pixels = env_cfg.get("pixels", None)
    if pixels is None or not bool(pixels.get("enabled", False)):
        raise ValueError("pixel policies require env_cfg.pixels.enabled=true")
    if bool(env_cfg.normalize_obs):
        raise ValueError("pixels must not be RMS-normalized; the agent scales by 1/255")

    channels = (1 if pixels.get("grayscale", True) else 3) * int(pixels.frame_stack)
    image_shape = [channels, int(pixels.image_size), int(pixels.image_size)]
    return {"arch": str(policy.arch), "image_shape": image_shape}


class RePPO:
    def __init__(
        self,
        cfg: DictConfig,
        env_cfg: DictConfig,
        *,
        exp_dir: Path,
        device: torch.device,
        seed: int,
        logger: MetricLogger,
    ):
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.exp_dir = Path(exp_dir)
        self.device = device
        self.seed = seed
        self.logger = logger

        self.N = int(cfg.N)  # Number of policies
        self.E = int(cfg.num_envs_per_policy)
        self.T = int(cfg.num_steps)  # Per env rollout length
        self.K = descriptor_dim(env_cfg.descriptor_set)  # Descriptor dim
        self.descriptor_names = descriptor_names(env_cfg.descriptor_set)
        self.logstd_init = float(cfg.logstd_init)
        self.agent_spec = build_agent_spec(cfg, env_cfg)

        self.rollout_collector = make_rollout_collector(
            env_cfg,
            N=self.N,
            E=self.E,
            T=self.T,
            K=self.K,
            hidden_dim=cfg.hidden_dim,
            device=device,
            seed=seed,
            agent_spec=self.agent_spec,
        )
        self.obs_shape = self.rollout_collector.obs_shape
        self.act_shape = self.rollout_collector.act_shape
        self.obs_dim = self.rollout_collector.obs_dim
        self.act_dim = self.rollout_collector.act_dim
        self.discrete = self.rollout_collector.discrete
        self.agent_act_dim = self.rollout_collector.agent_act_dim

        self.batch_per_policy = self.E * self.T
        self.minibatch_per_policy = self.batch_per_policy // cfg.num_minibatches
        self.steps_per_iter = self.N * self.E * self.T
        self.num_iters = cfg.total_timesteps // self.steps_per_iter
        self.lr_schedule_timesteps = int(cfg.get("lr_schedule_timesteps") or cfg.total_timesteps)

        agents = [
            make_agent(
                self.obs_dim,
                self.agent_act_dim,
                descriptor_dim=self.K,
                hidden_dim=cfg.hidden_dim,
                discrete=self.discrete,
                logstd_init=self.logstd_init,
                spec=self.agent_spec,
            )
            for _ in range(self.N)
        ]
        self.agent_pop = FunctionalPolicyPopulation(agents).to(device)
        self.optim = optim.Adam(self.agent_pop.parameters(), lr=cfg.learning_rate, eps=1e-5)

        self.pop = PopulationState.initialize(self.N, self.K, eps_f=cfg.eps_f, device=device)
        self.rollout_collector.start(self.agent_pop)

        (self.exp_dir / "checkpoints").mkdir(exist_ok=True, parents=True)

        n_params = self.agent_pop.num_params_per_policy()
        self.logger.info(
            f"[RePPO] env={env_cfg.env_id} N={self.N} E={self.E} T={self.T} K={self.K} "
            f"steps/iter={self.steps_per_iter} iters={self.num_iters} "
            f"batch/policy={self.batch_per_policy} mb/policy={self.minibatch_per_policy} "
            f"params/policy={n_params} rollout_workers={self.rollout_collector.num_policy_workers} "
            f"env_workers={self.rollout_collector.num_env_workers} "
            f"arch={self.agent_spec['arch'] if self.agent_spec else 'mlp'} "
            f"force_c_zero={cfg.force_c_zero} repulsion_mode={cfg.repulsion_mode} "
            f"use_exact_wk={cfg.use_exact_wk} "
            f"scale_descriptor_returns={cfg.scale_descriptor_returns} "
            f"lr_anneal={cfg.lr_anneal} gamma_sq={cfg.gamma_sq} beta_b={cfg.beta_b} "
            f"logstd_init={self.logstd_init} "
            f"lr_schedule_timesteps={self.lr_schedule_timesteps}"
        )

    def _update(self, c_coeff: torch.Tensor, w_k: torch.Tensor) -> dict[str, float]:
        """Compute GAE for task and diversity rewards and per-policy PPO updates."""
        T, N, E = self.T, self.N, self.E
        batch = self.rollout_collector.batch

        # [1] ===== Compute advantages =====
        with torch.no_grad():
            next_values = self.agent_pop.get_values(self.rollout_collector.next_obs)
            descriptor_rewards = batch.descriptors
            if self.cfg.scale_descriptor_returns:
                # Descriptor returns are ~|d|/(1-gamma); this keeps their value loss on the task scale.
                descriptor_rewards = descriptor_rewards * (1.0 - self.cfg.gamma)
            channel_rewards = torch.cat([batch.rewards.unsqueeze(-1), descriptor_rewards], dim=-1)
            dones = batch.dones.view(T, N * E)
            next_done = self.rollout_collector.next_done.view(N * E)
            if batch.descriptor_dones is not None:
                d_dones = batch.descriptor_dones.view(T, N * E, 1).expand(-1, -1, self.K)
                d_next = self.rollout_collector.next_descriptor_done.view(N * E, 1)
                dones = torch.cat([dones.unsqueeze(-1), d_dones], dim=-1)
                next_done = torch.cat([next_done.unsqueeze(-1), d_next.expand(-1, self.K)], dim=-1)
            advantages, returns = compute_gae(
                rewards=channel_rewards.view(T, N * E, 1 + self.K),
                values=batch.values.view(T, N * E, 1 + self.K),
                dones=dones,
                next_value=next_values.view(N * E, 1 + self.K),
                next_done=next_done,
                gamma=self.cfg.gamma,
                gae_lambda=self.cfg.gae_lambda,
            )
            advantages = advantages.view(T, N, E, 1 + self.K)
            returns = returns.view(T, N, E, 1 + self.K)

        # [2] ===== PPO Update for all policies =====
        bsz = self.batch_per_policy
        minibatch_per_policy = self.minibatch_per_policy

        b_obs = batch.obs.permute(1, 0, 2, 3).reshape(N, bsz, self.obs_dim)
        b_actions = batch.actions.permute(1, 0, 2, 3).reshape(N, bsz, self.act_dim)
        b_logprobs = batch.logprobs.permute(1, 0, 2).reshape(N, bsz)
        b_v_old = batch.values.permute(1, 0, 2, 3).reshape(N, bsz, 1 + self.K)
        b_adv = advantages.permute(1, 0, 2, 3).reshape(N, bsz, 1 + self.K)
        b_ret = returns.permute(1, 0, 2, 3).reshape(N, bsz, 1 + self.K)

        sums = {
            "pg": torch.zeros(N, device=self.device),
            "v_r": torch.zeros(N, device=self.device),
            "v_d": torch.zeros(N, device=self.device),
            "ent": torch.zeros(N, device=self.device),
            "kl": torch.zeros(N, device=self.device),
            "clip": torch.zeros(N, device=self.device),
        }
        n_mb = 0
        grad_norm = torch.zeros(N, device=self.device)

        for _epoch in range(self.cfg.update_epochs):
            perms = torch.stack([torch.randperm(bsz, device=self.device) for _ in range(N)])
            for i in range(0, bsz, minibatch_per_policy):
                idx = perms[:, i : i + minibatch_per_policy]
                idx_obs = idx.unsqueeze(-1).expand(N, minibatch_per_policy, self.obs_dim)
                idx_act = idx.unsqueeze(-1).expand(N, minibatch_per_policy, self.act_dim)
                idx_val = idx.unsqueeze(-1).expand(N, minibatch_per_policy, 1 + self.K)

                obs_mb = b_obs.gather(1, idx_obs)
                if obs_mb.dtype != torch.float32:  # uint8 pixel buffer
                    obs_mb = obs_mb.to(torch.float32)
                actions_mb = b_actions.gather(1, idx_act)
                logprobs_mb = b_logprobs.gather(1, idx)
                ret_mb = b_ret.gather(1, idx_val)
                adv_mb = b_adv.gather(1, idx_val)
                v_old_mb = b_v_old.gather(1, idx_val)

                # Advantage normalization: Normalize each reward's advantage then add together
                if self.cfg.norm_adv:
                    adv_norm = (adv_mb - adv_mb.mean(1, keepdim=True)) / (
                        adv_mb.std(1, keepdim=True) + 1e-8
                    )
                else:
                    adv_norm = adv_mb
                adv_combined = w_k.unsqueeze(1) * adv_norm[:, :, 0]
                if self.K > 0:
                    adv_combined = adv_combined + (adv_norm[:, :, 1:] * c_coeff.unsqueeze(1)).sum(
                        -1
                    )

                _, new_lp, ent, new_v = self.agent_pop.get_action_and_value(obs_mb, actions_mb)
                logratio = new_lp - logprobs_mb
                ratio = logratio.exp()
                with torch.no_grad():
                    kl = ((ratio - 1) - logratio).mean(dim=1)
                    clipfrac = ((ratio - 1).abs() > self.cfg.clip_coef).float().mean(dim=1)

                pg = torch.maximum(
                    -adv_combined * ratio,
                    -adv_combined
                    * torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef),
                ).mean(dim=1)

                if self.cfg.clip_vloss:
                    v_unc = (new_v - ret_mb) ** 2
                    v_clp_pred = v_old_mb + torch.clamp(
                        new_v - v_old_mb, -self.cfg.clip_coef, self.cfg.clip_coef
                    )
                    v_per_head = 0.5 * torch.maximum(v_unc, (v_clp_pred - ret_mb) ** 2)
                else:
                    v_per_head = 0.5 * (new_v - ret_mb) ** 2
                v_loss_r = v_per_head[:, :, 0].mean(dim=1)
                v_loss_d = (
                    v_per_head[:, :, 1:].mean(dim=(1, 2))
                    if self.K > 0
                    else torch.zeros(N, device=self.device)
                )
                v_loss = v_loss_r + v_loss_d

                loss = pg + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * ent.mean(dim=1)
                self.optim.zero_grad(set_to_none=True)
                loss.sum().backward()
                grad_norm = self.agent_pop.clip_grad_norm_(self.cfg.max_grad_norm)
                self.optim.step()

                sums["pg"] += pg.detach()
                sums["v_r"] += v_loss_r.detach()
                sums["v_d"] += v_loss_d.detach()
                sums["ent"] += ent.mean(dim=1).detach()
                sums["kl"] += kl.detach()
                sums["clip"] += clipfrac.detach()
                n_mb += 1

        with torch.no_grad():
            ret_r = b_ret[:, :, 0]
            v_r_old = b_v_old[:, :, 0]
            explained_var_r = 1 - (ret_r - v_r_old).var(dim=1) / (ret_r.var(dim=1) + 1e-8)

        per_policy = {
            "policy_loss": sums["pg"] / n_mb,
            "value_loss_r": sums["v_r"] / n_mb,
            "value_loss_d": sums["v_d"] / n_mb,
            "entropy": sums["ent"] / n_mb,
            "approx_kl": sums["kl"] / n_mb,
            "clipfrac": sums["clip"] / n_mb,
            "explained_variance_r": explained_var_r,
            "grad_norm": grad_norm.detach(),
        }
        out = {f"train/{k}_mean": float(v.mean().item()) for k, v in per_policy.items()}
        for k, v in per_policy.items():
            out[f"train/{k}_per_policy"] = v.detach().cpu().tolist()
        return out

    @torch.no_grad()
    def evaluate(
        self,
        num_episodes: int,
    ) -> dict:
        return self.rollout_collector.evaluate(self.agent_pop, num_episodes)

    def save(self, tag: str) -> None:
        path = self.exp_dir / "checkpoints" / f"{tag}.pt"
        torch.save(
            {
                "agent_states": self.agent_pop.policy_state_dicts(),
                "obs_rms": [
                    r.state_dict() if r is not None else None
                    for r in self.rollout_collector.obs_rms
                ],
                "f_hat": self.pop.f_hat.detach().cpu(),
                "b_hat": self.pop.b_hat.detach().cpu(),
                "env_id": self.env_cfg.env_id,
                "descriptor_set": self.env_cfg.descriptor_set,
                "env_config": OmegaConf.to_container(self.env_cfg, resolve=True),
                "obs_dim": self.obs_dim,
                "act_dim": self.agent_act_dim,
                "discrete": self.discrete,
                "K": self.K,
                "N": self.N,
                "hidden_dim": self.cfg.hidden_dim,
                "agent_spec": self.agent_spec,
                "clip_obs": self.rollout_collector.clip_obs,
            },
            path,
        )
        self.logger.info(f"Saved population checkpoint to {path}")

    def train(self) -> None:
        global_step = 0
        t0 = time.time()

        for itr in tqdm(
            range(1, self.num_iters + 1), desc="RePPO", unit="iter", dynamic_ncols=True
        ):
            iter_start = time.time()
            if self.cfg.anneal_lr:
                lr_progress = ((itr - 1) * self.steps_per_iter) / max(1, self.lr_schedule_timesteps)
                # Linear decay at twice the nominal rate, floored at 5% of the base
                # rate: the floor is reached halfway through lr_schedule_timesteps.
                lr = self.cfg.learning_rate * max(0.05, 1.0 - 2 * lr_progress)
                if self.cfg.lr_anneal == "linear":
                    # CleanRL: linear decay to 0 over the run's iterations.
                    lr = self.cfg.learning_rate * (1.0 - (itr - 1) / self.num_iters)
                self.optim.param_groups[0]["lr"] = lr

            stats = self.rollout_collector.collect(self.agent_pop)
            global_step += self.steps_per_iter
            rollout_t = time.time() - iter_start

            f_ep = torch.tensor(
                [float(np.mean(rs)) if rs else 0.0 for rs in stats["ep_returns"]],
                device=self.device,
                dtype=torch.float32,
            )
            # If no episode has finished for an agent, we shouldn't update its EMA
            ema_mask = torch.tensor(
                [bool(rs) for rs in stats["ep_returns"]], device=self.device, dtype=torch.bool
            )
            b_ep = torch.stack(
                [
                    (
                        torch.as_tensor(
                            np.stack(bd, axis=0), device=self.device, dtype=torch.float32
                        ).mean(dim=0)
                    )
                    if bd
                    else torch.zeros(self.K, device=self.device, dtype=torch.float32)
                    for bd in stats["ep_behavior_descriptors"]
                ],
                dim=0,
            )
            # EMA update
            self.pop.update(
                f_ep=f_ep,
                b_ep=b_ep,
                beta_f=self.cfg.beta_f,
                beta_b=self.cfg.beta_b,
                ema_mask=ema_mask,
            )

            coeffs = compute_diversity_coefficients(
                self.pop,
                K_nn=self.cfg.K_nn,
                gamma_sq=self.cfg.gamma_sq,
                eps_f=self.cfg.eps_f,
                eps_b=self.cfg.eps_b,
                C_max=self.cfg.C_max,
                repulsion_mode=self.cfg.repulsion_mode,
                use_exact_wk=self.cfg.use_exact_wk,
            )
            if self.cfg.force_c_zero:
                coeffs.c_coeff.zero_()
                coeffs.w_k.fill_(1.0)

            # PPO updates
            update_start = time.time()
            train_m = self._update(coeffs.c_coeff, coeffs.w_k)
            update_t = time.time() - update_start

            iter_t = time.time() - iter_start
            sps = int(global_step / max(1e-6, time.time() - t0))

            # Logging
            metrics = {
                "iter": itr,
                "descriptor_names": self.descriptor_names,
                "charts/lr": self.optim.param_groups[0]["lr"],
                "charts/sps": sps,
                "charts/iter_sps": int(self.steps_per_iter / max(1e-6, iter_t)),
                "charts/rollout_seconds": rollout_t,
                "charts/update_seconds": update_t,
                "quality/mean_fitness_ema": float(self.pop.f_hat.mean().item()),
                "quality/max_fitness_ema": float(self.pop.f_hat.max().item()),
                "quality/min_fitness_ema": float(self.pop.f_hat.min().item()),
                "diversity/c_norm_mean": float(coeffs.c_coeff.norm(dim=1).mean().item()),
                "diversity/c_norm_max": float(coeffs.c_coeff.norm(dim=1).max().item()),
                "diversity/b_spread": (
                    0.0 if self.N <= 1 else float(self.pop.b_hat.std(dim=0).mean().item())
                ),
                "diversity/b_prime_spread": (
                    0.0 if self.N <= 1 else float(coeffs.b_prime.std(dim=0).mean().item())
                ),
                "diversity/vendi_score": vendi_score(self.pop.b_hat),
                "f_hat": self.pop.f_hat.detach().cpu().tolist(),
                "b_hat": self.pop.b_hat.detach().cpu().tolist(),
                "c_coeff": coeffs.c_coeff.detach().cpu().tolist(),
                "b_prime": coeffs.b_prime.detach().cpu().tolist(),
                "w_k": coeffs.w_k.detach().cpu().tolist(),
                **train_m,
            }
            ep_flat = [r for rs in stats["ep_returns"] for r in rs]
            if ep_flat:
                ep_lens_flat = [length for lengths in stats["ep_lengths"] for length in lengths]
                metrics["charts/ep_return_mean"] = float(np.mean(ep_flat))
                metrics["charts/ep_return_max"] = float(np.max(ep_flat))
                metrics["charts/ep_return_min"] = float(np.min(ep_flat))
                metrics["charts/episodes_completed"] = len(ep_flat)
                metrics["charts/episodes_per_1k_steps"] = float(
                    1000.0 * len(ep_flat) / max(1, self.steps_per_iter)
                )
                if ep_lens_flat:
                    metrics["charts/ep_len_mean"] = float(np.mean(ep_lens_flat))
                    metrics["charts/ep_len_min"] = int(np.min(ep_lens_flat))
                    metrics["charts/ep_len_max"] = int(np.max(ep_lens_flat))
                metrics["charts/per_policy_ep_return_mean"] = [
                    float(np.mean(rs)) if rs else float("nan") for rs in stats["ep_returns"]
                ]
                metrics["charts/per_policy_ep_len_mean"] = [
                    float(np.mean(ls)) if ls else float("nan") for ls in stats["ep_lengths"]
                ]

            if self.cfg.eval.enabled and itr % self.cfg.eval.every_iters == 0:
                with torch.no_grad():
                    metrics.update(self.evaluate(self.cfg.eval.num_episodes))

            summary = None
            if itr % self.cfg.log_every_iters == 0:
                ret = metrics.get("charts/ep_return_mean", float("nan"))
                ep_len = metrics.get("charts/ep_len_mean", float("nan"))
                summary = (
                    f"[RePPO iter={itr}/{self.num_iters} step={global_step} sps={sps} "
                    f"(roll={rollout_t:.1f}s upd={update_t:.1f}s)] "
                    f"ep_ret={ret:.1f} ep_len={ep_len:.0f} "
                    f"c_norm={metrics['diversity/c_norm_mean']:.2f} "
                    f"b_spread={metrics['diversity/b_spread']:.3f} "
                    f"vendi={metrics['diversity/vendi_score']:.2f} "
                    f"kl={train_m['train/approx_kl_mean']:.3f}"
                )
            self.logger.log(global_step, metrics, summary=summary)

            if self.cfg.checkpoint_every_iters and itr % self.cfg.checkpoint_every_iters == 0:
                self.save(f"iter{itr}")

    def close(self) -> None:
        self.rollout_collector.close()
