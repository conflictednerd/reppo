"""PPGA emitter: PPO + gradient arborescence inside the QD scheduler."""
import itertools
import logging

import numpy as np
import torch
from ribs.archives import ArchiveBase
from ribs.emitters._dqd_emitter_base import DQDEmitterBase
from ribs.emitters.opt import AdamOpt, GradientAscentOpt
from ribs.emitters.opt._xnes import ExponentialES
from ribs.emitters.rankers import _get_ranker

from src.baselines.ppga.ppo import PPO

log = logging.getLogger("rl")


class PPGAEmitter(DQDEmitterBase):
    def __init__(self,
                 ppo: PPO,
                 archive: ArchiveBase,
                 x0: np.ndarray,
                 sigma0: float,
                 batch_size: int = 100,
                 ranker: str = "2imp",
                 restart_rule: str = 'no_improvement',
                 normalize_grad: bool = True,
                 epsilon: float = 1e-8,
                 seed: int | None = None,
                 bounds: list[float] | None = None,
                 *,
                 grad_opt: str = 'ppo',
                 step_size: float | None = None,
                 normalize_obs: bool = True,
                 normalize_returns: bool = True,
                 adaptive_stddev: bool = True
                 ):
        DQDEmitterBase.__init__(self, archive, len(x0), bounds)
        self._epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._sigma0 = sigma0
        self._grad_coefficients = None
        self._normalize_grads = normalize_grad
        self._jacobian_batch = None
        self._ranker = _get_ranker(ranker)
        self._ranker.reset(self, archive, self._rng)
        self.ppo = ppo

        assert grad_opt in ['ppo', 'adam', 'gradient_ascent'], f'Invalid gradient optimizer passed in, {grad_opt=}'
        self._grad_opt = None
        if grad_opt == 'adam':
            self._grad_opt = AdamOpt(self._x0, step_size)
        elif grad_opt == 'gradient_ascent':
            self._grad_opt = GradientAscentOpt(self._x0, step_size)
        elif grad_opt == 'ppo':
            self.ppo.theta = self._x0
            self._grad_opt = self.ppo
        else:
            raise ValueError(f"Invalid Gradient Ascent Optimizer {grad_opt}")

        self._restart_rule = restart_rule
        self._restarts = 0
        self._itrs = 0
        _ = self._check_restart(0)
        self._restart_rule = restart_rule

        self._num_coefficients = archive.measure_dim + 1

        self.opt_seed = seed
        self.opt_lambda = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initial_bounds = ([-2.0] * (archive.measure_dim + 1), [2.0] * (archive.measure_dim + 1))
        self._initial_bounds[0][0] = 0.0
        self.opt = ExponentialES(solution_dim=self._num_coefficients,
                                 device=self.device,
                                 sigma0=sigma0,
                                 popsize=self.opt_lambda,
                                 seed=seed,
                                 initial_bounds=self._initial_bounds)

        self._batch_size = batch_size
        self._restarts = 0
        self._itrs = 0
        self._step_size = step_size
        self._normalize_obs = normalize_obs
        self._normalize_returns = normalize_returns
        self._mean_agent_obs_normalizer = None
        self._mean_agent_return_normalizer = None

    @property
    def x0(self):
        return self._x0

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batch_size_dqd(self):
        return 1

    @property
    def restarts(self):
        return self._restarts

    @property
    def itrs(self):
        return self._itrs

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def theta(self):
        return self._grad_opt.theta

    @property
    def mean_agent_obs_normalizer(self):
        return self._mean_agent_obs_normalizer

    @mean_agent_obs_normalizer.setter
    def mean_agent_obs_normalizer(self, obs_normalizer):
        self._mean_agent_obs_normalizer = obs_normalizer

    @property
    def mean_agent_return_normalizer(self):
        return self._mean_agent_return_normalizer

    @mean_agent_return_normalizer.setter
    def mean_agent_return_normalizer(self, rew_normalizer):
        self._mean_agent_return_normalizer = rew_normalizer

    def update_theta(self, new_theta):
        self._grad_opt.theta = new_theta

    def ask_dqd(self):
        return self._grad_opt.theta[None]

    def ask(self):
        self._grad_coefficients = self.opt.ask()
        noise = np.expand_dims(self._grad_coefficients, axis=2)
        return self._grad_opt.theta + np.sum(
            np.multiply(self._jacobian_batch, noise), axis=1)

    def _check_restart(self, num_parents):
        if isinstance(self._restart_rule, (int, np.integer)):
            return self._itrs % self._restart_rule == 0
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        if self._restart_rule == "basic":
            return False
        raise ValueError(f"Invalid restart_rule {self._restart_rule}")

    def tell_dqd(self,
                 solution_batch,
                 objective_batch,
                 measures_batch,
                 jacobian_batch,
                 status_batch,
                 value_batch,
                 metadata_batch=None):
        if self._normalize_grads:
            norms = (np.linalg.norm(jacobian_batch, axis=2, keepdims=True) +
                     self._epsilon)
            jacobian_batch /= norms
        self._jacobian_batch = jacobian_batch

    def tell(self,
             solution_batch,
             objective_batch,
             measures_batch,
             status_batch,
             value_batch,
             metadata_batch=None):
        if self._jacobian_batch is None:
            raise RuntimeError("tell() was called without calling tell_dqd().")

        metadata_batch = itertools.repeat(
            None) if metadata_batch is None else metadata_batch

        new_sols = status_batch.astype(bool).sum()

        indices, ranking_values = self._ranker.rank(
            self, self.archive, self._rng, solution_batch, objective_batch,
            measures_batch, status_batch, value_batch, metadata_batch)

        num_parents = self._batch_size

        value_batch_parents = value_batch[indices][:num_parents]
        mean_value = np.mean(value_batch_parents)
        max_value = np.max(value_batch)
        log.debug(f'{mean_value=}, {max_value=}')
        coeffs, coeff_rankings = self._grad_coefficients[indices], ranking_values[indices]
        coeffs, coeff_rankings = coeffs[:num_parents], coeff_rankings[:num_parents]
        self.opt.tell(value_batch)

        stop_status = self.opt.check_stop(ranking_values) or self._check_restart(new_sols)
        if stop_status:
            if self.archive.stats.num_elites > 0:
                new_elite = self.archive.sample_elites(1)
                new_theta, measures, obj = new_elite.solution_batch[0], new_elite.measures_batch[0], new_elite.objective_batch[0]
                log.debug(f'XNES is restarting with a new solution whose measures are {measures} and objective is {obj}')
                if self._normalize_obs:
                    self.mean_agent_obs_normalizer.load_state_dict(metadata_batch[0]['obs_normalizer'])
                if self._normalize_returns:
                    self._mean_agent_return_normalizer.load_state_dict(metadata_batch[0]['return_normalizer'])
                self._grad_opt.theta = new_theta
            else:
                log.debug('XNES is restarting with an empty archive; keeping current theta')
            self.opt = ExponentialES(self._num_coefficients,
                                     self.device,
                                     self._sigma0,
                                     self.opt_lambda,
                                     seed=self.opt_seed,
                                     initial_bounds=self._initial_bounds)
            self._ranker.reset(self, self.archive, self._rng)
            self._restarts += 1

        self._itrs += 1

        return stop_status
