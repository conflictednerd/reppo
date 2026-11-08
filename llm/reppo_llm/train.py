"""RePPO for a population of LoRA policies on one frozen base, on math reasoning: fitness is the correctness of the
last \\boxed{} answer (math_correct), descriptors are reasoning-style text features (descriptors.py).
reppo.force_c_zero=true gives independent PPO members (Ind-PPO); member_prompts (one instruction per member, added
before the problem) with iterations=0 is the prompting baseline.

Usage: python -m reppo_llm.train conf/math.yaml conf/math_<model>.yaml [key=value ...]   (source scripts/env.sh;
scripts/launch.sh runs one process per GPU)

One iteration: sync adapters to vLLM -> generate samples_per_prompt responses per prompt per member (same prompts for
all members) -> fitness + descriptors -> EMA fitness/descriptors and diversity coefficients -> per-member PPO update.

Member-parallel runs (scripts/launch.sh) start one process per GPU, each owning N / world members with its own vLLM
engine; the processes exchange per-member fitness and descriptor means (and the initial normalization statistics)
through a TCP store, so every process computes the same diversity coefficients.
"""

import gzip
import importlib.util
import json
import os
import random
import shutil
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from math_verify import parse, verify
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from peft.tuners.lora import Linear as LoraLinear
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from reppo_llm.descriptors import Descriptors, last_boxed
from reppo_llm.ppo import (
    gae,
    gather_hidden,
    masked_mean,
    masked_whiten,
    policy_loss,
    response_positions,
    token_budget_chunks,
    token_logprobs,
    value_loss,
)

REPO = Path(__file__).resolve().parents[2]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# src/utils/population.py, loaded by path so that the root package and its dependencies are not imported
_spec = importlib.util.spec_from_file_location("population", REPO / "src/utils/population.py")
population = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(population)

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def math_correct(text: str, answer: str, stopped: bool) -> float:
    """1 if the response ended (EOS or a stop string) and its last \\boxed{...} matches the reference answer under
    math-verify, else 0: a truncated response or one without a boxed answer is wrong."""
    box = last_boxed(text)
    return float(stopped and box is not None and verify(parse(f"${answer}$"), parse(box)))


def lora_forward(self: LoraLinear, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """PEFT's LoRA Linear forward without its per-module fp32 round trip of the base output, with the adapter update
    computed in the activation dtype (bf16): the fp32 adapter weights are cast per call and stay the optimizer's
    master copy (mixed precision; this code uses no dropout, DoRA or merged adapters)."""
    result = self.base_layer(x, *args, **kwargs)
    if self.disable_adapters:
        return result
    for a in self.active_adapters:
        if a in self.lora_A:
            d = F.linear(F.linear(x, self.lora_A[a].weight.to(x.dtype)), self.lora_B[a].weight.to(x.dtype))
            result = result.add_(d if self.scaling[a] == 1.0 else d * self.scaling[a])
    return result


LoraLinear.forward = lora_forward


def render_prompts(
    cfg: DictConfig, tok: AutoTokenizer, prompts: list[str], member_prompt: str | None
) -> list[list[int]]:
    """Token ids of the prompts: the member prompt (if any) and the prompt are filled into cfg.user_template, then
    wrapped in the chat template with cfg.system_prompt, or used as raw text with the tokenizer's BOS for base models
    without one (use_chat_template false)."""
    users = [cfg.user_template.format("\n\n".join(x for x in (member_prompt, p) if x)) for p in prompts]
    if not cfg.use_chat_template:
        return [tok(u)["input_ids"] for u in users]
    system = [{"role": "system", "content": cfg.system_prompt}] if cfg.get("system_prompt") else []
    texts = [tok.apply_chat_template(system + [{"role": "user", "content": u}], add_generation_prompt=True,
                                     tokenize=False) for u in users]
    return [tok(t, add_special_tokens=False)["input_ids"] for t in texts]


def stop_ids(cfg: DictConfig, tok: AutoTokenizer) -> list[int]:
    """The policy's EOS ids plus cfg.extra_stop_tokens (e.g. the chat turn end of base models)."""
    eos = GenerationConfig.from_pretrained(cfg.policy_path).eos_token_id or tok.eos_token_id
    return (eos if isinstance(eos, list) else [eos]) + tok.convert_tokens_to_ids(list(cfg.get("extra_stop_tokens", [])))


def pad_batch(prompt_ids: list[list[int]], resp_ids: list[list[int]], pad_id: int, device: torch.device) -> Batch:
    """Right-padded prompt+response ids, prompt / response lengths and (B, R) response tokens. No attention mask is
    needed: with right padding and causal attention, real tokens never attend to the padding."""
    seqs = [p + r for p, r in zip(prompt_ids, resp_ids)]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, : len(s)] = torch.tensor(s)
    p_lens = torch.tensor([len(p) for p in prompt_ids])
    r_lens = torch.tensor([len(r) for r in resp_ids])
    R = int(r_lens.max())
    toks = torch.full((len(seqs), R), pad_id, dtype=torch.long)
    for j, r in enumerate(resp_ids):
        toks[j, : len(r)] = torch.tensor(r)
    return ids.to(device), p_lens.to(device), r_lens.to(device), toks.to(device)


class Trainer:
    def __init__(self, cfg: DictConfig):
        """Build the vLLM engine, the LoRA population (actor + critic adapters, value heads) and descriptors."""
        self.cfg = cfg
        self.N = int(cfg.N)
        self.rank, self.world = int(os.environ.get("REPPO_RANK", 0)), int(os.environ.get("REPPO_WORLD", 1))
        self.members = list(range(self.rank * self.N // self.world, (self.rank + 1) * self.N // self.world))
        if self.world > 1:
            self.store = torch.distributed.TCPStore("127.0.0.1", int(os.environ["REPPO_PORT"]), self.world,
                                                    self.rank == 0, timeout=timedelta(hours=3))
        self.desc = Descriptors(cfg.descriptors)
        self.K = len(self.desc.names)
        self.out = Path(cfg.out_dir)
        (self.out / "samples").mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, self.out / "config.yaml")
        self.metrics_f = open(self.out / "metrics.jsonl", "a") if self.rank == 0 else None
        self.shm = Path(cfg.shm_dir) / f"rank{self.rank}"
        self.shm.mkdir(parents=True, exist_ok=True)

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.tok = AutoTokenizer.from_pretrained(cfg.policy_path)
        self.pad_id = self.tok.convert_tokens_to_ids(cfg.pad_token)
        self.eos_ids = stop_ids(cfg, self.tok)
        # Shuffled train prompts and a fixed eval subset; eval.repeats > 1 samples each eval prompt that many times
        train = pd.read_parquet(cfg.data.train).sample(frac=1.0, random_state=cfg.seed)
        evald = pd.read_parquet(cfg.data.eval).sample(frac=1.0, random_state=0).head(cfg.eval.num_prompts)
        self.train_prompts, self.eval_prompts = train.prompt.tolist(), evald.prompt.tolist() * cfg.eval.get("repeats", 1)
        self.answers = dict(zip(train.prompt, train.answer)) | dict(zip(evald.prompt, evald.answer))  # reference answers
        print(f"[init] prompts: {len(self.train_prompts)} train, {len(self.eval_prompts)} eval", flush=True)
        self.cursor = 0
        self.sync_version = 0

        # vLLM first; the engine runs in a spawned process on the first visible GPU (shared with training). With
        # vllm_sleep it sleeps outside generation (weights offloaded to CPU, KV cache released) to free GPU memory.
        self.sleep = cfg.get("vllm_sleep", False)
        self.llm = LLM(
            model=cfg.policy_path,
            enable_lora=True,
            max_loras=len(self.members),
            max_lora_rank=cfg.lora.r,
            gpu_memory_utilization=cfg.vllm_gpu_util,
            max_model_len=cfg.max_prompt_tokens + cfg.max_new_tokens,
            dtype="bfloat16",
            seed=cfg.seed * 1000 + self.rank,  # independent sampling streams across processes
            generation_config="vllm",  # do not inherit the model's sampling defaults
            max_num_seqs=512,
            enable_sleep_mode=self.sleep,
        )
        self.sampling = SamplingParams(
            temperature=cfg.temperature,
            top_p=1.0,
            max_tokens=cfg.max_new_tokens,
            stop_token_ids=self.eos_ids,
            stop=list(cfg.get("stop_strings", [])),
            logprobs=0,
        )
        self.eval_sampling = SamplingParams(
            temperature=cfg.get("eval_temperature", cfg.temperature),
            top_p=cfg.get("eval_top_p", 1.0),
            max_tokens=cfg.max_new_tokens,
            stop_token_ids=self.eos_ids,
            stop=list(cfg.get("stop_strings", [])),
            logprobs=0,
        )

        self.dev = torch.device("cuda:0")
        base = AutoModelForCausalLM.from_pretrained(cfg.policy_path, dtype=torch.bfloat16, attn_implementation="sdpa")
        base.config.use_cache = False
        base.to(self.dev)
        if cfg.gradient_checkpointing:  # true, or the fraction of decoder layers checkpointed (the first ones)
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            for layer in base.model.layers[round(float(cfg.gradient_checkpointing) * len(base.model.layers)) :]:
                layer.gradient_checkpointing = False
        actor_cfg = LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=0.0, target_modules=LORA_TARGETS)
        critic_cfg = LoraConfig(
            r=cfg.critic_lora.r, lora_alpha=cfg.critic_lora.alpha, lora_dropout=0.0, target_modules=LORA_TARGETS
        )
        # each member's random init is seeded by its id, so it does not depend on the process layout
        torch.manual_seed(cfg.seed * 1000 + self.members[0])
        self.model = get_peft_model(base, actor_cfg, adapter_name=f"m{self.members[0]}")
        for i in self.members[1:]:
            torch.manual_seed(cfg.seed * 1000 + i)
            self.model.add_adapter(f"m{i}", actor_cfg)
        for i in self.members:
            torch.manual_seed(cfg.seed * 1000 + 100 + i)
            self.model.add_adapter(f"c{i}", critic_cfg)
        self.inner = self.model.base_model.model  # the *ForCausalLM with LoRA layers injected
        self.backbone = getattr(self.inner, self.inner.base_model_prefix)
        # frozen LM head, fp32 copy for log-probs; Granite divides its logits by config.logits_scaling
        scale = getattr(self.inner.config, "logits_scaling", 1.0)
        self.lm_w32 = self.inner.get_output_embeddings().weight.detach().float() / scale
        H = self.inner.config.hidden_size
        self.heads = torch.nn.ModuleList([torch.nn.Linear(H, 1 + self.K) for _ in self.members]).to(self.dev)
        for h, i in zip(self.heads, self.members):
            torch.manual_seed(cfg.seed * 1000 + 200 + i)
            torch.nn.init.normal_(h.weight, std=1.0 / (H + 1))
            torch.nn.init.zeros_(h.bias)
        torch.manual_seed(cfg.seed)

        # per local member (index k of self.members): actor / critic parameters and optimizers
        self.actor_params, self.critic_params, self.opt_a, self.opt_c = [], [], [], []
        for k, i in enumerate(self.members):
            a = [p for n, p in self.model.named_parameters() if f".m{i}." in n and "lora_" in n]
            c = [p for n, p in self.model.named_parameters() if f".c{i}." in n and "lora_" in n]
            c += list(self.heads[k].parameters())
            for p in a + c:
                p.requires_grad_(True)
            if cfg.lora_b_init_std > 0:  # symmetry breaking between otherwise identical members
                g = torch.Generator(device="cpu").manual_seed(cfg.seed * 1000 + i)
                for n, p in self.model.named_parameters():
                    if f"lora_B.m{i}." in n:
                        p.data.copy_(torch.randn(p.shape, generator=g) * cfg.lora_b_init_std)
            self.actor_params.append(a)
            self.critic_params.append(c)
            kw = dict(betas=tuple(cfg.optim.betas), eps=cfg.optim.eps, weight_decay=cfg.optim.weight_decay)
            self.opt_a.append(torch.optim.AdamW(a, lr=cfg.optim.actor_lr, **kw))
            self.opt_c.append(torch.optim.AdamW(c, lr=cfg.optim.critic_lr, **kw))
        n_a, n_c = sum(p.numel() for p in self.actor_params[0]), sum(p.numel() for p in self.critic_params[0])
        print(f"[init] N={self.N} K={self.K} members {self.members} actor params/member={n_a:,} "
              f"critic params/member={n_c:,}", flush=True)

        self.pop = population.PopulationState.initialize(self.N, self.K, eps_f=cfg.reppo.eps_f, device="cpu")
        self.score_mu, self.score_sigma = None, None

    def gather(self, key: str, obj: object) -> list:
        """`obj` (JSON-serializable) of every process, in rank order, through the TCP store."""
        if self.world == 1:
            return [obj]
        self.store.set(f"{key}/{self.rank}", json.dumps(obj))
        return [json.loads(self.store.get(f"{key}/{r}")) for r in range(self.world)]

    # ---------- data / generation ----------

    def render(self, prompts: list[str], i: int) -> list[list[int]]:
        """Token ids of the prompts for member i."""
        member = self.cfg.member_prompts[i] if self.cfg.get("member_prompts") else None
        return render_prompts(self.cfg, self.tok, prompts, member)

    def sync_adapters(self) -> list[LoRARequest]:
        """Publish the current actor adapters to vLLM under fresh names.

        A fresh name/id per sync (no load_inplace) because vLLM keys prefix-cache blocks by lora_name, so a reused
        name would serve KV computed with old weights (every prompt shares the system prompt block); load_inplace
        would also re-read the file on every scheduler step. The LRU cache (max_cpu_loras = local members) evicts the
        previous versions. The prefix cache is reset as well, belt and braces.
        """
        self.sync_version += 1
        reqs = []
        for i in self.members:
            self.model.save_pretrained(self.shm / f"v{self.sync_version}", selected_adapters=[f"m{i}"])
            # PEFT saves a non-"default" adapter into a subfolder named after it
            path = self.shm / f"v{self.sync_version}" / f"m{i}"
            reqs.append(LoRARequest(f"m{i}_v{self.sync_version}", self.sync_version * self.N + i + 1, str(path)))
        self.llm.reset_prefix_cache()
        old = self.shm / f"v{self.sync_version - 2}"  # keep the previous version until vLLM has moved on
        if old.exists():
            shutil.rmtree(old)
        return reqs

    def generate(
        self, prompt_ids: list[list[list[int]]], lora_reqs: list[LoRARequest], sampling: SamplingParams | None = None
    ) -> list[list[dict]]:
        """One sample per prompt per local member (prompt_ids[k]: local member k's rendered prompts): per-member
        lists of {ids, vlp (vLLM log-probs), text, eos}."""
        n = len(prompt_ids[0])
        inputs = [{"prompt_token_ids": p} for member in prompt_ids for p in member]
        loras = [req for req in lora_reqs for _ in range(n)]
        if self.sleep:
            torch.cuda.empty_cache()  # return the trainer's cached blocks before vLLM maps its memory again
            self.llm.wake_up()
        outs = self.llm.generate(inputs, sampling or self.sampling, lora_request=loras, use_tqdm=False)
        if self.sleep:
            self.llm.sleep(level=1)  # also clears the prefix cache
        res = [[] for _ in self.members]
        for j, o in enumerate(outs):
            c = o.outputs[0]
            ids = list(c.token_ids)
            vlp = [c.logprobs[t][ids[t]].logprob for t in range(len(ids))]
            stopped = c.finish_reason == "stop"  # an EOS token or one of cfg.stop_strings (dropped from the text)
            if stopped and not isinstance(c.stop_reason, str) and (not ids or ids[-1] not in self.eos_ids):
                raise RuntimeError(f"stop without an EOS token at the end: {ids[-5:]}")
            res[j // n].append({"ids": ids, "vlp": vlp, "text": c.text, "eos": stopped})
        return res

    def fitness(self, prompts: list[str], samples: list[list[dict]]) -> torch.Tensor:
        """Raw fitness (answer correctness, 0 or 1) of each local member's samples, (local members, n_prompts)."""
        raw = [math_correct(x["text"], self.answers[p], x["eos"]) for member in samples for p, x in zip(prompts, member)]
        return torch.tensor(raw, dtype=torch.float32).view(len(samples), len(prompts))

    def normalized_scores(self, prompts: list[str], samples: list[list[dict]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw fitness and fitness normalized by the initial policy's (mu, sigma); each (local members, n_prompts)."""
        raw = self.fitness(prompts, samples)
        return raw, (raw - self.score_mu) / self.score_sigma

    @torch.no_grad()
    def fit_normalization(self, lora_reqs: list[LoRARequest]) -> None:
        """Freeze the fitness (mu, sigma) and the descriptor means (on the correct answers) on one training-style batch
        of the initial policy (training prompts and sampling), pooled over all processes."""
        n, k = self.cfg.prompts_per_iter, self.cfg.samples_per_prompt
        prompts = [p for p in self.train_prompts[:n] for _ in range(k)]
        samples = self.generate([self.render(prompts, i) for i in self.members], lora_reqs)
        raw = self.fitness(prompts, samples).flatten().tolist()
        scores = torch.tensor(sum(self.gather("init_scores", raw), []))
        self.score_mu, self.score_sigma = scores.mean().item(), scores.std().item()
        correct = [x["text"] for member in samples for x in member]
        correct = [t for t, r in zip(correct, raw) if r > 0]
        correct = sum(self.gather("init_correct", correct), [])  # the same pooled texts, so the same fit, everywhere
        self.desc.fit(correct, self.cfg.get("descriptor_scales"))
        if self.rank == 0:  # the behaviour space this run uses
            json.dump({"descriptors": self.desc.names, "stats (mu, sigma)": self.desc.stats,
                       "n_fit_samples": len(correct), "score_mu": self.score_mu, "score_sigma": self.score_sigma},
                      open(self.out / "descriptor_stats.json", "w"), indent=1)
        print(f"[init] score mu={self.score_mu:.3f} sigma={self.score_sigma:.3f}; descriptor stats {self.desc.stats}",
              flush=True)

    def member_descriptors(
        self, prompts: list[str], samples: list[list[dict]], raw: torch.Tensor
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Raw descriptor values and (n_prompts, K) descriptors of each local member's samples; the descriptors are
        multiplied by the raw fitness (correctness), so incorrect answers sit at the center."""
        x = self.desc.raw([s["text"] for member in samples for s in member])
        d = self.desc.normalize(x) * raw.flatten().numpy()[:, None]
        n = len(prompts)
        split = range(len(samples))
        return [x[k * n : (k + 1) * n] for k in split], [d[k * n : (k + 1) * n] for k in split]

    # ---------- model passes ----------

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        """Last hidden states of the backbone under the active adapter (right-padded ids, no mask: see pad_batch)."""
        return self.backbone(input_ids=ids).last_hidden_state

    @torch.no_grad()
    def rollout_stats(
        self, k: int, batch: Batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Old log-probs, entropies, values and the response mask for local member k, in length-sorted token-budget
        microbatches."""
        i = self.members[k]
        ids, p_lens, r_lens, toks = batch
        pos, mask = response_positions(p_lens, r_lens, ids.shape[1])
        B, R_all = mask.shape
        old = torch.zeros(B, R_all, device=self.dev)
        ent = torch.zeros_like(old)
        val = torch.zeros(B, R_all, 1 + self.K, device=self.dev)
        self.model.eval()
        for j in token_budget_chunks(torch.arange(B, device=self.dev), p_lens + r_lens, self.cfg.forward_tokens):
            L = int((p_lens[j] + r_lens[j]).max())
            R = int(r_lens[j].max())
            args = (toks[j, :R], mask[j, :R], self.cfg.temperature)
            pj = pos[j, :R].clamp(max=L - 1)
            self.model.set_adapter(f"m{i}")
            h = gather_hidden(self.hidden(ids[j, :L]), pj)
            old[j, :R], ent[j, :R] = token_logprobs(self.lm_w32, h, *args, with_entropy=True)
            self.model.set_adapter(f"c{i}")
            h = gather_hidden(self.hidden(ids[j, :L]), pj)
            val[j, :R] = self.heads[k](h.float())
        self.model.train()
        return old, ent, val * mask[..., None], mask

    def update_member(
        self,
        k: int,
        batch: Batch,
        old_lp: torch.Tensor,
        adv: torch.Tensor,
        old_v: torch.Tensor,
        ret: torch.Tensor,
        mask: torch.Tensor,
        train_actor: bool,
    ) -> dict[str, float]:
        """PPO epochs over minibatches for local member k (critic only during the critic warmup); mean stats."""
        cfg, i = self.cfg, self.members[k]
        ids, p_lens, r_lens, toks = batch
        pos, _ = response_positions(p_lens, r_lens, ids.shape[1])
        B = ids.shape[0]
        stats = []
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(B, device=self.dev)
            for m0 in range(0, B, cfg.minibatch):
                mb_idx = perm[m0 : m0 + cfg.minibatch]
                n_tok = mask[mb_idx].sum()  # losses are minibatch token means, however the microbatches split
                self.opt_a[k].zero_grad(set_to_none=True)
                self.opt_c[k].zero_grad(set_to_none=True)
                for j in token_budget_chunks(mb_idx, p_lens + r_lens, cfg.train_tokens):
                    L = int((p_lens[j] + r_lens[j]).max())
                    R = int(r_lens[j].max())
                    mk = mask[j, :R]
                    pj = pos[j, :R].clamp(max=L - 1)
                    # Actor then critic, each forward immediately followed by its backward: with gradient
                    # checkpointing, the forward is recomputed under whichever adapter is active at backward time.
                    st = {}
                    if train_actor:
                        self.model.set_adapter(f"m{i}")
                        h = gather_hidden(self.hidden(ids[j, :L]), pj)
                        new_lp = token_logprobs(self.lm_w32, h, toks[j, :R], mk, cfg.temperature)
                        pg_loss, st = policy_loss(new_lp, old_lp[j, :R], adv[j, :R], mk, cfg.clip, n_tok)
                        pg_loss.backward()
                        del h, new_lp, pg_loss
                    self.model.set_adapter(f"c{i}")
                    h = gather_hidden(self.hidden(ids[j, :L]), pj)
                    vpred = self.heads[k](h.float())
                    v_loss, st_v = value_loss(vpred, old_v[j, :R], ret[j, :R], mk, cfg.vclip, n_tok)
                    v_loss.backward()
                    del h, vpred, v_loss
                    st.update(st_v)
                    stats.append(st)
                gn_a = torch.nn.utils.clip_grad_norm_(self.actor_params[k], cfg.optim.grad_clip) if train_actor else 0.0
                gn_c = torch.nn.utils.clip_grad_norm_(self.critic_params[k], cfg.optim.grad_clip)
                if train_actor:
                    self.opt_a[k].step()
                self.opt_c[k].step()
                stats[-1].update(grad_norm_actor=float(gn_a), grad_norm_critic=float(gn_c))
        keys = ("approx_kl", "clipfrac", "grad_norm_actor", "grad_norm_critic")
        return {key: float(np.mean([s[key] for s in stats if key in s])) for key in keys if key in stats[-1]}

    # ---------- evaluation ----------

    @torch.no_grad()
    def evaluate(self, it: int, lora_reqs: list[LoRARequest]) -> dict[str, float]:
        """Score the eval prompts (at eval_temperature) and write samples/eval_iterXXXX.jsonl."""
        prompt_ids = [self.render(self.eval_prompts, i) for i in self.members]
        samples = self.generate(prompt_ids, lora_reqs, self.eval_sampling)
        raw, s = self.normalized_scores(self.eval_prompts, samples)
        m, rows = {}, []
        xs, descs = self.member_descriptors(self.eval_prompts, samples, raw)
        for k, i in enumerate(self.members):
            for j, x in enumerate(samples[k]):
                rows.append({"member": i, "prompt": self.eval_prompts[j], "response": x["text"],
                             "score": s[k, j].item(), "raw": raw[k, j].item(), "eos": x["eos"], "len": len(x["ids"]),
                             "desc": descs[k][j].round(4).tolist(), "desc_raw": xs[k][j].round(4).tolist()})
            m[f"eval/score_m{i}"] = s[k].mean().item()
            m[f"eval/raw_score_m{i}"] = raw[k].mean().item()
            m[f"eval/len_m{i}"] = float(np.mean([len(x["ids"]) for x in samples[k]]))
            m[f"eval/eos_m{i}"] = float(np.mean([x["eos"] for x in samples[k]]))
            ok = raw[k].numpy() > 0
            for c, n in enumerate(self.desc.names):
                m[f"eval/{n}_m{i}"] = float(descs[k][:, c].mean())
                m[f"eval/raw_{n}_m{i}"] = float(xs[k][ok, c].mean()) if ok.any() else float("nan")
        rows = sum(self.gather(f"eval_rows_{it}", rows), [])
        if self.rank == 0:
            with open(self.out / "samples" / f"eval_iter{it:04d}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
        m["eval/score_mean"] = float(np.mean([r["score"] for r in rows]))
        m["eval/raw_score_mean"] = float(np.mean([r["raw"] for r in rows]))
        return m

    # ---------- main loop ----------

    def train(self) -> None:
        """Run cfg.iterations RePPO iterations, with evaluation and checkpoints."""
        cfg = self.cfg
        warmup = max(1, int(cfg.optim.warmup_frac * cfg.iterations))
        start = 1
        if cfg.get("resume") and (self.out / "checkpoints" / "resume").exists():
            start = self.resume() + 1
        else:
            lora_reqs = self.sync_adapters()
            self.fit_normalization(lora_reqs)
            self.log(0, self.evaluate(0, lora_reqs))
        for it in range(start, cfg.iterations + 1):
            t0 = time.time()
            train_actor = it > cfg.critic_warmup_iters
            lr_scale = min(1.0, (it - cfg.critic_warmup_iters) / warmup) if train_actor else 0.0
            for opt in self.opt_a:
                for g in opt.param_groups:
                    g["lr"] = cfg.optim.actor_lr * lr_scale

            n = cfg.prompts_per_iter
            if self.cursor + n > len(self.train_prompts):
                random.shuffle(self.train_prompts)
                self.cursor = 0
            chunk = self.train_prompts[self.cursor : self.cursor + n]
            prompts = [p for p in chunk for _ in range(cfg.samples_per_prompt)]
            self.cursor += n
            prompt_ids = [self.render(prompts, i) for i in self.members]
            lora_reqs = self.sync_adapters()
            samples = self.generate(prompt_ids, lora_reqs)
            t_gen = time.time() - t0
            raw, s = self.normalized_scores(prompts, samples)
            t_fit = time.time() - t0 - t_gen

            # Descriptors: terminal descriptor rewards and their per-member means
            xs, descs = self.member_descriptors(prompts, samples, raw)
            d_rew = torch.tensor(np.stack(descs), dtype=torch.float32)
            b_ep = d_rew.mean(1)
            t_desc = time.time() - t0 - t_gen - t_fit
            # Fitness EMA and diversity coefficients over the whole population (inactive for N=1 or K=0)
            f_all, b_all = zip(*self.gather(f"pop_{it}", [s.mean(1).tolist(), b_ep.tolist()]))
            self.pop.update(f_ep=torch.tensor(sum(f_all, [])), b_ep=torch.tensor(sum(b_all, [])).view(self.N, self.K),
                            beta_f=cfg.reppo.beta_f, beta_b=cfg.reppo.beta_b)
            coeffs = population.compute_diversity_coefficients(
                self.pop, K_nn=cfg.reppo.K_nn, gamma_sq=cfg.reppo.gamma_sq, eps_f=cfg.reppo.eps_f,
                eps_b=cfg.reppo.eps_b, C_max=cfg.reppo.C_max, repulsion_mode=cfg.reppo.repulsion_mode,
            )
            if cfg.reppo.force_c_zero:  # independent PPO members (the c = 0 control)
                coeffs.c_coeff.zero_()
                coeffs.w_k.fill_(1.0)

            metrics = {"time/gen": t_gen, "time/fitness": t_fit, "time/desc": t_desc}
            for k, i in enumerate(self.members):
                batch = pad_batch(prompt_ids[k], [x["ids"] for x in samples[k]], self.pad_id, self.dev)
                old_lp, ent, values, mask = self.rollout_stats(k, batch)
                r_lens = batch[2]
                rewards = torch.zeros(values.shape, device=self.dev)  # terminal rewards: fitness, then descriptors
                last = r_lens - 1
                rewards[torch.arange(len(last)), last, 0] += s[k].to(self.dev)
                rewards[torch.arange(len(last)), last, 1:] += d_rew[k].to(self.dev)
                adv, ret = gae(rewards, values, mask, cfg.gamma, cfg.lam)
                adv_n = torch.stack([masked_whiten(adv[..., c], mask) for c in range(1 + self.K)], -1)
                comb = coeffs.w_k[i] * adv_n[..., 0]
                if self.K > 0:
                    comb = comb + (adv_n[..., 1:] * coeffs.c_coeff[i].to(self.dev)).sum(-1)
                comb = comb * mask
                st = self.update_member(k, batch, old_lp, comb, values, ret, mask, train_actor)

                vlp = torch.zeros_like(old_lp)  # sampler (vLLM) log-probs; should match the trainer's old_lp
                for j, x in enumerate(samples[k]):
                    vlp[j, : len(x["vlp"])] = torch.tensor(x["vlp"])
                v, r = values[mask], ret[mask]  # (tokens, 1 + K), the critic before this iteration's update
                ev = (1 - (r - v).var(0) / (r.var(0) + 1e-8)).tolist()
                metrics.update({
                    f"score_m{i}": s[k].mean().item(),
                    f"raw_score_m{i}": raw[k].mean().item(),
                    f"len_m{i}": r_lens.float().mean().item(),
                    f"eos_m{i}": float(np.mean([x["eos"] for x in samples[k]])),
                    f"entropy_m{i}": masked_mean(ent, mask).item(),
                    f"vllm_abs_lp_diff_m{i}": masked_mean((vlp - old_lp).abs(), mask).item(),
                    f"value_explained_var_m{i}": ev[0],
                    **{f"value_explained_var_{n}_m{i}": e for n, e in zip(self.desc.names, ev[1:])},
                    **{f"{name}_m{i}": v for name, v in st.items()},
                })
            for k, i in enumerate(self.members):  # raw descriptor means over the correct answers, in their own units
                ok = raw[k].numpy() > 0
                for c, n in enumerate(self.desc.names):
                    metrics[f"raw_{n}_m{i}"] = float(xs[k][ok, c].mean()) if ok.any() else float("nan")
            if it % cfg.eval.every == 0 or it == cfg.iterations:
                self.save_samples(it, prompts, samples, raw, xs, descs)
            metrics["time/iter"] = time.time() - t0
            metrics["f_hat"] = self.pop.f_hat.tolist()
            metrics["b_hat"] = self.pop.b_hat.tolist()
            metrics["b_ep"] = sum(b_all, [])  # this iteration's descriptor means, before the EMA
            metrics["c_coeff"] = coeffs.c_coeff.tolist()
            if it % cfg.eval.every == 0 or it == cfg.iterations:
                metrics.update(self.evaluate(it, lora_reqs=self.sync_adapters()))
            if it % cfg.save_every == 0 or it == cfg.iterations:
                self.save(it)
            self.log(it, metrics)

    def save_samples(
        self, it: int, prompts: list[str], samples: list[list[dict]], raw: torch.Tensor, xs: list[np.ndarray],
        descs: list[np.ndarray],
    ) -> None:
        """This process's training batch (responses, correctness, raw and gated descriptors) as
        samples/train_iterXXXX_rankR.jsonl.gz."""
        with gzip.open(self.out / "samples" / f"train_iter{it:04d}_rank{self.rank}.jsonl.gz", "wt") as f:
            for k, i in enumerate(self.members):
                for j, x in enumerate(samples[k]):
                    row = {"member": i, "prompt": prompts[j], "response": x["text"], "raw": raw[k, j].item(),
                           "eos": x["eos"], "len": len(x["ids"]), "desc_raw": xs[k][j].round(4).tolist(),
                           "desc": descs[k][j].round(4).tolist()}
                    f.write(json.dumps(row) + "\n")

    def save(self, it: int) -> None:
        """Actor adapters (checkpoints/iterXXXX/m{i}, kept) and the full training state (checkpoints/resume/iterXXXX,
        only the latest kept): actor and critic adapters, value heads, optimizers, population and normalization
        statistics, data order and RNG states (vLLM's sampling stream is not restored)."""
        d = self.out / "checkpoints" / f"iter{it:04d}"
        self.model.save_pretrained(d, selected_adapters=[f"m{i}" for i in self.members])
        r = self.out / "checkpoints" / "resume" / f"iter{it:04d}" / f"rank{self.rank}"
        self.model.save_pretrained(r, selected_adapters=[f"{a}{i}" for i in self.members for a in "mc"])
        torch.save({
            "it": it, "heads": [h.state_dict() for h in self.heads], "opt_a": [o.state_dict() for o in self.opt_a],
            "opt_c": [o.state_dict() for o in self.opt_c], "f_hat": self.pop.f_hat, "b_hat": self.pop.b_hat,
            "score_mu": self.score_mu, "score_sigma": self.score_sigma, "desc_stats": self.desc.stats,
            "cursor": self.cursor, "train_prompts": self.train_prompts,
            "rng": (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()),
        }, r / "state.pt")
        self.gather(f"saved_{it}", it)  # every process has written its state
        if self.rank == 0:
            for old in (self.out / "checkpoints" / "resume").iterdir():
                if old.name != f"iter{it:04d}":
                    shutil.rmtree(old)

    def resume(self) -> int:
        """Load the latest full training state written by save (all processes); returns its iteration."""
        d = sorted((self.out / "checkpoints" / "resume").iterdir())[-1] / f"rank{self.rank}"
        for i in self.members:
            for a in "mc":
                set_peft_model_state_dict(self.model, load_file(d / f"{a}{i}" / "adapter_model.safetensors"),
                                          adapter_name=f"{a}{i}")
        s = torch.load(d / "state.pt", weights_only=False)
        for k in range(len(self.members)):
            self.heads[k].load_state_dict(s["heads"][k])
            self.opt_a[k].load_state_dict(s["opt_a"][k])
            self.opt_c[k].load_state_dict(s["opt_c"][k])
        self.pop.f_hat, self.pop.b_hat = s["f_hat"], s["b_hat"]
        self.score_mu, self.score_sigma, self.desc.stats = s["score_mu"], s["score_sigma"], s["desc_stats"]
        self.cursor, self.train_prompts = s["cursor"], s["train_prompts"]
        random.setstate(s["rng"][0])
        np.random.set_state(s["rng"][1])
        torch.set_rng_state(s["rng"][2])
        torch.cuda.set_rng_state(s["rng"][3])
        print(f"[resume] iteration {s['it']} from {d}", flush=True)
        return s["it"]

    def log(self, it: int, m: dict) -> None:
        """Merge all processes' metrics (shared keys such as timings: the first process's values), append them to
        metrics.jsonl and print a one-line summary of member 0 (first process only)."""
        m = {key: v for part in reversed(self.gather(f"metrics_{it}", m)) for key, v in part.items()}
        if self.rank > 0:
            return
        m = {"iter": it, **m}
        self.metrics_f.write(json.dumps(m) + "\n")
        self.metrics_f.flush()
        keys = {"score_m0", "len_m0", "eos_m0", "approx_kl_m0", "clipfrac_m0", "value_explained_var_m0",
                "vllm_abs_lp_diff_m0", "eval/score_mean", "eval/len_m0", "time/iter"}
        print(f"[it {it}] " + " ".join(f"{k}={v:.4g}" for k, v in m.items() if k in keys), flush=True)


def main() -> None:
    """Train with config files (merged in order) plus key=value overrides."""
    files = [a for a in sys.argv[1:] if a.endswith(".yaml")]
    overrides = [a for a in sys.argv[1:] if not a.endswith(".yaml")]
    cfg = OmegaConf.merge(*[OmegaConf.load(f) for f in files], OmegaConf.from_cli(overrides))
    torch.backends.cuda.matmul.allow_tf32 = True  # fp32 LM head / value heads: log-probs within ~1e-5 of fp32
    # the trainer shares its GPU with vLLM: avoid allocator fragmentation (vLLM is compatible without KV connectors)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # SIGTERM -> SystemExit, so the vLLM engine process is shut down instead of orphaned
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    trainer = Trainer(cfg)
    try:
        trainer.train()
    finally:
        trainer.llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
