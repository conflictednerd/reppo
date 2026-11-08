"""Math evaluation of a population on MATH-500, Minerva and OlympiadBench: n samples per problem per member
(cfg.eval_temperature / eval_top_p), graded by math_correct (the last \\boxed{...} under math-verify; a truncated
response is wrong). Writes one row per sample (bench, member, problem, response, len, eos, correct) for
scripts/analyze.py; prints pass@1.

Usage: python scripts/eval_math.py <out.jsonl> conf/math.yaml conf/math_<model>.yaml [conf/overlay.yaml ...]
[checkpoint=<dir>] [n=8]   (source scripts/env.sh). Without checkpoint the base policy is evaluated (once per member
prompt, if any).
"""

import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reppo_llm.train import math_correct, render_prompts, stop_ids  # noqa: E402


def benchmarks() -> dict[str, tuple[list[str], list[str]]]:
    """name -> (problems, answers)."""
    out = {}
    for name, path, split, q, a in [
        ("math500", "HuggingFaceH4/MATH-500", "test", "problem", "answer"),
        ("minerva", "math-ai/minervamath", "test", "question", "answer"),
        ("olympiad", "knoveleng/OlympiadBench", "train", "question", "final_answer"),
    ]:
        d = load_dataset(path)[split]
        out[name] = (d[q], [str(x[0] if isinstance(x, list) else x) for x in d[a]])
    return out


def main() -> None:
    """Generate, grade and print per-member pass@1 for every benchmark."""
    out = sys.argv[1]
    files = [a for a in sys.argv[2:] if a.endswith(".yaml")]
    overrides = [a for a in sys.argv[2:] if "=" in a]
    cfg = OmegaConf.merge(*[OmegaConf.load(f) for f in files], OmegaConf.from_cli(overrides))
    ckpt = cfg.get("checkpoint")
    members = list(range(cfg.N)) if ckpt or cfg.get("member_prompts") else [0]
    tok = AutoTokenizer.from_pretrained(cfg.policy_path)
    stop = stop_ids(cfg, tok)
    llm = LLM(model=cfg.policy_path, enable_lora=bool(ckpt), max_loras=len(members), max_lora_rank=cfg.lora.r,
              gpu_memory_utilization=0.85, max_model_len=cfg.max_prompt_tokens + cfg.max_new_tokens, dtype="bfloat16",
              seed=0, generation_config="vllm")
    k = int(cfg.get("n", 8))
    sampling = SamplingParams(n=k, temperature=cfg.eval_temperature, top_p=cfg.eval_top_p,
                              max_tokens=cfg.max_new_tokens, stop_token_ids=stop, stop=list(cfg.get("stop_strings", [])))
    rows = []
    for name, (problems, answers) in benchmarks().items():
        for i in members:
            member = cfg.member_prompts[i] if cfg.get("member_prompts") else None
            texts = [{"prompt_token_ids": ids} for ids in render_prompts(cfg, tok, problems, member)]
            lora = LoRARequest(f"m{i}", i + 1, str(Path(ckpt) / f"m{i}")) if ckpt else None
            for p, a, o in zip(problems, answers, llm.generate(texts, sampling, lora_request=lora, use_tqdm=False)):
                for c in o.outputs:
                    rows.append({"bench": name, "member": i, "problem": p, "response": c.text, "len": len(c.token_ids),
                                 "eos": c.finish_reason == "stop",
                                 "correct": math_correct(c.text, a, c.finish_reason == "stop")})
            acc = np.mean([r["correct"] for r in rows if r["bench"] == name and r["member"] == i])
            print(f"{name:8s} member {i}: pass@1 {acc:.3f} (k={k})", flush=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
