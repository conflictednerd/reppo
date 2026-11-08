"""Paper metrics from evaluation_results.json files, one markdown table per task.

    uv run python scripts/compute_metrics.py outputs/reppo outputs/ind_ppo outputs/ppga

Each directory is one method and holds <env>_s<seed>/evaluation_results.json. For every env, f* is the fitness of
the best policy of any method in any seed; policies with fitness >= 0.1 f* (0.25 f* on Ant) are retained. Best
fitness is taken before the filter; mean fitness, the Vendi Score (RBF kernel on the mean descriptors, sigma^2 = 0.5)
and qVS = mean fitness x Vendi Score on the retained policies; a run with no retained policy scores 0. Mean +- s.e.
over seeds.
"""

import json
import sys
from pathlib import Path

import numpy as np

SIGMA_SQ = 0.5
FRAC = {"ant": 0.25}


def vendi_score(desc):
    n = len(desc)
    if n == 0:
        return 0.0
    d2 = ((desc[:, None] - desc[None]) ** 2).sum(-1)
    w = np.linalg.eigvalsh(np.exp(-d2 / (2 * SIGMA_SQ)) / n)
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))


def load(path):
    d = json.loads(path.read_text())
    return (np.array([p["return_mean"] for p in d["policies"]]),
            np.array([p["behavior_descriptor_mean"] for p in d["policies"]]))


def mean_se(x):
    x = np.array(x, dtype=float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def main():
    runs = {}  # env -> method -> [(fitness, descriptors) per seed]
    for method_dir in map(Path, sys.argv[1:]):
        for f in sorted(method_dir.glob("*_s*/evaluation_results.json")):
            env = f.parent.name.rsplit("_s", 1)[0]
            runs.setdefault(env, {}).setdefault(method_dir.name, []).append(load(f))
    for env, methods in runs.items():
        bar = FRAC.get(env, 0.1) * max(fit.max() for rs in methods.values() for fit, _ in rs)
        print(f"\n## {env}\n\n| method | seeds | best fitness | mean fitness | Vendi Score | qVS | retained |")
        print("|---|---|---|---|---|---|---|")
        for method, rs in methods.items():
            rows = []
            for fit, desc in rs:
                keep = fit >= bar
                fm = fit[keep].mean() if keep.any() else 0.0
                vs = vendi_score(desc[keep])
                rows.append((fit.max(), fm, vs, fm * vs, keep.sum()))
            cells = [f"{m:.{p}f} ± {s:.{p}f}" for (m, s), p in zip(map(mean_se, zip(*rows)), (0, 0, 2, 0, 0))]
            print(f"| {method} | {len(rs)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
