"""Test-set analysis of scripts/eval_math.py outputs: per-member and mean pass@1 (overall and per benchmark), pass@k,
behaviour descriptors (correctness x tanh((x - mu) / sigma), with a run's descriptor_stats.json), mean pairwise
correlation of the members' per-problem accuracies, member identifiability, and paired bootstrap 95% CIs over problems
of the pass@k differences: every file against the first one, then every other pair.

pass@k is the unbiased estimator. One member with n samples per problem: pass@k = 1 - C(n - c, k) / C(n, k). M members
with n samples each: pass@(M j) = 1 - prod_m C(n - c_m, j) / C(n, j); pass@1 is the members' mean accuracy.
Identifiability: accuracy of a TF-IDF + logistic regression classifier that predicts the member from the answer part of
a response, trained on a random half of the problems and tested on the other half.

Usage: python scripts/analyze.py --stats <run>/descriptor_stats.json base=<base.jsonl> <name>=<eval.jsonl> ...
"""

import argparse
import collections
import itertools
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reppo_llm.descriptors import Descriptors, answer_part  # noqa: E402


def pass_at_k(c: np.ndarray, n: int) -> dict[int, np.ndarray]:
    """Per-problem pass@k from the correct counts c (problems, members) of n samples per member."""
    M = c.shape[1]
    if M == 1:
        return {k: np.array([1 - comb(n - x, k) / comb(n, k) for x in c[:, 0]]) for k in (1, 5, 10, 20, 40) if k <= n}
    pk = {M * j: np.array([1 - np.prod([comb(n - x, j) / comb(n, j) for x in row]) for row in c]) for j in (1, 2, 4, 8)
          if j <= n}
    return {1: c.mean(1) / n, **pk}


def identifiability(rows: list[dict], keys: list[tuple[str, str]]) -> float:
    """Member prediction accuracy on the held-out half of the problems."""
    train = set(np.random.default_rng(0).permutation(len(keys))[: len(keys) // 2])
    index = {q: i for i, q in enumerate(keys)}
    split = np.array([index[(r["bench"], r["problem"])] in train for r in rows])
    X = np.array([answer_part(r["response"]) for r in rows], dtype=object)
    y = np.array([r["member"] for r in rows])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(vec.fit_transform(X[split]), y[split])
    return float((clf.predict(vec.transform(X[~split])) == y[~split]).mean())


def main() -> None:
    """Print one table per file, then the bootstrap comparisons."""
    p = argparse.ArgumentParser()
    p.add_argument("--stats", required=True, help="descriptor_stats.json of the run whose behaviour space is used")
    p.add_argument("runs", nargs="+", help="name=path of eval_math.py outputs; the first is the bootstrap reference")
    args = p.parse_args()
    st = json.load(open(args.stats))
    desc = Descriptors(st["descriptors"])
    desc.stats = st["stats (mu, sigma)"]
    runs = dict(a.split("=", 1) for a in args.runs)
    keys, PK = None, {}
    for name, path in runs.items():
        rows = [json.loads(line) for line in open(path)]
        keys = keys or sorted({(r["bench"], r["problem"]) for r in rows})
        benches = sorted({b for b, _ in keys})
        members = sorted({r["member"] for r in rows})
        acc = collections.defaultdict(list)  # (member, problem) -> correctness of each sample
        for r in rows:
            acc[r["member"], (r["bench"], r["problem"])].append(float(r["correct"]))
        A = np.array([[np.mean(acc[m, q]) for q in keys] for m in members])  # (members, problems) accuracy
        n = len(acc[members[0], keys[0]])
        PK[name] = pass_at_k(np.array([[sum(acc[m, q]) for m in members] for q in keys]).astype(int), n)
        correct = np.array([float(r["correct"]) for r in rows])
        d = desc.normalize(desc.raw([r["response"] for r in rows])) * correct[:, None]
        member = np.array([r["member"] for r in rows])
        bd = np.array([d[member == m].mean(0) for m in members])

        print(f"\n=== {name}: {len(members)} members x {n} samples per problem")
        print(f"{'':6s} {'pass@1':>7s} " + " ".join(f"{b:>8s}" for b in benches) + " | "
              + " ".join(f"{x:>12s}" for x in desc.names))
        sel = {b: np.array([q[0] == b for q in keys]) for b in benches}
        for k, m in enumerate(members):
            print(f"m{m:<5d} {A[k].mean():7.3f} " + " ".join(f"{A[k, sel[b]].mean():8.3f}" for b in benches) + " | "
                  + " ".join(f"{x:12.3f}" for x in bd[k]))
        print(f"{'mean':6s} {A.mean():7.3f} " + " ".join(f"{A[:, sel[b]].mean():8.3f}" for b in benches) + " | "
              + " ".join(f"{x:12.3f}" for x in bd.mean(0)))
        print(f"{'spread':6s} {'':7s} " + " ".join(f"{'':8s}" for _ in benches) + " | "
              + " ".join(f"{x:12.3f}" for x in bd.max(0) - bd.min(0)))
        print("pass@k " + " ".join(f"@{k} {v.mean():.3f}" for k, v in sorted(PK[name].items())))
        if len(members) > 1:
            corr = np.mean([np.corrcoef(A[i], A[j])[0, 1] for i, j in itertools.combinations(range(len(members)), 2)])
            print(f"pairwise accuracy correlation {corr:.3f}  identifiability {identifiability(rows, keys):.3f}",
                  flush=True)

    rng = np.random.default_rng(0)

    def ci(x: np.ndarray) -> str:
        """Mean difference in points and its paired bootstrap 95% CI over problems."""
        bs = x[rng.integers(0, len(x), (5000, len(x)))].mean(1)
        return f"{100 * x.mean():+.1f} [{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]"

    ref, *others = runs
    print("\npaired bootstrap, points (95% CI)")
    for k in sorted(PK[ref]):
        print(f"pass@{k:<2d} " + " | ".join(f"{a} - {ref} {ci(PK[a][k] - PK[ref][k])}" for a in others if k in PK[a]))
    for a, b in itertools.combinations(others, 2):
        print(f"{a} - {b}: " + " | ".join(f"@{k} {ci(PK[a][k] - PK[b][k])}" for k in sorted(PK[a]) if k in PK[b]))


if __name__ == "__main__":
    main()
