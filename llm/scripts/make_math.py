"""Math reasoning prompts: MATH train levels 3-5 for training (5,584 problems) and a validation set for in-loop
evaluation and checkpoint selection: VAL_N MATH test level 3-5 problems outside MATH-500 (a test set) and the training
set. The test sets (MATH-500, Minerva, OlympiadBench) are loaded by scripts/eval_math.py.

Writes data/math/prompts_{train,val}.parquet with columns prompt (the problem; configs add the instruction through
user_template), answer (the reference final answer) and source.
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT = Path(__file__).resolve().parents[1] / "data" / "math"
VAL_N = 384


def last_boxed(solution: str) -> str:
    """Content of the last \\boxed{...} in a MATH solution (braces matched)."""
    start = solution.rindex("\\boxed{") + len("\\boxed{")
    depth, i = 1, start
    while depth:
        depth += {"{": 1, "}": -1}.get(solution[i], 0)
        i += 1
    return solution[start : i - 1]


def levels_3_to_5(d: pd.DataFrame, source: str) -> pd.DataFrame:
    """The level 3-5 problems with a boxed reference answer, as (prompt, answer, source)."""
    d = d[d.level.isin(["Level 3", "Level 4", "Level 5"]) & d.solution.str.contains("\\\\boxed\\{")]
    return pd.DataFrame({"prompt": d.problem, "answer": d.solution.map(last_boxed), "source": source})


def main() -> None:
    """Build and save the train and validation prompt tables."""
    math = load_dataset("DigitalLearningGmbH/MATH-lighteval")
    train = levels_3_to_5(math["train"].to_pandas(), "MATH")
    test = levels_3_to_5(math["test"].to_pandas(), "MATH-val")
    math500 = set(load_dataset("HuggingFaceH4/MATH-500")["test"]["problem"])
    val = test[~test.prompt.isin(math500) & ~test.prompt.isin(train.prompt)].sample(n=VAL_N, random_state=0)
    OUT.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUT / "prompts_train.parquet")
    val.to_parquet(OUT / "prompts_val.parquet", index=False)
    print(f"{len(train)} train, {len(val)} validation prompts -> {OUT}")


if __name__ == "__main__":
    main()
