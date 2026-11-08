#!/bin/bash
# RePPO and Ind-PPO (N = 5 LoRA members) on MATH for Qwen3-1.7B-Base and Granite-4.1-3B-Base, the prompting baselines
# and the base models, their test evaluation and the analysis. Run from llm/ after `uv sync`; training uses one
# process per GPU (scripts/launch.sh), evaluation one GPU.
source scripts/env.sh
mkdir -p ../outputs/llm/eval

python scripts/make_math.py

bash scripts/launch.sh 0,1,2,3,4 conf/math.yaml conf/math_qwen3.yaml
bash scripts/launch.sh 0,1,2,3,4 conf/math.yaml conf/math_qwen3.yaml reppo.force_c_zero=true out_dir=../outputs/llm/math_qwen3_ind_ppo shm_dir=/dev/shm/reppo_llm/math_qwen3_ind_ppo
bash scripts/launch.sh 0,1,2,3,4 conf/math.yaml conf/math_granite.yaml
bash scripts/launch.sh 0,1,2,3,4 conf/math.yaml conf/math_granite.yaml reppo.force_c_zero=true out_dir=../outputs/llm/math_granite_ind_ppo shm_dir=/dev/shm/reppo_llm/math_granite_ind_ppo

# Test sets (MATH-500, Minerva, OlympiadBench), 8 samples per problem per member, 40 for the base model. The trained
# checkpoint is the evaluated iteration with the highest eval/raw_score_mean in the run's metrics.jsonl (paper: 70 for
# Qwen3 RePPO, 90 for the other three runs).
python scripts/eval_math.py ../outputs/llm/eval/qwen3_base.jsonl conf/math.yaml conf/math_qwen3.yaml n=40
python scripts/eval_math.py ../outputs/llm/eval/qwen3_corners.jsonl conf/math.yaml conf/math_qwen3.yaml conf/prompting_math.yaml
python scripts/eval_math.py ../outputs/llm/eval/qwen3_emphasis.jsonl conf/math.yaml conf/math_qwen3.yaml conf/prompting_math_focus.yaml
python scripts/eval_math.py ../outputs/llm/eval/qwen3_reppo.jsonl conf/math.yaml conf/math_qwen3.yaml checkpoint=../outputs/llm/math_qwen3/checkpoints/iter0070
python scripts/eval_math.py ../outputs/llm/eval/qwen3_ind_ppo.jsonl conf/math.yaml conf/math_qwen3.yaml checkpoint=../outputs/llm/math_qwen3_ind_ppo/checkpoints/iter0090
python scripts/eval_math.py ../outputs/llm/eval/granite_base.jsonl conf/math.yaml conf/math_granite.yaml n=40
python scripts/eval_math.py ../outputs/llm/eval/granite_corners.jsonl conf/math.yaml conf/math_granite.yaml conf/prompting_math.yaml
python scripts/eval_math.py ../outputs/llm/eval/granite_emphasis.jsonl conf/math.yaml conf/math_granite.yaml conf/prompting_math_focus.yaml
python scripts/eval_math.py ../outputs/llm/eval/granite_reppo.jsonl conf/math.yaml conf/math_granite.yaml checkpoint=../outputs/llm/math_granite/checkpoints/iter0090
python scripts/eval_math.py ../outputs/llm/eval/granite_ind_ppo.jsonl conf/math.yaml conf/math_granite.yaml checkpoint=../outputs/llm/math_granite_ind_ppo/checkpoints/iter0090

# pass@1 and pass@k, descriptors in the RePPO run's behaviour space, identifiability, paired bootstrap CIs
python scripts/analyze.py --stats ../outputs/llm/math_qwen3/descriptor_stats.json base=../outputs/llm/eval/qwen3_base.jsonl corners=../outputs/llm/eval/qwen3_corners.jsonl emphasis=../outputs/llm/eval/qwen3_emphasis.jsonl ind_ppo=../outputs/llm/eval/qwen3_ind_ppo.jsonl reppo=../outputs/llm/eval/qwen3_reppo.jsonl
python scripts/analyze.py --stats ../outputs/llm/math_granite/descriptor_stats.json base=../outputs/llm/eval/granite_base.jsonl corners=../outputs/llm/eval/granite_corners.jsonl emphasis=../outputs/llm/eval/granite_emphasis.jsonl ind_ppo=../outputs/llm/eval/granite_ind_ppo.jsonl reppo=../outputs/llm/eval/granite_reppo.jsonl
