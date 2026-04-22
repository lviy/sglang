LOG_TS=$(date +%Y%m%d%H%M)
export PYTHONPATH=/gfs/platform/public/infra/lxr/sglang/python:$PYTHONPATH

pip install --upgrade transformers
pip install sglang-kernel --force-reinstall

CUDA_VISIBLE_DEVICES=0,1 \
python3 ./scripts/playground/bench_speculative.py \
--model-path /gfs/space/chatrl/public/models/ZhipuAI/GLM-4.7-Flash-0309 \
--tp-size 2 \
--trust-remote-code \
--batch-size 128 \
--steps 2 3 4 \
--topk 1 2 \
--num_draft_tokens 3 4 5 \
2>&1 | tee "/gfs/platform/public/infra/lxr/logs/sglang_spec_bench_${LOG_TS}.log"
