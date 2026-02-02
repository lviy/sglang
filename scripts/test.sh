#!/bin/bash

# ================= 配置区域 =================
# 模型路径
MODEL_PATH="/gfs/platform/public/infra/Moonlight-16B-A3B-Instruct"
# SGLang 源码路径
SGL_WORKSPACE="/sgl-workspace/sglang"

# 日志设置
LOG_DIR="/sgl-workspace/sglang/dump"
mkdir -p "${LOG_DIR}"
LOG_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/sglang_server_${LOG_TIME}.log"

# ================= 环境清理 =================
echo "Cleaning up environment..."
pkill -9 sglang
sleep 2
pkill -9 ray
pkill -9 python
pkill -9 redis
sleep 2

# ================= 环境变量设置 =================
export PYTHONBUFFERED=1
# 显式将 SGLang 源码加入 PYTHONPATH，确保运行的是你 workspace 里的代码
export PYTHONPATH="${SGL_WORKSPACE}/python:${PYTHONPATH}"

# 检测 NVLink (保留原脚本逻辑，用于设置 NCCL 优化)
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    export NCCL_NVLS_ENABLE=1
    echo "NVLink detected, NCCL_NVLS_ENABLE=1"
else
    export NCCL_NVLS_ENABLE=0
    echo "No NVLink detected"
fi

# ================= 启动 SGLang Server =================
echo "Starting SGLang Server..."
echo "Log file: ${LOG_FILE}"

# 注意：
# 1. SLIME 的 rollout 逻辑是“客户端”行为，这里我们启动的是“服务端”。
# 2. 启动后，你需要另开一个终端运行 benchmark 脚本或发送请求。

nohup python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 2 \
    --ep 2 \
    --mem-fraction-static 0.85 \
    --enable-eplb \
    --eplb-rebalance-num-iterations 500 \
    --eplb-algorithm auto \
    --eplb-min-rebalancing-utilization-threshold 1.0 \
    --ep-num-redundant-experts 0 \
    --moe-a2a-backend none \
    --cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 512 \
    --trust-remote-code \
    > "${LOG_FILE}" 2>&1 &

# 获取 PID
SERVER_PID=$!
echo "SGLang Server launched with PID: ${SERVER_PID}"
echo "To follow logs: tail -f ${LOG_FILE}"

# 等待服务就绪（可选）
# sleep 60