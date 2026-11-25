#!/bin/bash
# A3 双机分层测试一键运行: 只输出配置行/打点行/错误行, 过滤 INFO、GM 等噪声。
#
# 用法(两个节点各执行一次, 先从节点后主节点):
#   RANK=1 ./scripts/run_a3_layered.sh
#   RANK=0 ./scripts/run_a3_layered.sh
#
# 可选参数会透传给 run_test_internode_a3.sh, 例如调整窗口大小:
#   RANK=0 ./scripts/run_a3_layered.sh --hccl-buffer 2048
#
# 日志: logs/a3-<时间>-rank<N>.log (完整输出, 未过滤)

set -o pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR" || exit 1

export RANK0_IP="${RANK0_IP:-61.47.19.68}"
export ASCEND_SLOG_PRINT_TO_STDOUT=1
unset ASCEND_GLOBAL_LOG_LEVEL

mkdir -p logs
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="logs/a3-${STAMP}-rank${RANK:-auto}.log"

echo "[run] RANK=${RANK:-auto} RANK0_IP=${RANK0_IP} 日志: ${LOG}"

bash tests/python/deepep/run_test_internode_a3.sh "$@" 2>&1 \
  | tee "$LOG" \
  | grep --line-buffered -E '\[ln\]|\[config\]|routing|ERROR|Error|failed|Failed|invalid|trap|exception|aclnn'

rc=${PIPESTATUS[0]}
echo "[run] 退出码 ${rc}, 完整日志: ${LOG}"
exit "$rc"
