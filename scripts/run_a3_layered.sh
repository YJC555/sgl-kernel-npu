#!/bin/bash
# A3 双机分层测试一键运行: 小规格配置(默认 1024 tokens / 1024MB 窗口), 只输出
# 配置行/打点行/错误行, 过滤 INFO、GM 等噪声。
#
# 用法(两个节点各执行一次, 先从节点后主节点):
#   RANK=1 ./scripts/run_a3_layered.sh
#   RANK=0 ./scripts/run_a3_layered.sh
#
# 只要求先跑通, 需要放大时显式传参:
#   RANK=0 ./scripts/run_a3_layered.sh --num-tokens 2048 --hccl-buffer 2048
#   RANK=0 NUM_TOKENS=2048 HCCL_BUFFSIZE=2048 ./scripts/run_a3_layered.sh
#
# 日志: logs/a3-<时间>-rank<N>.log (完整输出, 未过滤)

set -o pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR" || exit 1

export RANK0_IP="${RANK0_IP:-61.47.19.68}"

# CANN(plog/device)日志统一落到仓库目录下, 便于直接 grep 打点
mkdir -p logs/cann
export ASCEND_PROCESS_LOG_PATH="${ASCEND_PROCESS_LOG_PATH:-$WORKDIR/logs/cann}"

# kernel printf 由 INFO 级别的 device log 承载, 级别必须 <= 1
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-1}"
# 默认不把 CANN host/GE 日志打到 stdout(否则 INFO/GE 刷屏), 全部落 logs/cann;
# 需要实时看全量日志时用 VERBOSE=1 覆盖
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"

NUM_TOKENS="${NUM_TOKENS:-1024}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"

# 让分层窗口按实际 token 数构造, 而不是默认的 8192/轮:
# 窗口需求与 slots(=ROUND*PER_ROUND_TOKENS) 成正比, 设小后显存和 HCCL 窗口都变小
export DEEPEP_NORMAL_LONG_SEQ_ROUND=1
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="$NUM_TOKENS"
export KEEP_LONG_SEQ_ENV=1

# 未显式指定时补上 --num-tokens, 保证 num_tokens <= ROUND*PER_ROUND_TOKENS
has_tokens_arg=0
for a in "$@"; do
  case "$a" in
    --num-tokens|--num_tokens) has_tokens_arg=1 ;;
  esac
done
extra_args=()
[ "$has_tokens_arg" = "0" ] && extra_args+=(--num-tokens "$NUM_TOKENS")

mkdir -p logs
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="logs/a3-${STAMP}-rank${RANK:-auto}.log"

echo "[run] script-rev=2 (终端只回显关键行, CANN 日志落 logs/cann)"
echo "[run] RANK=${RANK:-auto} RANK0_IP=${RANK0_IP} num_tokens=${NUM_TOKENS} 日志: ${LOG}"
echo "[run] CANN 日志目录: $ASCEND_PROCESS_LOG_PATH"

bash tests/python/deepep/run_test_internode_a3.sh "${extra_args[@]}" "$@" > "$LOG" 2>&1 &
test_pid=$!
echo "[run] 已启动 pid=$test_pid (完整输出进日志, 终端只显示关键行)"

# 只把打点/配置/错误行实时回显到终端, INFO 不进终端
tail -n +1 -f "$LOG" 2>/dev/null \
  | grep --line-buffered -E '\[ln\]|\[config\]|routing|ERROR|Error|failed|Failed|invalid|trap|exception|aclnn' &
filter_pid=$!

wait "$test_pid"
rc=$?
kill "$filter_pid" 2>/dev/null
wait "$filter_pid" 2>/dev/null

echo "[run] 退出码 ${rc}, 完整日志: ${LOG}"

# 打点可能落在 CANN 日志目录(PRINT_TO_STDOUT=0)或进程输出(VERBOSE=1), 两处都收
grep -rh '\[ln\]' "$ASCEND_PROCESS_LOG_PATH" "$LOG" 2>/dev/null | tail -200 > logs/last-trace.log
trace_cnt=$(wc -l < logs/last-trace.log 2>/dev/null || echo 0)
if [ "${trace_cnt}" -gt 0 ]; then
  echo "[run] 打点汇总 logs/last-trace.log (${trace_cnt} 行), 最后 20 行:"
  tail -20 logs/last-trace.log
else
  echo "[run] 没抓到 [ln] 打点; 查 trap: grep -r 'MTE_ERR\|FIX_ERR\|aicore error\|trap' $ASCEND_PROCESS_LOG_PATH | tail -20"
fi
exit "$rc"
