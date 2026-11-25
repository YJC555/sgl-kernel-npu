#!/bin/bash
# 等主节点(本机)和从节点的 NPU 都空闲后，自动拉起双机 DeepEP 任务。
# 用法:
#   ./scripts/run_deepep_when_idle.sh            # ssh 模式: 两节点都空闲后自动拉起两边
#   ./scripts/run_deepep_when_idle.sh -n         # 只监测并打印状态，不真正启动(演练)
#   ./scripts/run_deepep_when_idle.sh -l         # 本机模式: 无 ssh 互通时，两台机器各跑一次
# 常用环境变量:
#   ./scripts/run_deepep_when_idle.sh              # 默认 A3: 主 61.47.19.68, 从 61.47.19.69
#   MASTER_IP=x SLAVE_IP=y SSH_USER=user ./scripts/run_deepep_when_idle.sh
#   RUN_CMD='bash tests/python/deepep/run_test_internode_a3.sh'   # 默认值即此
#   RANK=0|1 ./scripts/run_deepep_when_idle.sh -l  # 本机模式建议显式指定 RANK
#
# 说明: 启动时会注入 RANK0_IP/RANK/MASTER_ADDR，无需再手工改 a3 脚本里的 RANK0_IP。
set -uo pipefail

# ================= A3 双机默认配置 =================
MASTER_IP="${MASTER_IP:-61.47.19.68}"         # 主节点 IP(本机)
SLAVE_IP="${SLAVE_IP:-61.47.19.69}"           # 从节点 IP
SSH_USER="${SSH_USER:-root}"                  # 从节点 ssh 用户
SSH_PORT="${SSH_PORT:-22}"
WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORKDIR_SLAVE="${WORKDIR_SLAVE:-$WORKDIR}"    # 从节点上的同一工程目录
RUN_CMD="${RUN_CMD:-bash tests/python/deepep/run_test_internode_a3.sh}"
LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"          # 轮询间隔(秒)
IDLE_TIMES="${IDLE_TIMES:-2}"                 # 连续空闲多少次才启动，避免刚结束的任务还没释放
START_DELAY="${START_DELAY:-10}"              # 从节点先起，等几秒再起主节点
# ======================================================

DRY_RUN=0
LOCAL_ONLY=0
while getopts ":nlh" opt; do
    case "$opt" in
        n) DRY_RUN=1 ;;
        l) LOCAL_ONLY=1 ;;
        h) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数 -$OPTARG, 用 -h 查看用法" >&2; exit 1 ;;
    esac
done

log() { echo "[$(date '+%F %T')] $*"; }

# npu-smi info 里每个被占用的进程是一行 "| <npu>  <pid> | <进程名> | <显存> |"，
# 只匹配这种行；有匹配 => 有进程在用 => busy(返回 0)，否则 idle(返回 1)。
busy_from_stdin() {
    awk '/^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*\|/ { found = 1 } END { exit found ? 0 : 1 }'
}

proc_lines_from_stdin() {
    awk '/^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*\|/'
}

command -v npu-smi >/dev/null 2>&1 || { log "本机没有 npu-smi，退出"; exit 1; }

log "主节点 $MASTER_IP (本机)，从节点 $SLAVE_IP (ssh $SSH_USER@$SLAVE_IP:$SSH_PORT)"
log "工程目录: 主=$WORKDIR 从=$WORKDIR_SLAVE"
log "启动命令: $RUN_CMD"
log "轮询 ${POLL_INTERVAL}s，连续空闲 ${IDLE_TIMES} 次后启动$([[ $DRY_RUN == 1 ]] && echo ' (演练模式)')"
[[ "$LOCAL_ONLY" == "1" ]] && log "本机模式: 只检测本机 NPU，不 ssh 对端"

idle_streak=0
while true; do
    local_out="$(npu-smi info 2>/dev/null || true)"
    remote_out=""
    if [[ "$LOCAL_ONLY" == "0" ]]; then
        remote_out="$(ssh -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=5 \
            "$SSH_USER@$SLAVE_IP" 'npu-smi info' 2>/dev/null || true)"
    fi

    if [[ -z "$local_out" ]]; then
        log "读取本机 npu-smi 失败，继续等待"
        idle_streak=0
        sleep "$POLL_INTERVAL"
        continue
    fi
    if [[ "$LOCAL_ONLY" == "0" && -z "$remote_out" ]]; then
        log "ssh 到 $SLAVE_IP 失败或无输出，继续等待(不盲跑)"
        idle_streak=0
        sleep "$POLL_INTERVAL"
        continue
    fi

    local_busy=0
    remote_busy=0
    printf '%s\n' "$local_out" | busy_from_stdin && local_busy=1
    if [[ "$LOCAL_ONLY" == "0" ]]; then
        printf '%s\n' "$remote_out" | busy_from_stdin && remote_busy=1
    fi

    if [[ "$local_busy" == "1" || "$remote_busy" == "1" ]]; then
        idle_streak=0
        if [[ "$LOCAL_ONLY" == "1" ]]; then
            log "本机 NPU 忙碌"
        else
            log "忙碌: 主节点=$( [[ $local_busy == 1 ]] && echo 占用 || echo 空闲 ) 从节点=$( [[ $remote_busy == 1 ]] && echo 占用 || echo 空闲 )"
        fi
        [[ "$local_busy" == "1" ]] && printf '%s\n' "$local_out" | proc_lines_from_stdin | sed 's/^/    主: /'
        [[ "$remote_busy" == "1" ]] && printf '%s\n' "$remote_out" | proc_lines_from_stdin | sed 's/^/    从: /'
    else
        idle_streak=$((idle_streak + 1))
        if [[ "$LOCAL_ONLY" == "1" ]]; then
            log "本机 NPU 空闲 (${idle_streak}/${IDLE_TIMES})"
        else
            log "两节点 NPU 均空闲 (${idle_streak}/${IDLE_TIMES})"
        fi
        if (( idle_streak >= IDLE_TIMES )); then
            break
        fi
    fi
    sleep "$POLL_INTERVAL"
done

stamp="$(date '+%Y%m%d-%H%M%S')"
mkdir -p "$LOG_DIR"

# 本机模式: 无 ssh 互通，两台机器各自执行一次，各自只等本机空闲
if [[ "$LOCAL_ONLY" == "1" ]]; then
    if [[ -z "${RANK:-}" ]]; then
        if [[ "$(hostname -I 2>/dev/null | awk '{print $1}')" == "$MASTER_IP" ]]; then
            RANK=0
        else
            RANK=1
        fi
        log "未指定 RANK，按 hostname -I 推断为 RANK=$RANK"
    fi
    role=$([[ "$RANK" == "0" ]] && echo master || echo slave)
    local_log="$LOG_DIR/$role-$stamp.log"
    local_cmd="cd '$WORKDIR' && RANK=$RANK RANK0_IP='$MASTER_IP' MASTER_ADDR='$MASTER_IP' setsid nohup bash -lc '$RUN_CMD' > '$local_log' 2>&1 < /dev/null &"

    if [[ "$DRY_RUN" == "1" ]]; then
        log "演练模式，将执行: $local_cmd"
        exit 0
    fi

    eval "$local_cmd"
    log "已在本机启动: RANK=$RANK ($role)，日志 $local_log"
    log "另一端也要在 HCCL 连接超时(默认约120s)内起来，建议两边同时执行本命令"
    log "查看打点: grep -h '\\[ln\\]' $local_log | tail -50"
    exit 0
fi

remote_log_dir="${REMOTE_LOG_DIR:-$WORKDIR_SLAVE/logs}"
remote_log="$remote_log_dir/slave-$stamp.log"
local_log="$LOG_DIR/master-$stamp.log"

remote_cmd="mkdir -p '$remote_log_dir' && cd '$WORKDIR_SLAVE' && RANK=1 RANK0_IP='$MASTER_IP' MASTER_ADDR='$MASTER_IP' setsid nohup bash -lc '$RUN_CMD' > '$remote_log' 2>&1 < /dev/null &"
local_cmd="cd '$WORKDIR' && RANK=0 RANK0_IP='$MASTER_IP' MASTER_ADDR='$MASTER_IP' setsid nohup bash -lc '$RUN_CMD' > '$local_log' 2>&1 < /dev/null &"

if [[ "$DRY_RUN" == "1" ]]; then
    log "演练模式，将执行:"
    log "  从节点: ssh -p $SSH_PORT $SSH_USER@$SLAVE_IP \"$remote_cmd\""
    log "  主节点: $local_cmd"
    exit 0
fi

log "先在从节点拉起任务"
ssh -p "$SSH_PORT" "$SSH_USER@$SLAVE_IP" "$remote_cmd" || { log "从节点拉起失败"; exit 1; }
sleep "$START_DELAY"
log "在主节点拉起任务"
eval "$local_cmd"

log "已启动"
log "  从节点日志: $SLAVE_IP:$remote_log"
log "  主节点日志: $local_log"
log "查看打点: grep -h '\\[ln\\]' $LOG_DIR/master-$stamp.log | tail -50"
