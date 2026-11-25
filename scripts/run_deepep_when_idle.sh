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
#   BUSY_RULE=hbm ./scripts/run_deepep_when_idle.sh  # 改用显存占用判断(默认为进程 pid/name)
#   BUSY_HBM_MB=2048 ./scripts/run_deepep_when_idle.sh # hbm 规则的显存阈值(MB)
#   IGNORE_PROC='^python$' ./scripts/run_deepep_when_idle.sh # 忽略名字匹配的进程
#
# 说明: 启动时会注入 RANK0_IP/RANK/MASTER_ADDR，无需再手工改 a3 脚本里的 RANK0_IP。
set -uo pipefail

# ================= A3 双机默认配置 =================
MASTER_IP="${MASTER_IP:-61.47.19.68}"         # 主节点 IP(本机)
SLAVE_IP="${SLAVE_IP:-61.47.19.69}"           # 从节点 IP
SSH_USER="${SSH_USER:-root}"                  # 从节点 ssh 用户
SSH_PORT="${SSH_PORT:-22}"
SSH_OPTS="${SSH_OPTS:--p $SSH_PORT -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=10}"
WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORKDIR_SLAVE="${WORKDIR_SLAVE:-$WORKDIR}"    # 从节点上的同一工程目录
RUN_CMD="${RUN_CMD:-bash tests/python/deepep/run_test_internode_a3.sh}"
LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"          # 轮询间隔(秒)
IDLE_TIMES="${IDLE_TIMES:-2}"                 # 连续空闲多少次才启动，避免刚结束的任务还没释放
START_DELAY="${START_DELAY:-10}"              # 从节点先起，等几秒再起主节点
BUSY_RULE="${BUSY_RULE:-proc}"                # proc: 按进程 pid/name 判断(默认); hbm: 按显存占用判断
BUSY_HBM_MB="${BUSY_HBM_MB:-1024}"            # hbm 规则阈值(MB)，低于它视为残留缓存不算占用
IGNORE_PROC="${IGNORE_PROC:-}"                # 忽略的进程名正则，例如 '^python$'
# ======================================================

DRY_RUN=0
LOCAL_ONLY=0
while getopts ":nlh" opt; do
    case "$opt" in
        n) DRY_RUN=1 ;;
        l) LOCAL_ONLY=1 ;;
        h) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数 -$OPTARG, 用 -h 查看用法" >&2; exit 1 ;;
    esac
done

log() { echo "[$(date '+%F %T')] $*"; }

# 统一走这里，便于替换认证方式(如改用 -i 指定私钥)
remote_sh() { ssh $SSH_OPTS "$SSH_USER@$SLAVE_IP" "$1"; }

# 解析 npu-smi info 的进程表，输出 "pid name"。
# A3 布局: | npu chip | pid | name | memory |；老布局: | npu pid | name | memory |
proc_pairs_from_stdin() {
    awk -F'|' '
        /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*\|/ {
            pid = $3; name = $4
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", pid)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
            if (pid !~ /^[0-9]+$/) {
                # 老布局: 进程名在第 3 列，pid 在第 2 列最后一个 token
                name = $3
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
                n = split($2, a, /[[:space:]]+/)
                pid = a[n]
            }
            if (pid ~ /^[0-9]+$/) print pid " " name
        }'
}

# 本机进程行中，只要有一个 pid 还活着且不是僵尸，就认为卡被占用。
proc_busy_local_from_stdin() {
    local pid name state
    while read -r pid name; do
        [[ -n "$IGNORE_PROC" && "$name" =~ $IGNORE_PROC ]] && continue
        [[ -d "/proc/$pid" ]] || continue                 # 已退出进程的残留记录
        state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null)"
        [[ "$state" == "Z" ]] && continue                 # 僵尸进程
        return 0
    done < <(proc_pairs_from_stdin)
    return 1
}

# ssh 模式: pid 在对端，需要在从节点上做同样的存活判断。
proc_busy_remote_from_stdin() {
    local pid name
    while read -r pid name; do
        [[ -n "$IGNORE_PROC" && "$name" =~ $IGNORE_PROC ]] && continue
        if remote_sh "test -d /proc/$pid && ! awk '{print \$3}' /proc/$pid/stat 2>/dev/null | grep -qx Z" \
            >/dev/null 2>&1; then
            return 0
        fi
    done < <(proc_pairs_from_stdin)
    return 1
}

# 从 npu-smi info 的芯片行里解析 "used / total"，任一行 used >= 阈值即认为被占用。
# 这样不会因为卡上有几百 MB 的基础缓存（或已退出进程的残留记录）而误判忙碌。
hbm_busy_from_stdin() {
    awk -v th="$BUSY_HBM_MB" '
        /^\|/ {
            s = $0
            while (match(s, /[0-9]+[[:space:]]*\/[[:space:]]*[0-9]+/)) {
                pair = substr(s, RSTART, RLENGTH)
                split(pair, p, "/")
                used = p[1] + 0
                total = p[2] + 0
                if (total > 0 && used >= th) { found = 1 }
                s = substr(s, RSTART + RLENGTH)
            }
        }
        END { exit found ? 0 : 1 }
    '
}

usage_lines_from_stdin() {
    awk '/^\|/ && /\//'
}

show_busy_lines() {
    if [[ "$BUSY_RULE" == "hbm" ]]; then
        usage_lines_from_stdin
    else
        proc_pairs_from_stdin | awk '{printf "pid=%s  name=%s\n", $1, $2}'
    fi
}

command -v npu-smi >/dev/null 2>&1 || { log "本机没有 npu-smi，退出"; exit 1; }

log "主节点 $MASTER_IP (本机)，从节点 $SLAVE_IP (ssh $SSH_USER@$SLAVE_IP:$SSH_PORT)"
log "工程目录: 主=$WORKDIR 从=$WORKDIR_SLAVE"
log "启动命令: $RUN_CMD"
log "占用判定: $BUSY_RULE"$([[ "$BUSY_RULE" == "hbm" ]] && echo " (显存阈值 ${BUSY_HBM_MB}MB，低于它视为残留缓存)" || echo " (按进程 pid/name，忽略已退出/僵尸进程)")
log "轮询 ${POLL_INTERVAL}s，连续空闲 ${IDLE_TIMES} 次后启动$([[ $DRY_RUN == 1 ]] && echo ' (演练模式)')"
[[ "$LOCAL_ONLY" == "1" ]] && log "本机模式: 只检测本机 NPU，不 ssh 对端"

# ssh 模式先探一次，不通就直接给出修复提示，避免无意义轮询
if [[ "$LOCAL_ONLY" == "0" ]]; then
    if ! probe_out="$(remote_sh 'npu-smi info' 2>&1)"; then
        log "ssh 到从节点失败: ${probe_out##*$'\n'}"
        log "请在主节点上配置免密后重试:"
        log "  ssh-copy-id -p $SSH_PORT $SSH_USER@$SLAVE_IP"
        log "  ssh -p $SSH_PORT -o BatchMode=yes $SSH_USER@$SLAVE_IP 'npu-smi info'"
        log "若端口/用户不同: SSH_USER=xxx SSH_PORT=yyy ./scripts/run_deepep_when_idle.sh"
        exit 2
    fi
    log "ssh 到从节点正常"
fi

idle_streak=0
while true; do
    local_out="$(npu-smi info 2>/dev/null || true)"
    remote_out=""
    if [[ "$LOCAL_ONLY" == "0" ]]; then
        remote_out="$(remote_sh 'npu-smi info' 2>/dev/null || true)"
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
    if [[ "$BUSY_RULE" == "hbm" ]]; then
        if printf '%s\n' "$local_out" | hbm_busy_from_stdin; then local_busy=1; fi
        if [[ "$LOCAL_ONLY" == "0" ]] && printf '%s\n' "$remote_out" | hbm_busy_from_stdin; then remote_busy=1; fi
    else
        if printf '%s\n' "$local_out" | proc_busy_local_from_stdin; then local_busy=1; fi
        if [[ "$LOCAL_ONLY" == "0" ]] && printf '%s\n' "$remote_out" | proc_busy_remote_from_stdin; then remote_busy=1; fi
    fi

    if [[ "$local_busy" == "1" || "$remote_busy" == "1" ]]; then
        idle_streak=0
        if [[ "$LOCAL_ONLY" == "1" ]]; then
            log "本机 NPU 忙碌"
        else
            log "忙碌: 主节点=$( [[ $local_busy == 1 ]] && echo 占用 || echo 空闲 ) 从节点=$( [[ $remote_busy == 1 ]] && echo 占用 || echo 空闲 )"
        fi
        if [[ "$local_busy" == "1" ]]; then
            printf '%s\n' "$local_out" | show_busy_lines | sed 's/^/    主: /'
        fi
        if [[ "$remote_busy" == "1" ]]; then
            printf '%s\n' "$remote_out" | show_busy_lines | sed 's/^/    从: /'
        fi
    else
        idle_streak=$((idle_streak + 1))
        if [[ "$BUSY_RULE" == "proc" ]]; then
            printf '%s\n' "$local_out" | proc_pairs_from_stdin | awk '{printf "    忽略残留记录: pid=%s name=%s\n", $1, $2}'
        fi
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
remote_sh "$remote_cmd" || { log "从节点拉起失败"; exit 1; }
sleep "$START_DELAY"
log "在主节点拉起任务"
eval "$local_cmd"

log "已启动"
log "  从节点日志: $SLAVE_IP:$remote_log"
log "  主节点日志: $local_log"
log "查看打点: grep -h '\\[ln\\]' $LOG_DIR/master-$stamp.log | tail -50"
