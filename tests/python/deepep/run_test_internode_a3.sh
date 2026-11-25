# A3 双节点: 节点内 16 卡走 HCCS, 跨节点先聚合再经 RoCE 发往对端
# 用法: 先在节点 0 执行 bash run_test_internode_a3.sh, 再在节点 1 执行同一条命令
# 也支持由外部注入: RANK0_IP=<主节点IP> bash run_test_internode_a3.sh
RANK0_IP="${RANK0_IP:-}"
IP=$(hostname -I | awk '{print $1}')

if [ -z "${RANK0_IP}" ]; then
  echo "please set RANK0_IP to the master node ip first"
  exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir" || exit

# 窗口容量支持外部传参, 优先级: 环境变量 HCCL_BUFFSIZE > --hccl-buffer 参数 > 默认值
#   HCCL_BUFFSIZE=8192 bash run_test_internode_a3.sh
#   bash run_test_internode_a3.sh --hccl-buffer 8192
HCCL_BUFFER_MB=""
python_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -b|--hccl-buffer)
      HCCL_BUFFER_MB="${2:?--hccl-buffer 需要 MB 数值}"
      shift 2
      ;;
    *)
      python_args+=("$1")
      shift
      ;;
  esac
done

export WORLD_SIZE=2                # 节点数(不是总进程数), 总 rank = 2 * 16
# 分层模式窗口需求(见 ops2/op_kernel/moe_distribute_dispatch_v2_layered.h)：
#   winSizeMin = moeExpertNum * bs * (h*2 + k*4*3) + 800MB(RDMA) + IPC
# 4096 tokens / 256 experts / h=7168 / k=8 时窗口需求约 1~2GB，默认取 2048MB；
# 不够时按报错提示用 --hccl-buffer 或 HCCL_BUFFSIZE 调大。
# "The HCCL_BUFFSIZE is xxxMB, the min value should be xxxxMB" 断言。
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-${HCCL_BUFFER_MB:-2048}}
export DEEPEP_HCCL_BUFFSIZE=${DEEPEP_HCCL_BUFFSIZE:-${HCCL_BUFFSIZE}}
echo "[config] HCCL_BUFFSIZE=${HCCL_BUFFSIZE}MB"
export HCCL_INTRA_PCIE_ENABLE=0     # A3 关闭 A2 的分层 PCIe 通信
export HCCL_INTRA_ROCE_ENABLE=1     # A3 跨节点走 RoCE
export DEEPEp_BAREMETAL=1           # 裸机模式: 跳过 K8s 主节点发现

# A3 分层通信: 节点内 16 卡先经 HCCS 聚合, 再经 RoCE 发到对端, 最后经 HCCS 送到目标专家
# 置 1 时依赖 CANN >= 9.2.0; 置 0 则回退到 HCCL 全互联(CamMoeDispatchNormal)
LAYERED=1
if [ "${LAYERED}" = "1" ]; then
  export DEEPEP_LAYERED_RANKS_PER_SERVER=16
  export DEEPEP_LAYERED_PROTOCAL=roce
fi

# 清理 shell 里可能残留的"蚂蚁搬家"长序列配置(之前跑过其他回归时 export 过)：
# Buffer 构造要求 num_tokens <= ROUND * PER_ROUND_TOKENS，本测试 num_tokens=4096，
# 残留 ROUND=1 / PER_ROUND_TOKENS=32 会直接触发断言。
if [ "${KEEP_LONG_SEQ_ENV:-0}" = "1" ]; then
  echo "[config] 保留外部长序列配置: ROUND=${DEEPEP_NORMAL_LONG_SEQ_ROUND:-unset} PER_ROUND_TOKENS=${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-unset}"
else
  unset DEEPEP_NORMAL_LONG_SEQ_ROUND DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS
fi

export MASTER_ADDR=${RANK0_IP}
if [ -n "${RANK:-}" ]; then
  # 外部脚本可直接注入 RANK，避免 hostname -I 与 RANK0_IP 不一致时判错
  export RANK
  echo "env rank ${RANK} (from env)"
elif [ "${IP}" = "${RANK0_IP}" ]; then
  echo "env rank 0"
  export RANK=0
else
  echo "env rank 1"
  export RANK=1
fi

python test_internode_a3.py --num-processes 16 "${python_args[@]}"
