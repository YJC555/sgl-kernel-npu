# A3 双节点: 节点内 16 卡走 HCCS, 跨节点先聚合再经 RoCE 发往对端
# 用法: 先在节点 0 执行 bash run_test_internode_a3.sh, 再在节点 1 执行同一条命令
RANK0_IP=""
IP=$(hostname -I | awk '{print $1}')

if [ -z "${RANK0_IP}" ]; then
  echo "please set RANK0_IP to the master node ip first"
  exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir" || exit

export WORLD_SIZE=2                # 节点数(不是总进程数), 总 rank = 2 * 16
export HCCL_BUFFSIZE=3000
export HCCL_INTRA_PCIE_ENABLE=0     # A3 关闭 A2 的分层 PCIe 通信
export HCCL_INTRA_ROCE_ENABLE=1     # A3 跨节点走 RoCE
export DEEPEp_BAREMETAL=1           # 裸机模式: 跳过 K8s 主节点发现

export MASTER_ADDR=${RANK0_IP}
if [ "${IP}" == "${RANK0_IP}" ]; then
  echo "env rank 0"
  export RANK=0
else
  echo "env rank 1"
  export RANK=1
fi

python test_internode_a3.py --num-processes 16
