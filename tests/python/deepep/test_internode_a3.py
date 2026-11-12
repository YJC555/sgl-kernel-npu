"""A3 双节点 internode dispatch / combine 测试。

拓扑（A3）:
    - 2 个节点，节点 0 与节点 1，每个节点 16 颗 NPU，共 32 个 EP rank
    - 本节点内 16 张卡走 HCCS
    - 跨节点时先在本节点内聚合，再通过 RoCE 组网发往对端节点，
      到对端后再经 HCCS 送到目标专家

测试逻辑与 test_internode_a2.py 保持一致:
    get_dispatch_layout 校验 -> dispatch/combine 正确性 -> 性能 tuning -> 可选 diagnose
"""

import argparse
import os
import random
import time
from typing import Optional

# noinspection PyUnresolvedReferences
import deep_ep
import torch
import torch.distributed as dist
import torch_npu
from utils import (
    bench,
    bench_kineto,
    calc_diff,
    diagnose_matrix,
    get_diff_threshold,
    init_dist,
    inplace_unique,
    per_token_cast_back,
)

MAX_BATCH_SIZE = 4096

# A3 硬件拓扑: 2 节点 x 16 卡
A3_LOCAL_RANKS = 16
A3_NUM_NODES = 2

# A3 正常模式 dispatch 在 device 侧对应的 kernel 名
# 分层模式（DEEPEP_LAYERED_RANKS_PER_SERVER=16）: 节点内 HCCS 聚合 -> RoCE 跨节点 -> 对端 HCCS 分发
A3_LAYERED_DISPATCH_KERNELS = (
    "CamMoeDispatchNormalLayered",
    "NotifyDispatchNormalLayered",
)
# 非分层模式: HCCL 全互联，CamMoeDispatchNormal 内部完成节点间通信
A3_FULLMESH_DISPATCH_KERNELS = ("CamMoeDispatchNormal", "NotifyDispatch")


def log_rank(rank: int, stage: str, message: str) -> None:
    """统一日志打印: 带 rank 前缀并立即 flush, 所有 rank 均会输出。"""
    print(f"[Rank {rank}] [{stage}] {message}", flush=True)


# noinspection PyShadowingNames
def test_main(
    args: argparse.Namespace,
    num_local_ranks: int,
    local_rank: int,
    num_ranks: int,
    rank: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
):
    # Settings
    base_num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    enable_diagnose = args.enable_diagnose
    enable_dynamic_tokens = args.enable_dynamic_tokens
    # A3: 每 16 个 rank 组成一个节点（节点内 HCCS），跨节点走 RoCE
    num_nodes = num_ranks // num_local_ranks
    # num_rdma_ranks > 1 说明启用了 A3 分层通信（DEEPEP_LAYERED_RANKS_PER_SERVER）
    is_layered = buffer.runtime.get_num_rdma_ranks() > 1
    dispatch_kernels = (
        A3_LAYERED_DISPATCH_KERNELS if is_layered else A3_FULLMESH_DISPATCH_KERNELS
    )
    # handle 布局不同: 分层/internode 为 (recv_src_idx, is_token_in_rank, send_head, topk_idx,
    # topk_weights, ...)，全互联(intranode) 为 (..., is_token_in_rank, send_head, topk_idx, topk_weights)
    topk_weights_idx = 4 if is_layered else 7
    expert_token_nums_type = int(os.getenv("MOE_EXPERT_TOKEN_NUMS_TYPE", 1))

    if enable_dynamic_tokens:
        fluctuation_percentage = 0.1
        min_fluctuation = 2

        if base_num_tokens < 10:
            fluctuation = random.randint(-min_fluctuation, min_fluctuation)
            num_tokens = base_num_tokens + fluctuation
        else:
            fluctuation = random.uniform(
                1 - fluctuation_percentage, 1 + fluctuation_percentage
            )
            num_tokens = int(base_num_tokens * fluctuation)

        # Ensure num_tokens is at least 1
        num_tokens = max(num_tokens, 1)
    else:
        num_tokens = base_num_tokens

    log_rank(
        rank,
        "config",
        f"num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, "
        f"num_experts={num_experts}, num_local_ranks={num_local_ranks}, num_nodes={num_nodes}, "
        f"num_rdma_ranks={buffer.runtime.get_num_rdma_ranks()}, is_layered={is_layered}, "
        f"dispatch_kernels={dispatch_kernels}, topk_weights_idx={topk_weights_idx}, "
        f"expert_token_nums_type={expert_token_nums_type}, "
        f"active_ranks={args.active_ranks or 'all'}",
    )

    assert num_experts % num_ranks == 0 and num_nodes >= 2
    assert num_tokens <= MAX_BATCH_SIZE
    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, "
            f"num_experts={num_experts}, num_local_ranks={num_local_ranks}, num_nodes={num_nodes}, "
            f"routing={'layered(HCCS->RoCE->HCCS)' if is_layered else 'fullmesh(HCCL)'}, "
            f"active_ranks={args.active_ranks}",
            flush=True,
        )

    experts_per_rank = num_experts // num_ranks

    if args.active_ranks:
        # Only assign tokens to the specified ranks
        try:
            active_ranks = [
                int(r.strip()) for r in args.active_ranks.split(",") if r.strip()
            ]
        except ValueError:
            raise ValueError(
                f"Invalid value in --active-ranks: {args.active_ranks}. "
                f"Must be a comma-separated list of integers, e.g., '0,1,3'."
            )

        # Validate range
        if any(r < 0 or r >= num_ranks for r in active_ranks):
            raise ValueError(
                f"Invalid rank in --active-ranks: {active_ranks}. "
                f"Ranks must be in range [0, {num_ranks-1}]."
            )

        if not active_ranks:
            raise ValueError(
                "Parsed --active-ranks is empty. Provide at least one valid rank."
            )

        valid_experts = torch.cat(
            [
                torch.arange(
                    r * experts_per_rank, (r + 1) * experts_per_rank, device="npu"
                )
                for r in active_ranks
            ]
        )
        # Randomly sample experts from active ranks only
        topk_idx = valid_experts[
            torch.randint(0, len(valid_experts), (num_tokens, num_topk), device="npu")
        ]
    else:
        # Default: random over all experts (original behavior)
        scores = (
            torch.randn(
                (num_tokens, num_experts), dtype=torch.float32, device="npu"
            ).abs()
            + 1
        )
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]

    topk_weights = (
        torch.ones((num_tokens, num_topk), dtype=torch.float32, device="npu") * rank
    )

    if args.topk_drop_prob > 0 or args.topk_drop_row >= 0:
        topk_idx_dropped = topk_idx.clone()
        topk_weights_dropped = topk_weights.clone()

        # Random drop (based on probability)
        if args.topk_drop_prob > 0:
            drop_mask = (
                torch.rand_like(topk_idx, dtype=torch.float32) < args.topk_drop_prob
            )
            topk_idx_dropped = topk_idx.clone()
            topk_idx_dropped = topk_idx_dropped.masked_fill(drop_mask, -1)

            # Construct topk_weights_dropped
            invalid_mask = topk_idx_dropped == -1
            topk_weights_dropped = topk_weights_dropped.masked_fill(invalid_mask, 0.0)

        # Fixed column drop (for the test_topk_minus1 scenario)
        if args.topk_drop_row >= 0 and args.topk_drop_row < num_tokens:
            topk_idx_dropped[args.topk_drop_row, :] = -1
            topk_weights_dropped[args.topk_drop_row, :] = 0

        # print drop ratio
        drop_ratio = (topk_idx_dropped == -1).float().mean().item()
        if rank == 0:
            print(
                f"[DEBUG] [rank {rank}] topk dropped ratio = {drop_ratio*100:.2f}%",
                flush=True,
            )
        topk_idx = topk_idx_dropped
        topk_weights = topk_weights_dropped

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # 跨节点（RoCE）发送的 token 数：专家按节点连续划分，[0, num_experts/num_nodes) 属于节点 0
    rdma_idx = topk_idx // (num_experts // num_nodes)
    rdma_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rdma_idx, num_nodes)
    num_rdma_token_sent = rdma_idx.ne(-1).sum().item()

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="npu")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="npu")
    token_idx_in_rank = torch.full(
        (num_ranks, num_tokens), -1, dtype=torch.long, device="npu"
    )

    for i in range(num_ranks):
        token_sel = (rank_idx == i).max(dim=-1)[0]  # [num_tokens]
        token_indices = torch.nonzero(token_sel, as_tuple=True)[0]  # [count]
        count = token_indices.numel()
        num_tokens_per_rank[i] = count
        if count > 0:
            token_idx_in_rank[i][token_indices] = torch.arange(count, device="npu")

    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = (token_idx_in_rank >= 0).to(torch.int)
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    log_rank(
        rank,
        "layout-input",
        f"local num_tokens_per_rank(sum={int(num_tokens_per_rank.sum())}, "
        f"max={int(num_tokens_per_rank.max())}, min={int(num_tokens_per_rank.min())}), "
        f"num_rdma_token_sent={num_rdma_token_sent}, "
        f"local num_tokens_per_expert(sum={int(num_tokens_per_expert.sum())})",
    )

    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    log_rank(rank, "layout-bench", f"get_dispatch_layout kernel: {t * 1000:.3f} ms")
    print("", flush=True)
    dist.barrier()
    time.sleep(1)

    try:
        try:
            return_values = buffer.get_dispatch_layout(topk_idx, num_experts)
        except Exception as e:
            log_rank(rank, "layout-check", f"调用 get_dispatch_layout 失败: {e}")
            raise

        (
            ref_num_tokens_per_rank,
            _,
            ref_num_tokens_per_expert,
            ref_is_token_in_rank,
            _,
        ) = return_values
        try:
            assert torch.allclose(
                ref_num_tokens_per_rank, num_tokens_per_rank
            ), f"Assertion num_tokens_per_rank failed on rank {rank}: Expected {num_tokens_per_rank}, Actual {ref_num_tokens_per_rank}"
            assert torch.allclose(
                ref_num_tokens_per_expert, num_tokens_per_expert
            ), f"Assertion num_tokens_per_expert failed on rank {rank}: Expected {num_tokens_per_expert}, Actual {ref_num_tokens_per_expert}"
            assert torch.allclose(
                ref_is_token_in_rank, is_token_in_rank
            ), f"Assertion is_token_in_rank failed on rank {rank}: Expected {is_token_in_rank}, Actual {ref_is_token_in_rank}"
        except AssertionError as e:
            log_rank(rank, "layout-check", f"结果校验失败: {e}")
            raise
        log_rank(rank, "layout-check", "get_dispatch_layout 结果与本地参考一致")
    except Exception as e:
        log_rank(rank, "layout-check", f"发生异常: {e}")

    # Config: 32 个 EP rank（A3 双节点）对应的推荐配置
    dispatch_config = deep_ep.Buffer.get_dispatch_config(num_ranks)
    combine_config = deep_ep.Buffer.get_combine_config(num_ranks)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    topk_weights = (
        torch.ones((num_tokens, num_topk), dtype=torch.float32, device="npu") * rank
    )
    topk_weights_pure_rand = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    )

    # Test diagnose function
    # noinspection PyShadowingNames
    def test_diagnose(
        dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
        combine_send_cost_stats: Optional[torch.Tensor] = None,
    ):
        log_rank(
            rank,
            "diagnose",
            f"dispatch_wait_recv_cost_stats={'on' if dispatch_wait_recv_cost_stats is not None else 'off'}, "
            f"combine_send_cost_stats={'on' if combine_send_cost_stats is not None else 'off'}",
        )
        for current_x in filter(lambda elem: elem is not None, (x_pure_rand,)):
            dispatch_args = {
                "x": current_x,
                "num_tokens_per_rank": num_tokens_per_rank,
                "is_token_in_rank": is_token_in_rank,
                "num_tokens_per_expert": num_tokens_per_expert,
                "config": dispatch_config,
                "topk_idx": topk_idx,
                "topk_weights": topk_weights_pure_rand,
                "dispatch_wait_recv_cost_stats": dispatch_wait_recv_cost_stats,
            }
            if dispatch_wait_recv_cost_stats is not None:
                bench(lambda: buffer.dispatch(**dispatch_args), num_warmups=0)
            if combine_send_cost_stats is not None:
                (
                    recv_x,
                    recv_topk_idx,
                    recv_topk_weights,
                    recv_num_tokens_per_expert_list,
                    handle,
                    event,
                ) = buffer.dispatch(**dispatch_args)
                recv_x = (
                    per_token_cast_back(*recv_x)
                    if isinstance(recv_x, tuple)
                    else recv_x
                )
                combine_args = {
                    "x": recv_x,
                    "handle": handle,
                    "topk_weights": handle[topk_weights_idx],
                    "config": combine_config,
                    "async_finish": False,
                    "combine_send_cost_stats": combine_send_cost_stats,
                }
                bench(lambda: buffer.combine(**combine_args), num_warmups=0)
        for stats, title in (
            (dispatch_wait_recv_cost_stats, "Dispatch wait recv cost"),
            (combine_send_cost_stats, "Combine send cost"),
        ):
            if stats is None:
                continue
            gather_list = (
                [torch.zeros_like(stats) for _ in range(group.size())]
                if rank == 0
                else None
            )
            dist.gather(stats, gather_list=gather_list, group=group, dst=0)
            if rank == 0:
                stats_mat = torch.stack(gather_list, dim=0)
                print(f"{title} stats:")
                print(stats_mat)
                res = diagnose_matrix(
                    stats_mat, thres_col=1.0, thres_row=2.0, thres_point=5.0
                )
                print(
                    f"[Diagnose {title}] abnormal_rows {res['abnormal_rows']}, "
                    f"abnormal_cols {res['abnormal_cols']}, abnormal_points {res['abnormal_points']}"
                )

    def test_correctness():
        for current_x in filter(lambda elem: elem is not None, (x_pure_rand, x)):
            x_tag = "x_pure_rand" if current_x is x_pure_rand else "x"
            if local_rank == 0:
                print(
                    f"[testing] Running with BF16, with top-k {num_topk} ...",
                    flush=True,
                )
            log_rank(
                rank,
                "correctness",
                f"dispatch start, data={x_tag}, x.shape={tuple(current_x.shape)}",
            )
            # Test dispatch
            dispatch_args = {
                "x": current_x,
                "num_tokens_per_rank": num_tokens_per_rank,
                "is_token_in_rank": is_token_in_rank,
                "num_tokens_per_expert": num_tokens_per_expert,
                "config": dispatch_config,
                "topk_idx": topk_idx,
                "topk_weights": (
                    topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                ),
            }

            (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                recv_num_tokens_per_expert_list,
                handle,
                event,
            ) = buffer.dispatch(**dispatch_args)
            recv_x = (
                per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
            )
            log_rank(
                rank,
                "correctness",
                f"dispatch done, data={x_tag}, recv_x.shape={tuple(recv_x.shape)}, "
                f"recv_num_tokens_per_expert={recv_num_tokens_per_expert_list}, "
                f"handle.len={len(handle)}, "
                f"handle[{topk_weights_idx}].shape={tuple(handle[topk_weights_idx].shape)}",
            )

            # Checks
            local_expert_token = gbl_num_tokens_per_expert.view(num_ranks, -1)[rank]
            if expert_token_nums_type == 0:
                local_expert_token_list = local_expert_token.cumsum(
                    dim=0
                ).tolist()  # 计算前缀和并转为 list
            else:
                local_expert_token_list = local_expert_token.tolist()

            assert local_expert_token_list == recv_num_tokens_per_expert_list
            log_rank(
                rank,
                "correctness",
                f"expert token nums check passed, data={x_tag}, "
                f"local_expert_token_list={local_expert_token_list}",
            )

            # Test combine
            combine_args = {
                "x": recv_x,
                "handle": handle,
                "config": combine_config,
                "async_finish": False,
                "topk_weights": handle[topk_weights_idx],
            }
            combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
            check_x = combined_x.float()
            ref_x = x_pure_rand if current_x is x_pure_rand else x
            mask = ~torch.all(topk_idx == -1, dim=1)
            desire_x = ref_x * handle[topk_weights_idx].masked_fill(
                topk_idx == -1, 0
            ).sum(dim=1).view(-1, 1)
            diff = calc_diff(check_x[mask], desire_x[mask])
            log_rank(
                rank,
                "correctness",
                f"combine done, data={x_tag}, combined_x.shape={tuple(combined_x.shape)}, "
                f"diff={diff:.6f}, threshold={get_diff_threshold('bf16')}",
            )
            assert diff < get_diff_threshold("bf16")

            log_rank(rank, "correctness", f"passed, data={x_tag}")
        if local_rank == 0:
            print("", flush=True)

    def test_tuning():
        dispatch_args = {
            "x": x,
            "num_tokens_per_rank": num_tokens_per_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "config": dispatch_config,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
        }
        recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)
        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
        log_rank(
            rank,
            "tuning",
            f"dispatch done for tuning, recv_x.shape={tuple(recv_x.shape)}, "
            f"handle.len={len(handle)}",
        )

        # For later tuning
        dispatch_bf16_remote_send_bytes = num_rdma_token_sent * hidden * 2
        dispatch_bf16_recv_bytes = recv_x.numel() * 2
        combine_bf16_send_bytes = dispatch_bf16_recv_bytes
        combine_bf16_remote_recv_bytes = dispatch_bf16_remote_send_bytes

        # Tune dispatch performance
        for current_x in filter(lambda elem: elem is not None, (x,)):
            recv_bytes = dispatch_bf16_recv_bytes
            remote_send_bytes = dispatch_bf16_remote_send_bytes

            tune_args = {
                "x": current_x,
                "config": dispatch_config,
                "num_tokens_per_rank": num_tokens_per_rank,
                "is_token_in_rank": is_token_in_rank,
                "num_tokens_per_expert": num_tokens_per_expert,
                "topk_idx": topk_idx,
                "topk_weights": topk_weights,
            }

            t, notify_t = bench_kineto(
                lambda: buffer.dispatch(**tune_args),
                dispatch_kernels,
            )
            log_rank(
                rank,
                "tuning",
                f"Dispatch (BF16) {recv_bytes / 1e9 / t:.2f} GB/s (HCCS), "
                f"{remote_send_bytes / 1e9 / t:.2f} GB/s (RoCE cross-node), "
                f"avg_t: {t * 1e6:.2f} us, notify_t: {notify_t * 1e6:.2f} us, "
                f"kernels={dispatch_kernels}",
            )
            if local_rank == 0:
                print("", flush=True)

        # Tune combine performance
        tune_args = {
            "x": recv_x,
            "handle": handle,
            "config": combine_config,
            "async_finish": False,
            "topk_weights": handle[topk_weights_idx],
        }
        t = bench(lambda: buffer.combine(**tune_args))[0]
        log_rank(
            rank,
            "tuning",
            f"Combine {combine_bf16_send_bytes / 1e9 / t:.2f} GB/s (HCCS), "
            f"{combine_bf16_remote_recv_bytes / 1e9 / t:.2f} GB/s (RoCE cross-node), "
            f"avg_t: {t * 1e6:.2f} us",
        )
        if local_rank == 0:
            print("", flush=True)

    log_rank(rank, "stage", "开始 correctness 校验")
    test_correctness()
    log_rank(rank, "stage", "correctness 校验完成, 开始性能 tuning")
    test_tuning()
    log_rank(rank, "stage", "性能 tuning 完成")

    # Diagnose test
    if enable_diagnose:
        log_rank(rank, "diagnose", "开始 diagnose 测试")
        dispatch_wait_recv_cost_stats = torch.zeros(
            (num_ranks,), dtype=torch.int32, device="npu"
        )
        combine_send_cost_stats = torch.zeros(
            (num_ranks,), dtype=torch.int32, device="npu"
        )
        test_diagnose(
            dispatch_wait_recv_cost_stats=dispatch_wait_recv_cost_stats,
            combine_send_cost_stats=combine_send_cost_stats,
        )
        log_rank(rank, "diagnose", "diagnose 测试完成")


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    print(f"[local_rank {local_rank}] 开始 init_dist", flush=True)
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    log_rank(
        rank,
        "init",
        f"dist 初始化完成, num_ranks={num_ranks}, num_local_ranks={num_local_ranks}",
    )

    # A3 拓扑校验: 2 节点 x 16 卡
    assert num_local_ranks == A3_LOCAL_RANKS, (
        f"A3 expects {A3_LOCAL_RANKS} NPUs per node, got num_local_ranks={num_local_ranks}. "
        f"Please run with --num-processes {A3_LOCAL_RANKS}."
    )
    assert num_ranks == A3_LOCAL_RANKS * A3_NUM_NODES, (
        f"A3 expects {A3_NUM_NODES} nodes x {A3_LOCAL_RANKS} NPUs = "
        f"{A3_LOCAL_RANKS * A3_NUM_NODES} ranks, got num_ranks={num_ranks}. "
        f"Check WORLD_SIZE (number of nodes) and RANK."
    )
    log_rank(
        rank,
        "init",
        f"A3 拓扑校验通过 ({A3_NUM_NODES} nodes x {A3_LOCAL_RANKS} NPUs)",
    )

    log_rank(rank, "init", f"local_rank={local_rank}, 开始创建 Buffer...")
    t0 = time.time()
    buffer = deep_ep.Buffer(
        group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1
    )
    log_rank(
        rank,
        "init",
        f"Buffer 创建完成, 耗时 {time.time() - t0:.2f}s, "
        f"num_rdma_ranks={buffer.runtime.get_num_rdma_ranks()}",
    )
    torch.manual_seed(rank)

    log_rank(rank, "stage", "开始 test_main")
    test_main(args, num_local_ranks, local_rank, num_ranks, rank, buffer, group)
    log_rank(rank, "done", "test_main 全部完成")
    if local_rank == 0:
        print("", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test A3 internode EP kernels")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of local processes to spawn, i.e. NPUs per node (default: 16)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=4096, help="Number of tokens (default: 4096)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256)"
    )
    parser.add_argument(
        "--active-ranks",
        type=str,
        default="",
        help="Comma-separated list of ranks that will receive tokens. "
        'Example: "0,1,3". If empty, all ranks may receive tokens.',
    )
    parser.add_argument(
        "--enable-diagnose",
        action="store_true",
        help="Whether to enable diagnose for testing",
    )
    parser.add_argument(
        "--topk-drop-prob",
        dest="topk_drop_prob",
        type=float,
        default=0.0,
        help="Probability of randomly dropping a top-k index (set to -1).",
    )
    parser.add_argument(
        "--topk-drop-row",
        dest="topk_drop_row",
        type=int,
        default=-1,
        help="If >=0, drop this specific top-k column (set index to -1 for testing).",
    )
    parser.add_argument(
        "--enable-dynamic-tokens",
        action="store_true",
        help="Whether to enable dynamic tokens for testing",
    )
    args = parser.parse_args()

    if os.getenv("DEEPEp_BAREMETAL") == "1":
        # 裸机双机模式: MASTER_ADDR / RANK / WORLD_SIZE 已由启动脚本注入, 跳过 K8s 主节点发现
        pass
    else:
        # K8s / Volcano 集群模式: 通过 ConfigMap 发现主节点 IP
        from test_internode_a2_set_env import set_environment

        set_environment()

    num_processes = args.num_processes
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    assert num_nodes == A3_NUM_NODES, (
        f"A3 test expects WORLD_SIZE={A3_NUM_NODES} (nodes), got WORLD_SIZE={num_nodes}."
    )
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
