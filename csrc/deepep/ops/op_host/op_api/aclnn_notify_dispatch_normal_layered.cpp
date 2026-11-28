#include "aclnn_notify_dispatch_normal_layered.h"
#include "aclnnInner_notify_dispatch_normal_layered.h"
#include "aclnn/opdev/platform.h"
#include <cstdint>
#include <cstdio>

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType type);

extern "C" aclnnStatus aclnnNotifyDispatchNormalLayeredGetWorkspaceSize(
    const aclTensor *roceContext, const aclTensor *topkIdx, char *commGroup, int32_t rankSize, int32_t rankId,
    int32_t ranksPerServer, int32_t moeExpertNum, int32_t slotsPerServer, int64_t epoch, int32_t hiddenSize,
    int64_t maxBs, int64_t expertAlignment, const aclTensor *recvTokensPerExpert, const aclTensor *layeredPlan,
    const aclTensor *expertTotalCount, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    std::fprintf(stderr, "[ln-host] aclnn notify ws enter\n");
    aclnnStatus ret = aclnnInnerNotifyDispatchNormalLayeredGetWorkspaceSize(
        roceContext, topkIdx, commGroup, rankSize, rankId, ranksPerServer, moeExpertNum, slotsPerServer, epoch,
        hiddenSize, maxBs, expertAlignment, recvTokensPerExpert, layeredPlan, expertTotalCount, workspaceSize,
        executor);
    std::fprintf(stderr, "[ln-host] aclnn notify ws ret=%d size=%llu\n", static_cast<int>(ret),
                 static_cast<unsigned long long>(ret == 0 ? (workspaceSize ? *workspaceSize : 0U) : 0U));
    return ret;
}

extern "C" aclnnStatus aclnnNotifyDispatchNormalLayered(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream)
{
    const auto soc = op::GetCurrentPlatformInfo().GetSocVersion();
    std::fprintf(stderr, "[ln-host] aclnn notify exec soc=%d hook=%d\n", static_cast<int>(soc),
                 NnopbaseSetHcclServerType != nullptr);
    if (NnopbaseSetHcclServerType) {
        // tiling 用 AlltoAll + SetCommEngine(AIV_ENGINE)，kernel 通过
        // GetHcclContext 直接读 QP 表，属于 MTE 服务端。与 moe_distribute_dispatch_v2
        // 的 SetCommArgs 一致：只有 910B(A2) 走 AICPU，其余（910_93/950）都走 MTE。
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    std::fprintf(stderr, "[ln-host] tp 4 pre-inner ws=%llu exec=%llx\n",
                 static_cast<unsigned long long>(workspaceSize),
                 static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(executor)));
    aclnnStatus ret = aclnnInnerNotifyDispatchNormalLayered(workspace, workspaceSize, executor, stream);
    std::fprintf(stderr, "[ln-host] tp 5 post-inner ret=%d\n", static_cast<int>(ret));
    return ret;
}
