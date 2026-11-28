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
    // tiling 声明 AlltoAll + SetCommEngine(AIV_ENGINE)，与 910_93 上跑通的
    // notify_dispatch / ops2 normal dispatch 一致：通信控制面在 AIV 核上（不拉起
    // AICPU），服务端类型为 MTE。只有 910B(A2) 的 BatchWrite 路径才用 AICPU。
    const NnopbaseHcclServerType serverType = NNOPBASE_HCCL_SERVER_TYPE_MTE;
    std::fprintf(stderr, "[ln-host] aclnn notify exec soc=%d hook=%d server=%d\n", static_cast<int>(soc),
                 NnopbaseSetHcclServerType != nullptr, static_cast<int>(serverType));
    if (NnopbaseSetHcclServerType) {
        // kernel 通过 GetHcclContext 取 MC2 op param：A3 是 HcclOpResParam，
        // A5 是 HcclCombinOpParam。
        NnopbaseSetHcclServerType(executor, serverType);
    }
    std::fprintf(stderr, "[ln-host] tp 4 pre-inner ws=%llu exec=%llx\n",
                 static_cast<unsigned long long>(workspaceSize),
                 static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(executor)));
    aclnnStatus ret = aclnnInnerNotifyDispatchNormalLayered(workspace, workspaceSize, executor, stream);
    std::fprintf(stderr, "[ln-host] tp 5 post-inner ret=%d\n", static_cast<int>(ret));
    return ret;
}
