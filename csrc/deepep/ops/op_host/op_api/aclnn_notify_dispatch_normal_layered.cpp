#include "aclnn_notify_dispatch_normal_layered.h"
#include "aclnnInner_notify_dispatch_normal_layered.h"
#include "aclnn/opdev/platform.h"
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
    std::fprintf(stderr, "[ln-host] aclnn notify ws ret=%d\n", static_cast<int>(ret));
    return ret;
}

extern "C" aclnnStatus aclnnNotifyDispatchNormalLayered(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream)
{
    const auto soc = op::GetCurrentPlatformInfo().GetSocVersion();
    const bool isA3 = soc == op::SocVersion::ASCEND910_93;
    const int type = isA3 ? static_cast<int>(NNOPBASE_HCCL_SERVER_TYPE_AICPU)
                          : static_cast<int>(NNOPBASE_HCCL_SERVER_TYPE_MTE);
    std::fprintf(stderr, "[ln-host] aclnn notify exec soc=%d type=%d hook=%d\n", static_cast<int>(soc), type,
                 NnopbaseSetHcclServerType != nullptr);
    if (NnopbaseSetHcclServerType) {
        // A3 的通信控制面在 AICPU（tiling 用 BatchWrite + SetAicpuBlockDim，
        // kernel 用 Hccl<HCCL_SERVER_TYPE_AICPU>::InitV2/SetCcTilingV2）；
        // A5 走 AIV 引擎，对应 MTE。类型必须与 tiling/kernel 一致，
        // 否则 HCCL 分配通信资源失败（ret=1）。
        NnopbaseSetHcclServerType(executor, isA3 ? NNOPBASE_HCCL_SERVER_TYPE_AICPU
                                                 : NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerNotifyDispatchNormalLayered(workspace, workspaceSize, executor, stream);
    std::fprintf(stderr, "[ln-host] aclnn notify exec ret=%d\n", static_cast<int>(ret));
    return ret;
}
