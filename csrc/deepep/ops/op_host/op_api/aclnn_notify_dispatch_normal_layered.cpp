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
#ifdef __DAV_C310__
    // A5: tiling 声明 AlltoAll + AIV_ENGINE，通信控制面在 AIV 核上，服务端为 MTE。
    const NnopbaseHcclServerType serverType = NNOPBASE_HCCL_SERVER_TYPE_MTE;
#else
    // A3: tiling 用 ops2 normal dispatch/combine 在 910_93 上的写法（opType 8 +
    // AlltoAll、不设 AIV_ENGINE），MC2 域由 AICPU/KFC 服务端驱动。kernel 侧用
    // Hccl<AICPU>::InitV2/SetCcTilingV2 绑定并通过 GetWindowsInAddr 取对端窗口。
    const NnopbaseHcclServerType serverType = NNOPBASE_HCCL_SERVER_TYPE_AICPU;
#endif
    std::fprintf(stderr, "[ln-host] aclnn notify exec soc=%d hook=%d server=%d\n", static_cast<int>(soc),
                 NnopbaseSetHcclServerType != nullptr, static_cast<int>(serverType));
    if (NnopbaseSetHcclServerType) {
        // kernel 通过 GetHcclContext 取 MC2 op param：两种芯片都是 HcclOpResParam。
        NnopbaseSetHcclServerType(executor, serverType);
    }
    std::fprintf(stderr, "[ln-host] tp 4 pre-inner ws=%llu exec=%llx\n",
                 static_cast<unsigned long long>(workspaceSize),
                 static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(executor)));
    aclnnStatus ret = aclnnInnerNotifyDispatchNormalLayered(workspace, workspaceSize, executor, stream);
    std::fprintf(stderr, "[ln-host] tp 5 post-inner ret=%d\n", static_cast<int>(ret));
    return ret;
}
