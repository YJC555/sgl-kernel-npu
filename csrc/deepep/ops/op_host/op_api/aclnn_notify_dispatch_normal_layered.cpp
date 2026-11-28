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
    const NnopbaseHcclServerType serverType = NNOPBASE_HCCL_SERVER_TYPE_MTE;
#else
    // tiling 声明 BatchWrite(opType 18)，这是 AICPU/KFC 域的写法：HCCL 在
    // HcclAllocComResourceBytiling 中按该声明分配资源，MTE 服务端会返回
    // HCCL_E_PARA(1)。arch22 的 910_93 路径同样设置 AICPU。
    const NnopbaseHcclServerType serverType = NNOPBASE_HCCL_SERVER_TYPE_AICPU;
#endif
    std::fprintf(stderr, "[ln-host] aclnn notify exec soc=%d hook=%d server=%d\n", static_cast<int>(soc),
                 NnopbaseSetHcclServerType != nullptr, static_cast<int>(serverType));
    if (NnopbaseSetHcclServerType) {
        // kernel 通过 GetHcclContext 取 MC2 op param：A3 是
        // HcclA2CombineOpParam（AICPU），A5 是 HcclCombinOpParam（MTE/AIV）。
        NnopbaseSetHcclServerType(executor, serverType);
    }
    std::fprintf(stderr, "[ln-host] tp 4 pre-inner ws=%llu exec=%llx\n",
                 static_cast<unsigned long long>(workspaceSize),
                 static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(executor)));
    aclnnStatus ret = aclnnInnerNotifyDispatchNormalLayered(workspace, workspaceSize, executor, stream);
    std::fprintf(stderr, "[ln-host] tp 5 post-inner ret=%d\n", static_cast<int>(ret));
    return ret;
}
