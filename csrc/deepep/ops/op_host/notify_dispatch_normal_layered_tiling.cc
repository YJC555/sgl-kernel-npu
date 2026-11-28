#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include "error_log.h"
#include "mc2_tiling_utils.h"
#include "register/op_def_registry.h"
#include "tiling/hccl/hccl_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "cam_moe_layered_host_validation.h"
#include "../op_kernel/notify_dispatch_normal_layered_tiling.h"
#include "cam_moe_layered_plan.h"
using namespace ge;
namespace {
constexpr uint32_t GROUP = 0, WORLD = 1, RANK = 2, RANKS = 3, EXPERTS = 4, SLOTS = 5, EPOCH = 6, HIDDEN = 7, MAX_BS = 8,
                   EXPERT_ALIGNMENT = 9;
constexpr uint32_t MAX_WORLD = 384, MAX_SERVERS = 32, MAX_TOPK = 16;
constexpr uint32_t MAX_PREFIX_EXPERTS = 1024;
constexpr uint64_t WORKSPACE_BYTES = 16ULL * 1024ULL * 1024ULL;
// MC2 (HCCL server type AICPU) needs the framework's system area at the head of
// the workspace, the same reservation arch22 makes in SetA2RuntimeConfig. The
// counter area of this op is appended right after it.
constexpr uint64_t SYSTEM_NEED_WORKSPACE = 16ULL * 1024ULL * 1024ULL;
constexpr uint64_t SHARED_WORK_UB_BYTES = 120ULL * 1024ULL;
#ifdef __DAV_C310__
constexpr uint64_t RDMA_UB_BYTES = 32ULL;
constexpr uint64_t HCOMM_UB_BYTES = 512ULL;
#else
constexpr uint64_t RDMA_UB_BYTES = 32ULL;
constexpr uint64_t HCOMM_UB_BYTES = 0ULL;
#endif
constexpr uint64_t REQUIRED_NOTIFY_UB_BYTES = RDMA_UB_BYTES + HCOMM_UB_BYTES + SHARED_WORK_UB_BYTES;
int64_t Attr(const gert::TilingContext *c, uint32_t i)
{
    auto *a = c->GetAttrs();
    auto *p = a ? a->GetAttrPointer<int64_t>(i) : nullptr;
    return p ? *p : -1;
}
bool Shape2D(const gert::StorageShape *s, int64_t &a, int64_t &b)
{
    if (!s || s->GetStorageShape().GetDimNum() != 2) return false;
    a = s->GetStorageShape().GetDim(0);
    b = s->GetStorageShape().GetDim(1);
    return a >= 0 && b > 0;
}
bool FitsUint32(uint64_t value)
{
    return value <= std::numeric_limits<uint32_t>::max();
}
bool MulFits(uint64_t a, uint64_t b, uint64_t &result)
{
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    result = a * b;
    return true;
}
bool AddFits(uint64_t a, uint64_t b, uint64_t &result)
{
    if (a > std::numeric_limits<uint64_t>::max() - b) return false;
    result = a + b;
    return true;
}
bool AlignElements(uint64_t value, uint64_t &result)
{
    constexpr uint64_t elementsPerIsolation = CamMoeLayered::CACHE_LINE_ISOLATION_BYTES / sizeof(int32_t);
    uint64_t rounded = 0;
    return AddFits(value, elementsPerIsolation - 1U, rounded) &&
           (result = (rounded / elementsPerIsolation) * elementsPerIsolation, true);
}
bool Align32(uint64_t value, uint64_t &result)
{
    uint64_t rounded = 0;
    return AddFits(value, 31U, rounded) && (result = (rounded / 32U) * 32U, true);
}
bool ValidPlanLayout(const CamMoeLayered::PlanLayout &p, uint64_t packetSlots, uint64_t planElements)
{
    return p.packetCount <= p.dispatchMemberCount && p.dispatchMemberCount <= p.packetSlot &&
           p.packetSlot <= p.expertCount && p.expertCount <= p.localExpertOffset &&
           p.localExpertOffset <= p.localExpertCount && p.localExpertCount <= p.packetDesc &&
           p.packetDesc <= p.memberDesc && p.memberDesc <= p.sourceReturnBase && p.reverseMember <= p.total &&
           static_cast<uint64_t>(p.packetSlot) + packetSlots <= planElements;
}
// Returns 0 only when both mc2 tiling blobs were produced. arch22 checks these
// return values; an unchecked failure here leaves a half-filled tiling behind
// and surfaces much later as an opaque HCCL resource-allocation error.
int SetComm(NotifyDispatchNormalLayeredTilingData *d, const std::string &g, bool crossServer)
{
    int init = 0;
    int cc = 0;
#ifdef __DAV_C310__
    (void)crossServer;
    // A5 (950): HCCL exposes the AIV communication engine, so the op keeps the
    // AlltoAll declaration and never needs an AI-RMA QP table.
    AscendC::Mc2CcTilingConfig c(g, 8U, "AlltoAll=level0:fullmesh;level1:pairwise");
    c.SetCommEngine(mc2tiling::AIV_ENGINE);
    init = static_cast<int>(c.GetTiling(d->mc2InitTiling));
    cc = static_cast<int>(c.GetTiling(d->mc2CcTiling));
#else
    // A3 (910_93): HCCL only publishes the AI-RMA QP table (aiRMAInfo) for a
    // single-sided batch write. Layered links just the same-slot card on the
    // peer server, so the inter-server level must be hierarchy.
    const std::string algConfig = crossServer ? "BatchWrite=level1:hierarchy" : "BatchWrite=level0:fullmesh";
    AscendC::Mc2CcTilingConfig c(g, 18U, algConfig);
    init = static_cast<int>(c.GetTiling(d->mc2InitTiling));
    cc = static_cast<int>(c.GetTiling(d->mc2CcTiling));
#endif
    std::fprintf(stderr, "[ln-host] mc2 tiling init=%d cc=%d\n", init, cc);
    return (init != 0 || cc != 0) ? -1 : 0;
}
}  // namespace
namespace optiling {
static ge::graphStatus NotifyTiling(gert::TilingContext *c)
{
    const char *n = c->GetNodeName();
    auto *d = c->GetTilingData<NotifyDispatchNormalLayeredTilingData>();
    auto *roceDesc = c->GetInputDesc(0);
    auto *s = c->GetInputShape(1);
    auto *desc = c->GetInputDesc(1);
    int64_t bs = 0, k = 0;
    const int64_t world = Attr(c, WORLD), rank = Attr(c, RANK), ranks = Attr(c, RANKS), experts = Attr(c, EXPERTS),
                  slots = Attr(c, SLOTS), epoch = Attr(c, EPOCH), hidden = Attr(c, HIDDEN), maxBs = Attr(c, MAX_BS),
                  expertAlignment = Attr(c, EXPERT_ALIGNMENT);
    auto *g = c->GetAttrs() ? c->GetAttrs()->GetAttrPointer<char>(GROUP) : nullptr;
    OP_TILING_CHECK(!d || !roceDesc || !desc || !Shape2D(s, bs, k) || roceDesc->GetDataType() != DT_INT32 ||
                        desc->GetDataType() != DT_INT32 || !g || world <= 0 || world > MAX_WORLD || rank < 0 ||
                        rank >= world || ranks <= 0 || world % ranks || world / ranks > MAX_SERVERS || experts <= 0 ||
                        experts > MAX_PREFIX_EXPERTS || experts % world || experts % (world / ranks) || slots <= 0 ||
                        maxBs <= 0 || slots != maxBs || bs <= 0 || bs > maxBs || k <= 0 || k > MAX_TOPK ||
                        hidden <= 0 || expertAlignment <= 0 ||
                        static_cast<uint64_t>(experts) > std::numeric_limits<uint32_t>::max() ||
                        static_cast<uint64_t>(slots) > std::numeric_limits<uint32_t>::max() ||
                        static_cast<uint64_t>(hidden) > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE(n, "invalid normal layered notify inputs or topology"), return GRAPH_FAILED);
    const uint64_t worldU = static_cast<uint64_t>(world), ranksU = static_cast<uint64_t>(ranks),
                   expertsU = static_cast<uint64_t>(experts), bsU = static_cast<uint64_t>(bs),
                   maxBsU = static_cast<uint64_t>(maxBs), slotsU = static_cast<uint64_t>(slots);
    const uint64_t serversU = worldU / ranksU, roundsU = 1U;
    const uint64_t prefixBytes = CamMoeLayered::AlignUp64(expertsU * sizeof(int32_t)) +
                                 2U * CamMoeLayered::AlignUp64(serversU * sizeof(int32_t));
    OP_TILING_CHECK(prefixBytes > SHARED_WORK_UB_BYTES / 2U,
                    OP_LOGE(n, "normal layered notify prefix table exceeds UB capacity"), return GRAPH_FAILED);
    uint64_t packetSlots = 0, roundsServers = 0, roundsServersLocalExperts = 0;
    uint64_t countSlicePadded = 0, packetSlicePadded = 0, memberCountSlicePadded = 0;
    uint64_t counterSlices = 0, counterElements = 0, roundMaxScratchElements = 0, workspaceElements = 0,
             counterBytes = 0;
    const bool arithmeticValid =
        FitsUint32(maxBsU) && MulFits(maxBsU, serversU, packetSlots) && MulFits(roundsU, serversU, roundsServers) &&
        MulFits(roundsServers, expertsU / serversU, roundsServersLocalExperts) && FitsUint32(serversU) &&
        FitsUint32(roundsU) && FitsUint32(packetSlots) && FitsUint32(roundsServers) &&
        FitsUint32(roundsServersLocalExperts) && AlignElements(roundsServersLocalExperts, countSlicePadded) &&
        AlignElements(roundsServers, packetSlicePadded) && AlignElements(roundsServers, memberCountSlicePadded) &&
        FitsUint32(countSlicePadded) && FitsUint32(packetSlicePadded) && FitsUint32(memberCountSlicePadded) &&
        AddFits(countSlicePadded, packetSlicePadded, counterSlices) &&
        AddFits(counterSlices, memberCountSlicePadded, counterSlices);
    OP_TILING_CHECK(!arithmeticValid, OP_LOGE(n, "normal layered tiling size overflows"), return GRAPH_FAILED);
    uint64_t hiddenBytes = 0, routeBytes = 0, recordBytes = 0, recordStride = 0;
    uint64_t partialBytes = 0, partialStride = 0, serverPairs = 0, dispatchBatchStride = 0, dispatchWindowBytes = 0;
    uint64_t packetMetaBytes = 0, tokenOffset = 0, tokenBytes = 0, memberOffset = 0, memberBytes = 0;
    uint64_t routeOffset = 0;
    uint64_t partialLanes = 0, localPartialWindowBytes = 0, returnLanes = 0, returnPartialWindowBytes = 0;
    uint64_t rankPairs = 0, stateLaneStride = 0, stateLanes = 0, stateWindowBytes = 0;
    const bool windowArithmetic =
        MulFits(static_cast<uint64_t>(hidden), 2U, hiddenBytes) &&
        MulFits(static_cast<uint64_t>(k), sizeof(CamMoeLayered::DispatchTopkRouteRecord), routeBytes) &&
        Align32(hiddenBytes, routeOffset) && AddFits(routeOffset, routeBytes, recordBytes) &&
        Align32(recordBytes, recordStride) && MulFits(static_cast<uint64_t>(hidden), sizeof(float), partialBytes) &&
        AddFits(sizeof(CamMoeLayered::PartialRecordHeader), partialBytes, partialBytes) &&
        Align32(partialBytes, partialStride) && MulFits(serversU, serversU, serverPairs) &&
        MulFits(slotsU, sizeof(CamMoeLayered::DispatchPacketMeta), packetMetaBytes) &&
        AddFits(sizeof(CamMoeLayered::DispatchBatchMeta), packetMetaBytes, tokenOffset) &&
        Align32(tokenOffset, tokenOffset) && MulFits(slotsU, recordStride, tokenBytes) &&
        AddFits(tokenOffset, tokenBytes, memberOffset) && Align32(memberOffset, memberOffset) &&
        MulFits(slotsU, static_cast<uint64_t>(k) * sizeof(CamMoeLayered::DispatchMemberMeta), memberBytes) &&
        AddFits(memberOffset, memberBytes, dispatchBatchStride) && Align32(dispatchBatchStride, dispatchBatchStride) &&
        MulFits(serversU, 2U, dispatchWindowBytes) &&
        MulFits(dispatchWindowBytes, dispatchBatchStride, dispatchWindowBytes) && MulFits(ranksU, ranksU, rankPairs) &&
        MulFits(rankPairs, serverPairs, partialLanes) && MulFits(partialLanes, slotsU, partialLanes) &&
        MulFits(partialLanes, partialStride, localPartialWindowBytes) && MulFits(ranksU, serverPairs, returnLanes) &&
        MulFits(returnLanes, slotsU, returnLanes) && MulFits(returnLanes, partialStride, returnPartialWindowBytes) &&
        AddFits(16ULL + CamMoeLayered::DispatchMaxChunks(static_cast<uint32_t>(slotsU),
                                                         static_cast<uint32_t>(recordStride)),
                rankPairs, stateLaneStride) &&
        MulFits(serverPairs, roundsU, stateLanes) && MulFits(stateLanes, stateLaneStride, stateLanes) &&
        MulFits(stateLanes, sizeof(uint64_t), stateWindowBytes) && FitsUint32(recordStride) &&
        FitsUint32(dispatchBatchStride) && FitsUint32(partialStride);
    OP_TILING_CHECK(!windowArithmetic, OP_LOGE(n, "normal layered Window stride or size overflows"),
                    return GRAPH_FAILED);
    uint64_t planPacketCount = CamMoeLayered::HeaderElements(), planDispatchMemberCount = 0, planPacketSlot = 0;
    uint64_t planExpertCount = 0, planLocalExpertOffset = 0, planLocalExpertCount = 0;
    uint64_t planPacketDesc = 0, planMemberDesc = 0, planSourceReturnBase = 0, planReverseCount = 0;
    uint64_t planReverseReturnBase = 0, planReversePacket = 0, planReverseMember = 0;
    uint64_t planPacketDescElements = 0, planMemberElements = 0, planReversePacketElements = 0;
    uint64_t planReverseMemberElements = 0, planIngressElements = 0, planTotal = 0, localExpertsU = expertsU / worldU;
    const bool planArithmetic =
        AddFits(planPacketCount, roundsServers, planDispatchMemberCount) &&
        AddFits(planDispatchMemberCount, roundsServers, planPacketSlot) &&
        AddFits(planPacketSlot, packetSlots, planExpertCount) &&
        AddFits(planExpertCount, roundsServersLocalExperts, planLocalExpertOffset) &&
        AddFits(planLocalExpertOffset, localExpertsU, planLocalExpertCount) &&
        AddFits(planLocalExpertCount, localExpertsU, planPacketDesc) &&
        MulFits(roundsServers, slotsU, planPacketDescElements) &&
        MulFits(planPacketDescElements, CamMoeLayered::PacketElements(), planPacketDescElements) &&
        AddFits(planPacketDesc, planPacketDescElements, planMemberDesc) &&
        MulFits(roundsServers, slotsU, planMemberElements) &&
        MulFits(planMemberElements, static_cast<uint64_t>(k), planMemberElements) &&
        MulFits(planMemberElements, CamMoeLayered::MemberElements(), planMemberElements) &&
        AddFits(planMemberDesc, planMemberElements, planSourceReturnBase) &&
        MulFits(serversU, 13U, planIngressElements) && AddFits(planIngressElements, 2U, planIngressElements) &&
        AddFits(planSourceReturnBase, planIngressElements, planReverseCount) &&
        AddFits(planReverseCount, serversU, planReverseReturnBase) &&
        AddFits(planReverseReturnBase, serversU, planReversePacket) &&
        MulFits(maxBsU, worldU, planReversePacketElements) &&
        MulFits(planReversePacketElements, CamMoeLayered::ReversePacketElements(), planReversePacketElements) &&
        AddFits(planReversePacket, planReversePacketElements, planReverseMember) &&
        MulFits(maxBsU, worldU, planReverseMemberElements) &&
        MulFits(planReverseMemberElements, k, planReverseMemberElements) &&
        MulFits(planReverseMemberElements, CamMoeLayered::ReverseMemberElements(), planReverseMemberElements) &&
        AddFits(planReverseMember, planReverseMemberElements, planTotal) && FitsUint32(planTotal);
    OP_TILING_CHECK(!planArithmetic, OP_LOGE(n, "normal layered plan layout overflows"), return GRAPH_FAILED);
    const uint32_t servers = static_cast<uint32_t>(serversU), rounds = static_cast<uint32_t>(roundsU);
    auto &l = d->layered;
    l.epWorldSize = world;
    l.epRankId = rank;
    l.ranksPerServer = ranks;
    l.serverNum = servers;
    l.rankInServer = rank % ranks;
    l.moeExpertNum = experts;
    l.bs = bs;
    l.realMaxBs = static_cast<uint32_t>(maxBs);
    l.maxBs = static_cast<uint32_t>(maxBs);
    l.k = k;
    l.h = hidden;
    l.validRounds = 1U;
    l.physicalRounds = rounds;
    l.slotsPerServer = slots;
    l.expertAlignment = static_cast<uint32_t>(expertAlignment);
    l.epoch = epoch < 0 ? 0 : epoch;
    l.elementBytes = 2;
    l.isHybridDeployment = Mc2TilingUtils::IsHybridDeployment() ? 1U : 0U;
    OP_TILING_CHECK(
        !CamMoeLayeredHost::ValidWindowInputArithmetic(l.moeExpertNum, l.epWorldSize, l.ranksPerServer, servers, rounds,
                                                       slots, k, recordStride, partialStride),
        OP_LOGE(n, "normal layered window arithmetic overflows"), return GRAPH_FAILED);
    const uint64_t controlStride = CamMoeLayered::AlignUp64(
        CamMoeLayered::AllRoundCountPayloadBytes(rounds, static_cast<uint32_t>(expertsU / serversU)),
        CamMoeLayered::CACHE_LINE_ISOLATION_BYTES);
    OP_TILING_CHECK(!FitsUint32(controlStride), OP_LOGE(n, "normal layered control stride overflows"),
                    return GRAPH_FAILED);
    l.recordStride = static_cast<uint32_t>(recordStride);
    l.partialStride = static_cast<uint32_t>(partialStride);
    l.controlStride = static_cast<uint32_t>(controlStride);
    auto w = CamMoeLayered::MakeWindowLayout(l.moeExpertNum, l.epWorldSize, l.ranksPerServer, servers, rounds, slots,
                                             l.recordStride, l.partialStride, l.k, l.isHybridDeployment != 0U);
    l.totalWinSize = Mc2TilingUtils::GetMaxWindowSize();
    OP_TILING_CHECK(!CamMoeLayeredHost::ValidWindowLayout(w, worldU, serversU, ranksU, roundsU) ||
                        w.dispatchPayloadCapacity != dispatchWindowBytes ||
                        w.localPartialBytes != localPartialWindowBytes ||
                        w.returnPartialBytes != returnPartialWindowBytes || w.stateLaneStride != stateLaneStride ||
                        w.requiredBytes < w.stateOffset || w.requiredBytes - w.stateOffset < stateWindowBytes ||
                        !CamMoeLayeredHost::ValidWindowCapacity(w, Mc2TilingUtils::GetMaxWindowSize()),
                    OP_LOGE(n,
                            "normal layered window layout or HCCL buffer capacity is invalid "
                            "(requiredBytes=%lu configuredBytes=%lu slots=%lu recordStride=%u partialStride=%u)",
                            w.requiredBytes, Mc2TilingUtils::GetMaxWindowSize(),
                            static_cast<unsigned long>(slots), l.recordStride, l.partialStride),
                    return GRAPH_FAILED);
    l.activeWindowBytes = w.requiredBytes;
    l.windowRequiredBytes = w.requiredBytes;
    const uint64_t maxProxyRank = (serversU - 1U) * ranksU + (ranksU - 1U);
    OP_TILING_CHECK(maxProxyRank >= worldU, OP_LOGE(n, "normal layered proxy rank is invalid"), return GRAPH_FAILED);
    auto p = CamMoeLayered::MakePlanLayout(l.bs, l.maxBs, l.k, l.moeExpertNum, l.epWorldSize, servers, l.ranksPerServer,
                                           rounds, slots);
    const uint64_t planElements = p.total;
    OP_TILING_CHECK(!ValidPlanLayout(p, packetSlots, planElements) || planElements != planTotal ||
                        p.packetCount != planPacketCount || p.dispatchMemberCount != planDispatchMemberCount ||
                        p.packetSlot != planPacketSlot || p.expertCount != planExpertCount ||
                        p.localExpertOffset != planLocalExpertOffset || p.localExpertCount != planLocalExpertCount ||
                        p.packetDesc != planPacketDesc || p.memberDesc != planMemberDesc ||
                        p.sourceReturnBase != planSourceReturnBase ||
                        p.outboundBase != planSourceReturnBase + serversU ||
                        p.outboundBytes != planSourceReturnBase + 3U * serversU ||
                        p.outboundTotalBytes != planSourceReturnBase + 5U * serversU ||
                        p.sendIngressBase != planSourceReturnBase + 5U * serversU + 2U ||
                        p.sendIngressBytes != planSourceReturnBase + 7U * serversU + 2U ||
                        p.recvIngressBase != planSourceReturnBase + 9U * serversU + 2U ||
                        p.recvIngressBytes != planSourceReturnBase + 11U * serversU + 2U ||
                        p.reverseCount != planReverseCount || p.reverseReturnBase != planReverseReturnBase ||
                        p.reversePacket != planReversePacket || p.reverseMember != planReverseMember,
                    OP_LOGE(n, "normal layered plan layout is invalid"), return GRAPH_FAILED);
    l.planElements = static_cast<uint32_t>(planElements);
    l.packetCountOffset = p.packetCount;
    l.dispatchMemberCountOffset = p.dispatchMemberCount;
    l.packetSlotOffset = p.packetSlot;
    l.expertCountOffset = p.expertCount;
    l.localExpertOffset = p.localExpertOffset;
    l.localExpertCountOffset = p.localExpertCount;
    l.packetDescOffset = p.packetDesc;
    l.memberDescOffset = p.memberDesc;
    auto platform = platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    l.aivNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, l.totalUbSize);
    auto *ws = c->GetWorkspaceSizes(1);
    const bool workspaceArithmetic =
        MulFits(static_cast<uint64_t>(l.aivNum), counterSlices, counterElements) &&
        MulFits(static_cast<uint64_t>(l.aivNum), CamMoeLayered::CACHE_LINE_ISOLATION_BYTES / sizeof(int32_t),
                roundMaxScratchElements) &&
        AddFits(counterElements, roundMaxScratchElements, workspaceElements) &&
        MulFits(workspaceElements, sizeof(int32_t), counterBytes);
    OP_TILING_CHECK(l.aivNum == 0U || !ws || !workspaceArithmetic || counterBytes > WORKSPACE_BYTES ||
                        l.totalUbSize < REQUIRED_NOTIFY_UB_BYTES,
                    OP_LOGE(n, "normal layered workspace or UB budget is invalid"), return GRAPH_FAILED);
    ws[0] = SYSTEM_NEED_WORKSPACE + CamMoeLayered::AlignUp64(counterBytes);
    const int commTiling = SetComm(d, std::string(g), servers > 1U);
    uint32_t aicpuBlockDim = 0U;
#ifndef __DAV_C310__
    // A3 keeps the communication control plane on AICPU. Cross-server needs the
    // full AICPU service block set; a single server only needs one block.
    aicpuBlockDim = servers > 1U ? mc2tiling::AICPU_NUM_BLOCKS_A2 : 1U;
    c->SetAicpuBlockDim(aicpuBlockDim);
#endif
    c->SetBlockDim(l.aivNum);
    c->SetScheduleMode(1);
    c->SetTilingKey(23U);
    std::fprintf(stderr,
                 "[ln-host] tiling world=%llu ranks=%llu servers=%llu aiv=%u aicpu=%u ws=%llu cross=%d comm=%d\n",
                 static_cast<unsigned long long>(worldU), static_cast<unsigned long long>(ranksU),
                 static_cast<unsigned long long>(serversU), l.aivNum, aicpuBlockDim,
                 static_cast<unsigned long long>(ws[0]), servers > 1U ? 1 : 0, commTiling);
    OP_TILING_CHECK(commTiling != 0, OP_LOGE(n, "mc2 cc tiling generation failed"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}
struct NotifyCompileInfo {};
ge::graphStatus ParseNotify(gert::TilingParseContext *)
{
    return GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(NotifyDispatchNormalLayered).Tiling(NotifyTiling).TilingParse<NotifyCompileInfo>(ParseNotify);
}  // namespace optiling
