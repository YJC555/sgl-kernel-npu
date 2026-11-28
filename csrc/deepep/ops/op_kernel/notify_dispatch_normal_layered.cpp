#if DEEPEP_CANN_VERSION_NUM >= DEEPEP_LAYERED_MIN_CANN_VERSION

#include <cstddef>
#include "kernel_operator.h"
#include "notify_dispatch_normal_layered_tiling.h"
#include "cam_moe_layered_plan.h"
#include "cam_moe_layered_comm.h"

using namespace AscendC;

namespace {
using namespace CamMoeLayered;

#ifdef __DAV_C310__
constexpr uint32_t kHcommUbBytes = HCOMM_UB_BUF_SIZE;
#else
constexpr uint32_t kHcommUbBytes = 0U;
#endif

constexpr uint32_t kInitTileBytes = 60U * 1024U;
constexpr uint32_t kInitTileElements = kInitTileBytes / sizeof(int32_t);
constexpr uint32_t kReduceTileBytes = 60U * 1024U;
constexpr uint32_t kReduceTileElements = kReduceTileBytes / sizeof(int32_t);
constexpr uint32_t kAggregateCopyTileBytes = 60U * 1024U;
constexpr uint32_t kSharedWorkUbBytes = 2U * kReduceTileBytes;
constexpr uint32_t kMaxPrefixExperts = 1024U;
constexpr uint32_t kMaxPrefixServers = 32U;
constexpr uint32_t kExpertPrefixBytes = kMaxPrefixExperts * sizeof(int32_t);
constexpr uint32_t kServerPrefixBytes = kMaxPrefixServers * sizeof(int32_t);
constexpr uint32_t kDataCopyThreshold = 128U;
constexpr uint32_t kMaxRowsScratchStride = CACHE_LINE_ISOLATION_BYTES / sizeof(int32_t);
constexpr uint32_t kUbCopyAlignmentElements = 32U / sizeof(int32_t);

static_assert(kInitTileBytes <= kReduceTileBytes);
static_assert(kAggregateCopyTileBytes <= kReduceTileBytes);
static_assert(kReduceTileBytes % 32U == 0U);
static_assert(kExpertPrefixBytes + 2U * kServerPrefixBytes <= kReduceTileBytes);
static_assert(CACHE_LINE_ISOLATION_BYTES % AscendC::CACHE_LINE_SIZE == 0U,
              "Aggregate owner slices must isolate Ascend C cache lines");

// Debug tracing. The kernel is AIV-only and every stage is separated by a
// cross-core sync, so one compact marker per stage printed from AIV 0 is
// enough to name the faulting function: the last marker printed is the stage
// that faulted, and a marker with no following stage means that stage hung.
// Inside a stage, TracePoint(id, detail) marks single statements. Its id is
// "stage * 100 + step", so grep "tp 402" in this file lands on the exact line,
// and it prints the calling AIV (only the first kDebugAivCount cores print, see
// below), so a hang in a per-core loop also names the core and the loop index
// that never came back. Keep the added points few and terse: kernel printf has
// a small per-core buffer and the last line printed is what identifies the
// fault on a trap.
// Compile with -DDEEPEP_LAYERED_DEBUG_LOG=0 to drop every print.
#ifndef DEEPEP_LAYERED_DEBUG_LOG
#define DEEPEP_LAYERED_DEBUG_LOG 1
#endif

// Only the first DEEPEP_LAYERED_DEBUG_AIVS AIVs emit TracePoints. The
// per-server work is owned by AIV 0..serverNum-1, so the default keeps exactly
// the cores that drive RDMA and drops the per-slice repeats that otherwise
// evict the earliest lines (boot/init/stage) from the small per-core buffer.
// Build with -DDEEPEP_LAYERED_DEBUG_AIVS=<aivNum> to trace every core again.
#ifndef DEEPEP_LAYERED_DEBUG_AIVS
#define DEEPEP_LAYERED_DEBUG_AIVS 2U
#endif
constexpr uint32_t kDebugAivCount = DEEPEP_LAYERED_DEBUG_AIVS;

#if DEEPEP_LAYERED_DEBUG_LOG
#define LAYERED_LOG(...) printf(__VA_ARGS__)
#else
#define LAYERED_LOG(...) \
    do {                 \
    } while (0)
#endif

class NotifyDispatchNormalLayered
{
public:
    __aicore__ inline NotifyDispatchNormalLayered() {}

    // Print the stage id from AIV 0, drain every pipeline of this core and then
    // converge all cores. Called before each stage, so the marker printed last
    // identifies the stage that faulted (507035) or hung.
    __aicore__ inline void StageBoundary(uint32_t stage)
    {
#if DEEPEP_LAYERED_DEBUG_LOG
        PipeBarrier<PIPE_ALL>();
        if (GetBlockIdx() == 0U) {
            LAYERED_LOG("[ln] stage %u\n", stage);
        }
        PipeBarrier<PIPE_ALL>();
#else
        (void)stage;
#endif
        SyncAll<true>();
    }

    // Fine-grained marker for statements *inside* a stage. Unlike
    // StageBoundary it does not synchronise cores, so it can be dropped
    // anywhere, and it prints from the first kDebugAivCount AIVs that reach it:
    // the last line in the device log then names the exact statement that
    // faulted (507035) or hung, plus the core and loop index when `detail`
    // carries one.
    // The id encodes "stage * 100 + step": grep "tp 402" finds the call site.
    __aicore__ inline void TracePoint(uint32_t point, uint32_t detail)
    {
#if DEEPEP_LAYERED_DEBUG_LOG
        if (static_cast<uint32_t>(GetBlockIdx()) >= kDebugAivCount) {
            return;
        }
        // Drain this core's pipelines so the marker is ordered after the
        // memory traffic that precedes it and before the one that follows.
        PipeBarrier<PIPE_ALL>();
        LAYERED_LOG("[ln] tp %u aiv %u v %u\n", point, static_cast<uint32_t>(GetBlockIdx()), detail);
        PipeBarrier<PIPE_ALL>();
#else
        (void)point;
        (void)detail;
#endif
    }

    __aicore__ inline void Init(GM_ADDR roceContext, GM_ADDR topkIdx, GM_ADDR layeredPlan, GM_ADDR recvTokensPerExpert,
                                GM_ADDR expertTotalCount, GM_ADDR workspace, const CamMoeLayeredInfo &info,
                                LocalTensor<uint64_t> &doorbell, LocalTensor<uint32_t> &head,
                                LocalTensor<int32_t> &initUb, LocalTensor<int32_t> &reduceSrcUb,
                                LocalTensor<int32_t> &reduceAccUb, LocalTensor<uint8_t> &aggregateCopyUb,
                                LocalTensor<uint8_t> &hcommUb, LocalTensor<int32_t> &packetPrefixUb,
                                LocalTensor<int32_t> &memberPrefixUb, LocalTensor<int32_t> &expertPrefixUb)
    {
        topk_ = (__gm__ int32_t *)topkIdx;
        plan_ = (__gm__ int32_t *)layeredPlan;
        recv_ = (__gm__ int32_t *)recvTokensPerExpert;
        total_ = (__gm__ int32_t *)expertTotalCount;
        workspace_ = (__gm__ int32_t *)workspace;
        info_ = info;
        doorbell_ = &doorbell;
        head_ = &head;
        initUb_ = &initUb;
        reduceSrcUb_ = &reduceSrcUb;
        reduceAccUb_ = &reduceAccUb;
        aggregateCopyUb_ = &aggregateCopyUb;
        packetPrefixUb_ = &packetPrefixUb;
        memberPrefixUb_ = &memberPrefixUb;
        expertPrefixUb_ = &expertPrefixUb;
        TracePoint(101U, info.epRankId);
        localExperts_ = info_.moeExpertNum / info_.epWorldSize;
        expertsPerServer_ = info_.moeExpertNum / info_.serverNum;
        sourceServer_ = info_.epRankId / info_.ranksPerServer;
        comm_.Init(info_.epRankId, roceContext, hcommUb, info_.isHybridDeployment != 0U);
        // 101 -> 102 brackets CommAdapter::Init, i.e. the GetHcclContext /
        // aiRMAInfo binding that the 910_93 path now relies on.
        TracePoint(102U, info.epRankId);
        layout_ = CamMoeLayered::MakeWindowLayout(
            info_.moeExpertNum, info_.epWorldSize, info_.ranksPerServer, info_.serverNum, info_.physicalRounds,
            info_.slotsPerServer, info_.recordStride, info_.partialStride, info_.k, info_.isHybridDeployment != 0U);
        countSlice_ = info_.physicalRounds * info_.serverNum * expertsPerServer_;
        packetSlice_ = info_.physicalRounds * info_.serverNum;
        countSlicePadded_ = AlignWorkspaceElements(countSlice_);
        packetSlicePadded_ = AlignWorkspaceElements(packetSlice_);
        memberCountSlicePadded_ = packetSlicePadded_;
#if DEEPEP_LAYERED_DEBUG_LOG
        if (GetBlockIdx() == 0U) {
            // Split into short lines: kernel printf has a per-call argument
            // limit and a small per-core log buffer, so keep each call compact.
            LAYERED_LOG("[ln] init rank=%u/%u inSrv=%u srv=%u rps=%u ex=%u\n", info_.epRankId, info_.epWorldSize,
                        info_.rankInServer, info_.serverNum, info_.ranksPerServer, info_.moeExpertNum);
            LAYERED_LOG("[ln] init aiv=%u bs=%u maxBs=%u slots=%u k=%u h=%u\n", info_.aivNum, info_.bs, info_.maxBs,
                        info_.slotsPerServer, info_.k, info_.h);
            // Low 32 bits of every pointer are enough to compare against
            // tensor.data_ptr() on the host and to spot swapped outputs.
            LAYERED_LOG("[ln] init gm roce=%x topk=%x recv=%x plan=%x\n",
                        static_cast<uint32_t>(reinterpret_cast<uint64_t>(roceContext)),
                        static_cast<uint32_t>(reinterpret_cast<uint64_t>(topkIdx)),
                        static_cast<uint32_t>(reinterpret_cast<uint64_t>(recv_)),
                        static_cast<uint32_t>(reinterpret_cast<uint64_t>(plan_)));
            LAYERED_LOG("[ln] init gm total=%x ws=%x planElems=%u rnd=%u\n",
                        static_cast<uint32_t>(reinterpret_cast<uint64_t>(total_)),
                        static_cast<uint32_t>(reinterpret_cast<uint64_t>(workspace_)), info_.planElements,
                        info_.physicalRounds);
            LAYERED_LOG("[ln] init off cnt=%u mcnt=%u slot=%u exp=%u leoff=%u lecnt=%u\n", info_.packetCountOffset,
                        info_.dispatchMemberCountOffset, info_.packetSlotOffset, info_.expertCountOffset,
                        info_.localExpertOffset, info_.localExpertCountOffset);
            // HCCL context / AiRMA binding, printed first because every later
            // snapshot dereferences the same context: ctx=0 means
            // GetHcclContext returned nothing, dim=0 with ctx!=0 means HCCL
            // filled a different context struct, and infoSz=0 means the AiRMA
            // section was never built for this domain.
            uint64_t ctxRaw = 0U;
            uint32_t ctxDim = 0U;
            uint64_t infoSize = 0U;
            uint64_t infoRaw = 0U;
            comm_.LoadContextDesc(ctxRaw, ctxDim, infoSize, infoRaw);
            LAYERED_LOG("[ln] ctx raw=%x dim=%u infoSz=%u info=%x\n", static_cast<uint32_t>(ctxRaw), ctxDim,
                        static_cast<uint32_t>(infoSize), static_cast<uint32_t>(infoRaw));
            // Peer window snapshot: a zero windowsIn for a cross-node rank is
            // exactly what turns WindowIn() into the null destination that
            // stage 6 currently writes to.
            const uint32_t rps = info_.ranksPerServer;
            uint32_t sameRank = info_.epRankId;
            uint32_t peerRank = info_.epRankId;
            uint64_t samePtr = 0U;
            uint64_t sameWin = 0U;
            uint64_t peerPtr = 0U;
            uint64_t peerWin = 0U;
            if (rps != 0U && info_.serverNum != 0U) {
                sameRank = (info_.epRankId / rps) * rps + (info_.epRankId + 1U) % rps;
                peerRank = ((info_.epRankId / rps + 1U) % info_.serverNum) * rps + info_.rankInServer;
            }
            comm_.LoadPeerWindow(sameRank, samePtr, sameWin);
            comm_.LoadPeerWindow(peerRank, peerPtr, peerWin);
            LAYERED_LOG("[ln] ctx sz=%u winMB=%u localIn=%x\n", comm_.RankSize(),
                        static_cast<uint32_t>(comm_.WinSizeBytes() >> 20U),
                        static_cast<uint32_t>(reinterpret_cast<uint64_t>(comm_.WindowIn(info_.epRankId))));
            LAYERED_LOG("[ln] ctx same r=%u ptr=%x win=%x\n", sameRank, static_cast<uint32_t>(samePtr),
                        static_cast<uint32_t>(sameWin));
            LAYERED_LOG("[ln] ctx peer r=%u ptr=%x win=%x\n", peerRank, static_cast<uint32_t>(peerPtr),
                        static_cast<uint32_t>(peerWin));
            // QP table bound by CommAdapter::Init; 0 means HCCL has not published
            // aiRMAInfo for this domain yet (PostRdmaSend would fault on it).
            uint64_t qpInfo = 0U;
            uint32_t qpNum = 0U;
            uint32_t qpSize = 0U;
            uint64_t qpAddr = 0U;
            comm_.LoadQpDesc(info_.epRankId, qpInfo, qpNum, qpSize, qpAddr);
            LAYERED_LOG("[ln] ctx qptab=%x qpnum=%u\n", static_cast<uint32_t>(qpInfo), qpNum);
        }
#endif
        TracePoint(103U, info.epRankId);
    }

    __aicore__ inline void Process()
    {
        StageBoundary(1);
        Cleanup();  // 清零 plan 普通元数据与各 AIV 私有计数片
        StageBoundary(2);
        InitPlanHeader();  // 初始化 plan 信息和状态（AIV 0）
        StageBoundary(3);
        BuildLocalRoutePlanParallel();  // 根据 topk_idx 构建本地路由关系
        StageBoundary(4);
        ReduceLocalCounts();  // 合并多 AIV 计数
        StageBoundary(5);
        PlanOutboundPayloads();  // 记录本地token按照server重排的偏移和大小
        StageBoundary(6);
        PublishCountControls();  // server间发送给同号卡
        StageBoundary(7);
        AggregateServerCounts();  // 本地rank重排并转发给server内其他rank
        StageBoundary(8);
        ApplyProxyAggregatesToOwner();  // 等待数据到达，本地按照专家重排，记录recv_x
        StageBoundary(9);
        FanoutBaseAckSlicesToProxies();  // 将计算好的行号发送给本地proxy
        StageBoundary(10);
        WaitAllBaseAckSlices();  // 等待所有的本地rank收齐
        StageBoundary(11);
        AssembleBaseAcksAtProxies();  // inbuffer搬到outbuffer
        StageBoundary(12);
        PublishBaseAcksFromProxies();  // 直接rdma发送，不需要重排
        StageBoundary(13);
        ApplyAllReturnedRowBases();  // 等待数据收齐，并回填每个member的行号
        StageBoundary(14);           // 整个接收计划已完成
    }

private:
    __aicore__ inline uint64_t CountSlice(uint32_t aiv) const
    {
        return (uint64_t)aiv * countSlicePadded_;
    }

    __aicore__ inline uint64_t PacketSlice(uint32_t aiv) const
    {
        return (uint64_t)info_.aivNum * countSlicePadded_ + (uint64_t)aiv * packetSlicePadded_;
    }

    __aicore__ inline uint64_t MemberCountSlice(uint32_t aiv) const
    {
        return (uint64_t)info_.aivNum * countSlicePadded_ + (uint64_t)info_.aivNum * packetSlicePadded_ +
               (uint64_t)aiv * memberCountSlicePadded_;
    }

    __aicore__ inline uint64_t MaxRowsScratch(uint32_t aiv) const
    {
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        const uint64_t counters =
            static_cast<uint64_t>(aivNum) * (countSlicePadded_ + packetSlicePadded_ + memberCountSlicePadded_);
        return counters + static_cast<uint64_t>(aiv) * kMaxRowsScratchStride;
    }

    __aicore__ inline uint32_t AlignWorkspaceElements(uint32_t elements) const
    {
        constexpr uint32_t elementsPerBlock = CACHE_LINE_ISOLATION_BYTES / sizeof(int32_t);
        return (elements + elementsPerBlock - 1U) / elementsPerBlock * elementsPerBlock;
    }

    __aicore__ inline void ZeroWorkspaceRange(__gm__ int32_t *dst, uint32_t elements)
    {
        if (elements == 0U) {
            return;
        }
        Duplicate(*initUb_, 0, kInitTileElements);
        for (uint32_t offset = 0; offset < elements; offset += kInitTileElements) {
            const uint32_t count = min(kInitTileElements, elements - offset);
            GlobalTensor<int32_t> dstTensor;
            dstTensor.SetGlobalBuffer(dst + offset);
            DataCopy(dstTensor, *initUb_, count);
        }
        SyncFunc<HardEvent::MTE3_S>();
    }

    __aicore__ inline void FillPlanRange(uint32_t begin, uint32_t end, int32_t value)
    {
        for (uint32_t offset = begin; offset < end;) {
            const uint32_t count = min(kInitTileElements, end - offset);
            Duplicate(*initUb_, value, count);
            GlobalTensor<int32_t> dst;
            dst.SetGlobalBuffer(plan_ + offset);
            DataCopyPad(dst, *initUb_,
                        DataCopyExtParams{1U, static_cast<uint32_t>(count * sizeof(int32_t)), 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE3_S>();
            offset += count;
        }
    }

    __aicore__ inline void CopyPlanToUb(uint32_t planOffset, LocalTensor<int32_t> &ub, uint32_t bytes)
    {
        GlobalTensor<int32_t> src;
        src.SetGlobalBuffer(plan_ + planOffset);
        DataCopyPad(ub, src, DataCopyExtParams{1U, bytes, 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_S>();
    }

    __aicore__ inline void CopyUbToPlan(uint32_t planOffset, LocalTensor<int32_t> &ub, uint32_t bytes,
                                        uint32_t ubOffset = 0U)
    {
        SyncFunc<HardEvent::S_MTE3>();
        GlobalTensor<int32_t> dst;
        dst.SetGlobalBuffer(plan_ + planOffset);
        DataCopyPad(dst, ub[ubOffset], DataCopyExtParams{1U, bytes, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }

    __aicore__ inline void StorePlanValue(uint32_t planOffset, int32_t value)
    {
        initUb_->SetValue(0U, value);
        CopyUbToPlan(planOffset, *initUb_, sizeof(int32_t));
    }

    __aicore__ inline void StoreInt32(__gm__ int32_t *dst, int32_t value)
    {
        initUb_->SetValue(0U, value);
        SyncFunc<HardEvent::S_MTE3>();
        GlobalTensor<int32_t> destination;
        destination.SetGlobalBuffer(dst);
        DataCopyPad(destination, *initUb_, DataCopyExtParams{1U, sizeof(int32_t), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }

    __aicore__ inline void Cleanup()
    {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        const uint64_t packetSlotElements = static_cast<uint64_t>(info_.maxBs) * info_.serverNum;
        const uint32_t packetSlotBegin = info_.packetSlotOffset;
        const uint32_t packetSlotEnd = packetSlotBegin + static_cast<uint32_t>(packetSlotElements);
        const uint32_t dataBegin = HeaderElements();
        const uint32_t dataEnd = info_.planElements;
        const uint32_t dataSize = dataEnd - dataBegin;
        const uint32_t begin = dataBegin + static_cast<uint32_t>(static_cast<uint64_t>(dataSize) * aiv / aivNum);
        const uint32_t end = dataBegin + static_cast<uint32_t>(static_cast<uint64_t>(dataSize) * (aiv + 1U) / aivNum);
        TracePoint(201U, begin);

        // Clear all ordinary plan metadata except packetSlot, whose sentinel
        // value is INVALID_SLOT rather than zero.
        if (begin < packetSlotBegin) {
            FillPlanRange(begin, min(end, packetSlotBegin), 0);
        }
        if (end > packetSlotEnd) {
            const uint32_t zeroBegin = begin > packetSlotEnd ? begin : packetSlotEnd;
            FillPlanRange(zeroBegin, end, 0);
        }
        TracePoint(202U, end);

        // Initialize each AIV's private reduction slices before route building.
        ZeroWorkspaceRange(workspace_ + CountSlice(aiv), countSlicePadded_);
        ZeroWorkspaceRange(workspace_ + PacketSlice(aiv), packetSlicePadded_);
        ZeroWorkspaceRange(workspace_ + MemberCountSlice(aiv), memberCountSlicePadded_);
        cacheReadInvalidate(reinterpret_cast<__gm__ uint8_t *>(workspace_ + CountSlice(aiv)),
                            static_cast<uint64_t>(countSlicePadded_) * sizeof(int32_t));
        cacheReadInvalidate(reinterpret_cast<__gm__ uint8_t *>(workspace_ + PacketSlice(aiv)),
                            static_cast<uint64_t>(packetSlicePadded_) * sizeof(int32_t));
        cacheReadInvalidate(reinterpret_cast<__gm__ uint8_t *>(workspace_ + MemberCountSlice(aiv)),
                            static_cast<uint64_t>(memberCountSlicePadded_) * sizeof(int32_t));

        // packetSlot is initialized independently because zero is a valid slot.
        const uint32_t packetBegin = static_cast<uint32_t>(packetSlotElements * aiv / aivNum);
        const uint32_t packetEnd = static_cast<uint32_t>(packetSlotElements * (aiv + 1U) / aivNum);
        FillPlanRange(packetSlotBegin + packetBegin, packetSlotBegin + packetEnd, INVALID_SLOT);
        TracePoint(203U, packetEnd);
    }

    __aicore__ inline void ReduceWorkspaceRange(uint32_t dstOffset, uint32_t srcOffset, uint32_t elements,
                                                uint64_t sourceBase, uint32_t sourceStride)
    {
        if (elements == 0U) {
            return;
        }
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        if (elements < kDataCopyThreshold) {  // 小于阈值直接对GM进行读写
            for (uint32_t i = 0; i < elements; ++i) {
                int32_t value = 0;
                for (uint32_t p = 0; p < aivNum; ++p) {
                    value += workspace_[sourceBase + static_cast<uint64_t>(p) * sourceStride + srcOffset + i];
                }
                reduceAccUb_->SetValue(i, value);
            }
            CopyUbToPlan(dstOffset, *reduceAccUb_, elements * sizeof(int32_t));
            return;
        }

        for (uint32_t offset = 0; offset < elements; offset += kReduceTileElements) {
            const uint32_t count = min(kReduceTileElements, elements - offset);
            Duplicate(*reduceAccUb_, 0, count);
            for (uint32_t p = 0; p < aivNum; ++p) {
                GlobalTensor<int32_t> srcTensor;
                srcTensor.SetGlobalBuffer(workspace_ + sourceBase + static_cast<uint64_t>(p) * sourceStride +
                                          srcOffset + offset);
                DataCopyExtParams copyParams{1U, static_cast<uint32_t>(count * sizeof(int32_t)), 0U, 0U, 0U};
                DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0};
                DataCopyPad(*reduceSrcUb_, srcTensor, copyParams, padParams);
                SyncFunc<HardEvent::MTE2_V>();
                Add(*reduceAccUb_, *reduceAccUb_, *reduceSrcUb_, count);
                SyncFunc<HardEvent::V_MTE2>();
            }
            SyncFunc<HardEvent::V_MTE3>();
            GlobalTensor<int32_t> dstTensor;
            dstTensor.SetGlobalBuffer(plan_ + dstOffset + offset);
            DataCopyExtParams copyParams{1U, static_cast<uint32_t>(count * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPad(dstTensor, *reduceAccUb_, copyParams);
            SyncFunc<HardEvent::MTE3_V>();
        }
    }

    __aicore__ inline void BuildPrefix(LocalTensor<int32_t> &prefixUb, uint32_t elements, uint64_t sourceBase,
                                       uint32_t sourceStride, uint32_t aiv)
    {
        Duplicate(prefixUb, 0, elements);
        for (uint32_t p = 0U; p < aiv; ++p) {
            GlobalTensor<int32_t> source;
            source.SetGlobalBuffer(workspace_ + sourceBase + static_cast<uint64_t>(p) * sourceStride);
            DataCopyPad(*reduceSrcUb_, source,
                        DataCopyExtParams{1U, static_cast<uint32_t>(elements * sizeof(int32_t)), 0U, 0U, 0U},
                        DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE2_V>();
            Add(prefixUb, prefixUb, *reduceSrcUb_, elements);
            SyncFunc<HardEvent::V_MTE2>();
        }
    }

    __aicore__ inline void InitPlanHeader()
    {
        if (GetBlockIdx() != 0U) {
            return;
        }

        auto *h = (__gm__ PlanHeader *)plan_;
        h->magic = PLAN_MAGIC;
        h->version = PLAN_VERSION;
        h->epochLo = (uint32_t)info_.epoch;
        h->epochHi = (uint32_t)(info_.epoch >> 32U);
        h->bs = info_.bs;
        h->maxBs = info_.maxBs;
        h->validRounds = info_.validRounds;
        h->hiddenSize = info_.h;
        h->topk = info_.k;
        h->expertNum = info_.moeExpertNum;
        h->worldSize = info_.epWorldSize;
        h->serverNum = info_.serverNum;
        h->ranksPerServer = info_.ranksPerServer;
        h->rankId = info_.epRankId;
        h->rankInServer = info_.rankInServer;
        h->physicalRounds = info_.physicalRounds;
        h->slotsPerServer = info_.slotsPerServer;
        h->recordStrideBytes = info_.recordStride;
        h->packetCountOffset = info_.packetCountOffset;
        h->dispatchMemberCountOffset = info_.dispatchMemberCountOffset;
        h->packetSlotOffset = info_.packetSlotOffset;
        h->expertCountOffset = info_.expertCountOffset;
        h->localExpertOffset = info_.localExpertOffset;
        h->localExpertCountOffset = info_.localExpertCountOffset;
        h->packetDescOffset = info_.packetDescOffset;
        h->memberDescOffset = info_.memberDescOffset;
        h->totalElements = info_.planElements;
        h->globalCombineRows = 0U;
        h->reserved[0] = 0U;
        h->reserved[1] = 0U;
        h->reserved[2] = 0U;
        h->reserved[3] = 0U;
        StoreInt32(total_, 0);
        cacheWriteThrough(reinterpret_cast<__gm__ uint8_t *>(h), sizeof(PlanHeader));
        TracePoint(301U, static_cast<uint32_t>(info_.planElements));
    }

    __aicore__ inline void BuildLocalRoutePlanParallel()
    {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t n = info_.aivNum == 0U ? 1U : info_.aivNum;
        const uint32_t begin = info_.bs * aiv / n;
        const uint32_t end = info_.bs * (aiv + 1U) / n;

        auto *counts = workspace_ + CountSlice(aiv);    // 保存发送给每个expert member的数量（本AIV内）
        auto *packets = workspace_ + PacketSlice(aiv);  // 保存发送给每个server packet的数量（本AIV内）
        auto *memberCounts = workspace_ + MemberCountSlice(aiv);  // 保存发送到每个server member数量（本AIV内）
        const uint32_t descScratch = 0U;
        const uint32_t descStride = CamMoeLayered::AlignUp(PacketElements(), kUbCopyAlignmentElements);
        const uint32_t memberScratch = info_.serverNum * descStride;
        const uint32_t memberScratchStride = CamMoeLayered::AlignUp(MemberElements(), kUbCopyAlignmentElements);
        const uint32_t packetSlotScratch = memberScratch + memberScratchStride;
#if DEEPEP_LAYERED_DEBUG_LOG
        if (aiv == 0U) {
            LAYERED_LOG("[ln] route aiv=%u tok=%u..%u dstride=%u mscratch=%u sscratch=%u\n", aiv, begin, end,
                        descStride, memberScratch, packetSlotScratch);
        }
#endif
        TracePoint(401U, begin);
        for (uint32_t token = begin; token < end; ++token) {
            if ((token & 63U) == 0U) {
                TracePoint(402U, token);
            }
            for (uint32_t target = 0U; target < info_.serverNum; ++target) {
                const uint32_t desc = descScratch + target * descStride;
                initUb_->SetValue(desc, static_cast<int32_t>(token));
                initUb_->SetValue(desc + 1U, 0);                              // 发给该server的member偏移
                initUb_->SetValue(desc + 2U, 0);                              // 发给该server的member数量
                initUb_->SetValue(desc + 3U, 0);                              // 发给该server的packet偏移
                initUb_->SetValue(packetSlotScratch + target, INVALID_SLOT);  // 记录要给哪些server发送数据
            }
            // 读取 top-k expert
            for (uint32_t topkSlot = 0; topkSlot < info_.k; ++topkSlot) {
                const int32_t expert = topk_[(uint64_t)token * info_.k + topkSlot];
                if (expert == INVALID_SLOT) {
                    continue;
                }
                const uint32_t expertId = static_cast<uint32_t>(expert);
                // topk_idx is device input. An id outside [0, moeExpertNum) is
                // not a valid expert: using it would index past the per-server
                // UB descriptor scratch (packetSlotScratch + target) and past
                // rows of the per-expert counter slice in GM.
                if (expertId >= info_.moeExpertNum) {
                    continue;
                }
                const uint32_t target = expertId / expertsPerServer_;  // 目标server
                // 每个 token/server 只占一个 packet。
                const uint32_t packet = target * info_.slotsPerServer + token;
                const uint32_t desc = descScratch + target * descStride;
                const uint32_t member = static_cast<uint32_t>(initUb_->GetValue(desc + 2U));
                if (member == 0U) {
                    initUb_->SetValue(desc + 1U, static_cast<int32_t>(memberCounts[target]));
                    initUb_->SetValue(desc + 3U, static_cast<int32_t>(packets[target]++));
                }
                initUb_->SetValue(desc + 2U, static_cast<int32_t>(member + 1U));
                memberCounts[target]++;
                const uint32_t mb = info_.memberDescOffset + (packet * info_.k + member) * MemberElements();
                const uint32_t ordinal = counts[expertId]++;
                initUb_->SetValue(memberScratch, expert);                              // 专家id
                initUb_->SetValue(memberScratch + 1U, static_cast<int32_t>(ordinal));  // 发给该expert member的偏移
                initUb_->SetValue(memberScratch + 2U, static_cast<int32_t>(topkSlot));  // topk id
                CopyUbToPlan(mb, *initUb_, sizeof(MemberDesc), memberScratch);
                initUb_->SetValue(packetSlotScratch + target, static_cast<int32_t>(token));
            }
            for (uint32_t target = 0U; target < info_.serverNum; ++target) {
                const uint32_t desc = descScratch + target * descStride;
                if (initUb_->GetValue(desc + 2U) > 0) {
                    const uint32_t packet = target * info_.slotsPerServer + token;
                    CopyUbToPlan(info_.packetDescOffset + packet * PacketElements(), *initUb_, sizeof(SourcePacketDesc),
                                 desc);
                }
            }
            CopyUbToPlan(info_.packetSlotOffset + token * info_.serverNum, *initUb_, info_.serverNum * sizeof(int32_t),
                         packetSlotScratch);
        }
        TracePoint(403U, end);
        cacheWriteThrough(reinterpret_cast<__gm__ uint8_t *>(counts),
                          static_cast<uint64_t>(countSlicePadded_) * sizeof(int32_t));
        cacheWriteThrough(reinterpret_cast<__gm__ uint8_t *>(packets),
                          static_cast<uint64_t>(packetSlicePadded_) * sizeof(int32_t));
        cacheWriteThrough(reinterpret_cast<__gm__ uint8_t *>(memberCounts),
                          static_cast<uint64_t>(memberCountSlicePadded_) * sizeof(int32_t));
        TracePoint(404U, 0U);
    }

    __aicore__ inline void ReduceLocalCounts()
    {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t n = info_.aivNum == 0U ? 1U : info_.aivNum;
        const uint32_t packetElements = info_.serverNum;
        const uint32_t expertElements = packetElements * expertsPerServer_;
        const uint32_t packetBegin = static_cast<uint32_t>(static_cast<uint64_t>(packetElements) * aiv / n);
        const uint32_t packetEnd = static_cast<uint32_t>(static_cast<uint64_t>(packetElements) * (aiv + 1U) / n);
        const uint32_t expertBegin = static_cast<uint32_t>(static_cast<uint64_t>(expertElements) * aiv / n);
        const uint32_t expertEnd = static_cast<uint32_t>(static_cast<uint64_t>(expertElements) * (aiv + 1U) / n);
        const uint32_t memberBegin = static_cast<uint32_t>(static_cast<uint64_t>(packetElements) * aiv / n);
        const uint32_t memberEnd = static_cast<uint32_t>(static_cast<uint64_t>(packetElements) * (aiv + 1U) / n);
        const uint64_t counterBytes = static_cast<uint64_t>(n) *
                                      (countSlicePadded_ + packetSlicePadded_ + memberCountSlicePadded_) *
                                      sizeof(int32_t);
        cacheReadInvalidate(reinterpret_cast<__gm__ uint8_t *>(workspace_), counterBytes);
        cacheReadInvalidate(reinterpret_cast<__gm__ uint8_t *>(plan_ + info_.packetDescOffset),
                            static_cast<uint64_t>(info_.serverNum) * info_.slotsPerServer * sizeof(SourcePacketDesc));
        cacheReadInvalidate(
            reinterpret_cast<__gm__ uint8_t *>(plan_ + info_.memberDescOffset),
            static_cast<uint64_t>(info_.serverNum) * info_.slotsPerServer * info_.k * sizeof(MemberDesc));
        // 本 rank 发给各目标 server 的 packet 总数
        ReduceWorkspaceRange(info_.packetCountOffset + packetBegin, packetBegin, packetEnd - packetBegin,
                             static_cast<uint64_t>(n) * countSlicePadded_, packetSlicePadded_);
        // 本 rank 发给各目标 server 的 member 总数
        ReduceWorkspaceRange(
            info_.dispatchMemberCountOffset + memberBegin, memberBegin, memberEnd - memberBegin,
            static_cast<uint64_t>(n) * countSlicePadded_ + static_cast<uint64_t>(n) * packetSlicePadded_,
            memberCountSlicePadded_);
        // 本 rank 发给各目标 expert 的 member 总数
        ReduceWorkspaceRange(info_.expertCountOffset + expertBegin, expertBegin, expertEnd - expertBegin, 0U,
                             countSlicePadded_);
        TracePoint(501U, expertEnd);
        const uint32_t begin = info_.bs * aiv / n;
        const uint32_t end = info_.bs * (aiv + 1U) / n;
        if (begin == end) {
            return;
        }
        // 在ub中提前计算前缀和
        BuildPrefix(*packetPrefixUb_, packetElements, static_cast<uint64_t>(n) * countSlicePadded_, packetSlicePadded_,
                    aiv);
        BuildPrefix(*memberPrefixUb_, packetElements,
                    static_cast<uint64_t>(n) * (countSlicePadded_ + packetSlicePadded_), memberCountSlicePadded_, aiv);
        BuildPrefix(*expertPrefixUb_, expertElements, 0U, countSlicePadded_, aiv);
        SyncFunc<HardEvent::V_S>();
        TracePoint(502U, 0U);
#if DEEPEP_LAYERED_DEBUG_LOG
        if (aiv == 0U) {
            LAYERED_LOG("[ln] reduce pkt=%u exp=%u tok=%u..%u aiv=%u cslice=%u\n", packetElements, expertElements,
                        begin, end, n, countSlicePadded_);
        }
#endif

        for (uint32_t token = begin; token < end; ++token) {
            if ((token & 63U) == 0U) {
                TracePoint(503U, token);
            }
            for (uint32_t target = 0; target < info_.serverNum; ++target) {
                const uint32_t packet = target * info_.slotsPerServer + token;
                CopyPlanToUb(info_.packetDescOffset + packet * PacketElements(), *initUb_, sizeof(SourcePacketDesc));
                const uint32_t memberCount = static_cast<uint32_t>(initUb_->GetValue(2U));
                if (memberCount == 0) {
                    continue;
                }
                // One token can contribute at most k members to a packet. A
                // larger value means the descriptor does not belong to this
                // invocation; copying that many bytes would overrun initUb_ and
                // then fault on a scalar UB read inside the member loop.
                if (memberCount > info_.k) {
                    continue;
                }
                initUb_->SetValue(1U, initUb_->GetValue(1U) +
                                          memberPrefixUb_->GetValue(target));  // 本地发送到目标server的第几个member
                initUb_->SetValue(3U, initUb_->GetValue(3U) +
                                          packetPrefixUb_->GetValue(target));  // 本地发送到目标server的第几个packet
                CopyUbToPlan(info_.packetDescOffset + packet * PacketElements(), *initUb_, sizeof(SourcePacketDesc));

                const uint32_t memberOffset = info_.memberDescOffset + packet * info_.k * MemberElements();
                const uint32_t memberBytes = memberCount * sizeof(MemberDesc);
                CopyPlanToUb(memberOffset, *initUb_, memberBytes);
                for (uint32_t m = 0U; m < memberCount; ++m) {
                    const uint32_t memberUbOffset = m * MemberElements();
                    const uint32_t expert = static_cast<uint32_t>(initUb_->GetValue(memberUbOffset));
                    const int32_t localRow = initUb_->GetValue(memberUbOffset + 1U);
                    // expertPrefixUb_ only holds moeExpertNum entries. Clamp a
                    // member descriptor that carries an out-of-range id instead
                    // of reading past the end of the UB prefix table.
                    const uint32_t expertIndex = expert < info_.moeExpertNum ? expert : 0U;
                    initUb_->SetValue(memberUbOffset + 1U, localRow + expertPrefixUb_->GetValue(expertIndex));
                }
                CopyUbToPlan(memberOffset, *initUb_, memberBytes);
            }
        }
        TracePoint(504U, 0U);
    }

    __aicore__ inline void PlanOutboundPayloads()
    {
        const auto p = MakePlanLayout(info_.bs, info_.maxBs, info_.k, info_.moeExpertNum, info_.epWorldSize,
                                      info_.serverNum, info_.ranksPerServer, 1U, info_.maxBs);
        if (GetBlockIdx() != 0U) {
            return;
        }
#if DEEPEP_LAYERED_DEBUG_LOG
        LAYERED_LOG("[ln] outbound srv=%u stride=%u\n", info_.serverNum, info_.recordStride);
#endif
        cacheReadInvalidate(reinterpret_cast<__gm__ uint8_t *>(plan_ + info_.packetCountOffset),
                            static_cast<uint64_t>(info_.serverNum) * sizeof(int32_t));
        cacheReadInvalidate(reinterpret_cast<__gm__ uint8_t *>(plan_ + info_.dispatchMemberCountOffset),
                            static_cast<uint64_t>(info_.serverNum) * sizeof(int32_t));
        for (uint32_t target = 0U; target < info_.serverNum; ++target) {
            uint64_t base = 0U;
            for (uint32_t previous = 0U; previous < target; ++previous) {  // 计算发给每个server的偏移
                base += DispatchBatchPayloadBytes(
                    static_cast<uint32_t>(plan_[info_.packetCountOffset + previous]),
                    static_cast<uint32_t>(plan_[info_.dispatchMemberCountOffset + previous]), info_.recordStride);
            }
            const uint64_t bytes =
                DispatchBatchPayloadBytes(static_cast<uint32_t>(plan_[info_.packetCountOffset + target]),
                                          static_cast<uint32_t>(plan_[info_.dispatchMemberCountOffset + target]),
                                          info_.recordStride);  // 计算发给每个server的长度
            StorePlanValue(p.outboundBase + 2U * target, static_cast<int32_t>(base));
            StorePlanValue(p.outboundBase + 2U * target + 1U,
                           static_cast<int32_t>(base >> 32U));  // 保存给server的发送偏移
            StorePlanValue(p.outboundBytes + 2U * target, static_cast<int32_t>(bytes));
            StorePlanValue(p.outboundBytes + 2U * target + 1U,
                           static_cast<int32_t>(bytes >> 32U));  // 保存给server的发送长度
            if (target + 1U == info_.serverNum) {
                const uint64_t total = base + bytes;
                StorePlanValue(p.outboundTotalBytes, static_cast<int32_t>(total));
                StorePlanValue(p.outboundTotalBytes + 1U, static_cast<int32_t>(total >> 32U));  // 记录总发送量
            }
        }
        TracePoint(601U, info_.serverNum);
    }

    __aicore__ inline bool Wait(__gm__ uint64_t *f, uint64_t expected)
    {
        while (true) {
            cacheReadInvalidate((__gm__ uint8_t *)f, sizeof(uint64_t));
            if (*f == expected) {
                return true;
            }
            GetSystemCycle();
        }
    }

    __aicore__ inline void CopyWindowFlag(__gm__ uint64_t *dst, uint64_t value)
    {
        // WindowIn(peer) is a mapped peer-window address.  Publish the flag
        // through MTE3 instead of issuing a scalar store to that address.
        doorbell_->SetValue(0, value);
        GlobalTensor<uint64_t> dstTensor;
        dstTensor.SetGlobalBuffer(dst);
        DataCopyPad(dstTensor, *doorbell_, DataCopyExtParams{1U, sizeof(uint64_t), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }

    __aicore__ inline void CopyBaseAckControl(__gm__ uint8_t *dst, const BaseAckControlMeta &control)
    {
        initUb_->SetValue(0U, static_cast<int32_t>(control.serverMaxRows));
        initUb_->SetValue(1U, static_cast<int32_t>(control.ingressOffsetLo));
        initUb_->SetValue(2U, static_cast<int32_t>(control.ingressOffsetHi));
        initUb_->SetValue(3U, static_cast<int32_t>(control.payloadBytesLo));
        initUb_->SetValue(4U, static_cast<int32_t>(control.payloadBytesHi));
        initUb_->SetValue(5U, 0);
        initUb_->SetValue(6U, 0);
        initUb_->SetValue(7U, 0);
        SyncFunc<HardEvent::S_MTE3>();
        GlobalTensor<int32_t> dstTensor;
        dstTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dst));
        DataCopyPad(dstTensor, *initUb_, DataCopyExtParams{1U, sizeof(BaseAckControlMeta), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }

    __aicore__ inline void CopyWindowBytes(__gm__ uint8_t *src, __gm__ uint8_t *dst, uint64_t bytes)
    {
        for (uint64_t offset = 0; offset < bytes;) {
            const uint32_t count =
                static_cast<uint32_t>(min(static_cast<uint64_t>(kAggregateCopyTileBytes), bytes - offset));
            GlobalTensor<uint8_t> srcTensor;
            srcTensor.SetGlobalBuffer(src + offset);
            const DataCopyExtParams copyParams{1U, count, 0U, 0U, 0U};
            const DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
            DataCopyPad(*aggregateCopyUb_, srcTensor, copyParams, padParams);
            SyncFunc<HardEvent::MTE2_V>();
            GlobalTensor<uint8_t> dstTensor;
            dstTensor.SetGlobalBuffer(dst + offset);
            DataCopyPad(dstTensor, *aggregateCopyUb_, copyParams);
            SyncFunc<HardEvent::MTE3_S>();
            offset += count;
        }
    }
    __aicore__ inline void PublishCountControls()
    {
        // 当前 rank 的发送窗口
        auto *out = (__gm__ uint8_t *)comm_.WindowOut(info_.epRankId);
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        cacheReadInvalidate(reinterpret_cast<__gm__ uint8_t *>(plan_ + info_.packetCountOffset),
                            static_cast<uint64_t>(info_.serverNum) * sizeof(int32_t));
        cacheReadInvalidate(reinterpret_cast<__gm__ uint8_t *>(plan_ + info_.expertCountOffset),
                            static_cast<uint64_t>(info_.serverNum) * expertsPerServer_ * sizeof(int32_t));
        // 按照server分核
        for (uint32_t target = aiv; target < info_.serverNum; target += aivNum) {
            // 为该 target server 和 source rank 分配的区域
            const uint64_t off = ControlLaneOffset(layout_, info_.epWorldSize, info_.epRankId, target);
            auto *payload = out + off;
            // 当前 rank 针对当前 targetServer 构造的完整 CountControl 消息地址
            auto *meta = (__gm__ CountControlMeta *)payload;
            for (uint32_t i = 0; i < 7U; ++i) {
                meta->padding[i] = 0U;
            }
            uint32_t returnBase = 0U;
            for (uint32_t preceding = 0U; preceding < target; ++preceding) {
                returnBase += static_cast<uint32_t>(plan_[info_.packetCountOffset + preceding]);
            }
            meta->sourceReturnBase = returnBase;  // 保存用于combine返程时的packet的本地偏移
            const auto p = MakePlanLayout(info_.bs, info_.maxBs, info_.k, info_.moeExpertNum, info_.epWorldSize,
                                          info_.serverNum, info_.ranksPerServer, 1U, info_.maxBs);
            StorePlanValue(p.sourceReturnBase + target, static_cast<int32_t>(returnBase));

            auto *entry = payload + CountControlDataOffset();
            ((__gm__ uint32_t *)entry)[0] = static_cast<uint32_t>(
                plan_[info_.packetCountOffset + target]);  // 保存当前rank发给目标server的packet数量
            auto *c = (__gm__ int32_t *)(entry + sizeof(uint32_t));
            for (uint32_t e = 0; e < expertsPerServer_; ++e) {
                c[e] = plan_[info_.expertCountOffset + target * expertsPerServer_ +
                             e];  // 保存当前rank发给目标专家的member数量
            }
            const uint64_t flagOff = ControlFlagOffset(layout_, info_.epWorldSize, info_.epRankId, target);
            auto *flag = (__gm__ uint64_t *)(out + flagOff);
            // flag
            *flag = ComposeFlag(info_.epoch, Phase::CONTROL_COUNT_READY);
            cacheWriteThrough(payload, layout_.controlStride);
            cacheWriteThrough((__gm__ uint8_t *)flag, sizeof(uint64_t));

            // 发送给同号卡
            TracePoint(701U, target);
            const uint32_t proxyRank = target * info_.ranksPerServer + info_.rankInServer;
            auto *in = (__gm__ uint8_t *)comm_.WindowIn(proxyRank);
            // 目的位置
            auto *dst = in + off;
            auto *remoteFlag = (__gm__ uint64_t *)(in + flagOff);
            // Split the payload and flag submissions so a stall in either WQE
            // names itself instead of both collapsing into a missing 702.
            TracePoint(703U, proxyRank);
#if DEEPEP_LAYERED_DEBUG_LOG
            if (static_cast<uint32_t>(GetBlockIdx()) < kDebugAivCount) {
                // Pointer-only view: no dereference, so it survives even when the
                // QP descriptor is bad, and it shows which branch is taken.
                LAYERED_LOG("[ln] ctl self=%u proxy=%u rps=%u inSrv=%u\n", info_.epRankId, proxyRank,
                            info_.ranksPerServer, info_.rankInServer);
                LAYERED_LOG("[ln] ctl dst=%x src=%x bytes=%u\n",
                            static_cast<uint32_t>(reinterpret_cast<uint64_t>(dst)),
                            static_cast<uint32_t>(reinterpret_cast<uint64_t>(payload)), layout_.controlStride);
            }
#endif
            if (proxyRank == info_.epRankId) {
                CopyWindowBytes(payload, dst, layout_.controlStride);
            } else {
#if DEEPEP_LAYERED_DEBUG_LOG
                if (static_cast<uint32_t>(GetBlockIdx()) < kDebugAivCount) {
                    // Zero-dereference descriptor view first: it survives even when
                    // the QP entry itself cannot be read.
                    uint64_t qpInfo = 0U;
                    uint32_t qpNum = 0U;
                    uint32_t qpSize = 0U;
                    uint64_t qpAddr = 0U;
                    comm_.LoadQpDesc(proxyRank, qpInfo, qpNum, qpSize, qpAddr);
                    LAYERED_LOG("[ln] qp dst=%u info=%x num=%u size=%u addr=%x\n", proxyRank,
                                static_cast<uint32_t>(qpInfo), qpNum, qpSize, static_cast<uint32_t>(qpAddr));
                    uint32_t qpHead = 0U;
                    uint32_t qpTail = 0U;
                    uint32_t qpDepth = 0U;
                    comm_.LoadQpState(proxyRank, qpHead, qpTail, qpDepth);
                    LAYERED_LOG("[ln] sq dst=%u head=%u tail=%u depth=%u\n", proxyRank, qpHead, qpTail, qpDepth);
                }
#endif
                comm_.PostRdmaSend(payload, dst, proxyRank, layout_.controlStride, *doorbell_, *head_);
            }
            TracePoint(704U, target);
            if (proxyRank == info_.epRankId) {
                CopyWindowFlag(remoteFlag, *flag);
            } else {
                comm_.PostReadyFlag((GM_ADDR)flag, (GM_ADDR)remoteFlag, proxyRank, *doorbell_, *head_);
            }
        }
        TracePoint(702U, 0U);
    }

    __aicore__ inline void AggregateServerCounts()
    {
        auto *in = (__gm__ uint8_t *)comm_.WindowIn(info_.epRankId);
        auto *out = (__gm__ uint8_t *)comm_.WindowOut(info_.epRankId);
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        const uint32_t proxyInServer = info_.rankInServer;
        const uint32_t ownerBegin = static_cast<uint32_t>(static_cast<uint64_t>(info_.ranksPerServer) * aiv / aivNum);
        const uint32_t ownerEnd =
            static_cast<uint32_t>(static_cast<uint64_t>(info_.ranksPerServer) * (aiv + 1U) / aivNum);
        // 等待所有的server间flag到达
        for (uint32_t sourceServer = aiv; sourceServer < info_.serverNum; sourceServer += aivNum) {
            const uint32_t source = proxyInServer + sourceServer * info_.ranksPerServer;
            // 等待源rank的flag
            const uint64_t flag = ControlFlagOffset(layout_, info_.epWorldSize, source, sourceServer_);
            TracePoint(801U, sourceServer);
            Wait((__gm__ uint64_t *)(in + flag), ComposeFlag(info_.epoch, Phase::CONTROL_COUNT_READY));
            cacheReadInvalidate(in + ControlLaneOffset(layout_, info_.epWorldSize, source, sourceServer_),
                                layout_.controlStride);
        }
        SyncAll<true>();
        TracePoint(802U, 0U);

        // 按照本地rank分核
        for (uint32_t ownerInServer = ownerBegin; ownerInServer < ownerEnd; ++ownerInServer) {
            // [ownerInServer]
            auto *aggregate = out + ProxyAggregateStageOffset(layout_, ownerInServer);
            for (uint32_t sourceServer = 0U; sourceServer < info_.serverNum; ++sourceServer) {
                const uint32_t source = proxyInServer + sourceServer * info_.ranksPerServer;
                auto *payload = in + ControlLaneOffset(layout_, info_.epWorldSize, source, sourceServer_);
                // 记录源rank发给目标专家的member数量
                auto *counts = (__gm__ int32_t *)(payload + CountControlDataOffset() + sizeof(uint32_t));
                for (uint32_t localExpert = 0U; localExpert < localExperts_; ++localExpert) {
                    const uint32_t expert = ownerInServer * localExperts_ + localExpert;
                    const int32_t count = counts[expert];
                    // [sourceServer][localExpert]
                    *(__gm__ int32_t *)(aggregate + ProxyOwnerCountOffset(sourceServer, localExpert, localExperts_)) =
                        count;
                }
            }

            cacheWriteThrough(aggregate, layout_.proxyAggregateStride);

            const uint32_t owner = sourceServer_ * info_.ranksPerServer + ownerInServer;
            auto *ownerWindow = (__gm__ uint8_t *)comm_.WindowIn(owner);  // 本地其他rank的接收缓冲区
            const uint32_t blockBytes = 32U;
            const uint32_t blockCount = static_cast<uint32_t>(layout_.proxyAggregateStride / blockBytes);
            TracePoint(803U, ownerInServer);
            for (uint32_t block = 0; block < blockCount;) {
                const uint32_t blockNum = min(kAggregateCopyTileBytes / blockBytes, blockCount - block);
                const uint32_t byteOffset = block * blockBytes;
                GlobalTensor<uint8_t> srcTensor;
                srcTensor.SetGlobalBuffer(aggregate + byteOffset);
                const DataCopyExtParams copyParams{1U, static_cast<uint32_t>(blockNum * blockBytes), 0U, 0U, 0U};
                const DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
                DataCopyPad(*aggregateCopyUb_, srcTensor, copyParams, padParams);
                SyncFunc<HardEvent::MTE2_V>();
                GlobalTensor<uint8_t> dstTensor;
                // 对端rank窗口中，当前rank的位置
                dstTensor.SetGlobalBuffer(ownerWindow + ProxyAggregateOffset(layout_, proxyInServer) + byteOffset);
                DataCopyPad(dstTensor, *aggregateCopyUb_, copyParams);
                SyncFunc<HardEvent::MTE3_S>();
                block += blockNum;
            }

            // flag
            auto *flag = (__gm__ uint64_t *)(ownerWindow + ProxyAggregateFlagOffset(layout_, proxyInServer));
            CopyWindowFlag(flag, ComposeFlag(info_.epoch, Phase::SERVER_FANOUT_DONE));
        }
        TracePoint(804U, 0U);
    }

    __aicore__ inline void ApplyProxyAggregatesToOwner()
    {
        auto *own = (__gm__ uint8_t *)comm_.WindowIn(info_.epRankId);
        auto *out = (__gm__ uint8_t *)comm_.WindowOut(info_.epRankId);
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        // 等待所有 proxy 发来的 aggregate
        for (uint32_t proxy = aiv; proxy < info_.ranksPerServer; proxy += aivNum) {
            auto *flag = (__gm__ uint64_t *)(own + ProxyAggregateFlagOffset(layout_, proxy));
            TracePoint(901U, proxy);
            Wait(flag, ComposeFlag(info_.epoch, Phase::SERVER_FANOUT_DONE));
        }
        SyncAll<true>();
        TracePoint(902U, 0U);

        if (aiv == 0U) {
            for (uint32_t proxy = 0U; proxy < info_.ranksPerServer; ++proxy) {
                cacheReadInvalidate(own + ProxyAggregateOffset(layout_, proxy), layout_.proxyAggregateStride);
            }
            uint32_t ownerRows = 0U;
            int32_t physicalOffset = 0;
            const uint32_t alignment = info_.expertAlignment == 0U ? 1U : info_.expertAlignment;
            TracePoint(903U, localExperts_);
            for (uint32_t local = 0U; local < localExperts_; ++local) {
                int32_t running = 0;
                for (uint32_t sourceServer = 0U; sourceServer < info_.serverNum; ++sourceServer) {
                    for (uint32_t proxy = 0U; proxy < info_.ranksPerServer; ++proxy) {
                        auto *stage = out + BaseAckStageOffset(layout_, proxy, sourceServer);
                        *(__gm__ int32_t *)(stage + static_cast<uint64_t>(local) * sizeof(int32_t)) =
                            running;  // 记录每个源rank发送本地专家的偏移
                        auto *counts = own + ProxyAggregateOffset(layout_, proxy);
                        running +=
                            *(__gm__ int32_t *)(counts + ProxyOwnerCountOffset(sourceServer, local, localExperts_));
                    }
                }
                plan_[info_.localExpertCountOffset + local] = running;    // 保存本地专家接收的member数量
                plan_[info_.localExpertOffset + local] = physicalOffset;  // 保存本地专家接收的member偏移
                const uint32_t alignedRows = (static_cast<uint32_t>(running) + alignment - 1U) / alignment * alignment;
                recv_[local] = static_cast<int32_t>(alignedRows);
                ownerRows += static_cast<uint32_t>(running);
                physicalOffset += static_cast<int32_t>(alignedRows);
            }
#if DEEPEP_LAYERED_DEBUG_LOG
            LAYERED_LOG("[ln] owner rows=%u phys=%d lexp=%u\n", ownerRows, physicalOffset, localExperts_);
#endif
            for (uint32_t proxy = 0U; proxy < info_.ranksPerServer; ++proxy) {
                *(__gm__ uint32_t *)(out + BaseAckStageOwnerRowsOffset(layout_, proxy)) = ownerRows;  // 总member数量
            }
            cacheWriteThrough(out + BaseAckStageOffset(layout_, 0U, 0U), layout_.baseAckStageBytes);
            cacheWriteThrough(reinterpret_cast<__gm__ uint8_t *>(plan_ + info_.localExpertCountOffset),
                              static_cast<uint64_t>(localExperts_) * sizeof(int32_t));
            cacheWriteThrough(reinterpret_cast<__gm__ uint8_t *>(plan_ + info_.localExpertOffset),
                              static_cast<uint64_t>(localExperts_) * sizeof(int32_t));
            cacheWriteThrough(reinterpret_cast<__gm__ uint8_t *>(recv_),
                              static_cast<uint64_t>(localExperts_) * sizeof(int32_t));
        }
        TracePoint(904U, 0U);
    }

    __aicore__ inline void FanoutBaseAckSlicesToProxies()
    {
        auto *out = (__gm__ uint8_t *)comm_.WindowOut(info_.epRankId);
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        const uint32_t proxyBegin = static_cast<uint32_t>(static_cast<uint64_t>(info_.ranksPerServer) * aiv / aivNum);
        const uint32_t proxyEnd =
            static_cast<uint32_t>(static_cast<uint64_t>(info_.ranksPerServer) * (aiv + 1U) / aivNum);
        // 当前out的内存布局已经是
        // BaseAckStage[proxy][sourceServer][localExpert]
        // 直接拷贝对应 sourceServer 的连续 localExpert 行起点
        for (uint32_t proxyInServer = proxyBegin; proxyInServer < proxyEnd; ++proxyInServer) {
            const uint32_t proxy = sourceServer_ * info_.ranksPerServer + proxyInServer;
            auto *proxyWindow = (__gm__ uint8_t *)comm_.WindowIn(proxy);
            TracePoint(1001U, proxyInServer);
            CopyWindowBytes(out + BaseAckStageOffset(layout_, proxyInServer, 0U),
                            proxyWindow + BaseAckFanoutOwnerOffset(layout_, info_.rankInServer),
                            layout_.baseAckStageProxyStride);
            CopyWindowFlag((__gm__ uint64_t *)(proxyWindow + BaseAckFanoutFlagOffset(layout_, info_.rankInServer)),
                           ComposeFlag(info_.epoch, Phase::SERVER_FANOUT_DONE));
        }
        TracePoint(1002U, 0U);
    }

    __aicore__ inline void WaitAllBaseAckSlices()
    {
        auto *own = (__gm__ uint8_t *)comm_.WindowIn(info_.epRankId);
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        for (uint32_t owner = aiv; owner < info_.ranksPerServer; owner += aivNum) {
            TracePoint(1101U, owner);
            Wait((__gm__ uint64_t *)(own + BaseAckFanoutFlagOffset(layout_, owner)),
                 ComposeFlag(info_.epoch, Phase::SERVER_FANOUT_DONE));
        }
        TracePoint(1102U, 0U);
    }

    __aicore__ inline void AssembleBaseAcksAtProxies()
    {
        auto *in = (__gm__ uint8_t *)comm_.WindowIn(info_.epRankId);
        auto *out = (__gm__ uint8_t *)comm_.WindowOut(info_.epRankId);
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        const auto p = MakePlanLayout(info_.bs, info_.maxBs, info_.k, info_.moeExpertNum, info_.epWorldSize,
                                      info_.serverNum, info_.ranksPerServer, 1U, info_.maxBs);
        cacheReadInvalidate(reinterpret_cast<__gm__ uint8_t *>(plan_ + p.outboundTotalBytes), 2U * sizeof(int32_t));
        const uint64_t outboundBytes = static_cast<uint32_t>(plan_[p.outboundTotalBytes]) |
                                       (static_cast<uint64_t>(static_cast<uint32_t>(plan_[p.outboundTotalBytes + 1U]))
                                        << 32U);  // 本rank的总发送量
        for (uint32_t sourceServer = aiv; sourceServer < info_.serverNum; sourceServer += aivNum) {
            const uint32_t sourceRank = sourceServer * info_.ranksPerServer + info_.rankInServer;
            TracePoint(1201U, sourceServer);
            // ack[server][rank]
            auto *ack = out + BaseAckLaneOffset(layout_, info_.epWorldSize, sourceRank, sourceServer_);
            uint32_t serverMaxRows = 0U;  // 计算最大行
            for (uint32_t owner = 0U; owner < info_.ranksPerServer; ++owner) {
                auto *ownerRowsAddr = in + BaseAckFanoutOwnerRowsOffset(layout_, owner);
                cacheReadInvalidate(ownerRowsAddr, sizeof(uint32_t));
                auto *ownerRows = (__gm__ uint32_t *)ownerRowsAddr;
                serverMaxRows = serverMaxRows > *ownerRows ? serverMaxRows : *ownerRows;
            }
            BaseAckControlMeta control{};
            control.serverMaxRows = serverMaxRows;
            // Outbound staging occupies a compact prefix of this rank's
            // payload arena. Ingress starts immediately after the actual
            // outbound bytes, not after a worst-case per-server reservation.
            uint64_t ingressBase = outboundBytes;
            for (uint32_t preceding = 0U; preceding < sourceServer; ++preceding) {
                const uint32_t rank = preceding * info_.ranksPerServer + info_.rankInServer;
                auto *lane = in + ControlLaneOffset(layout_, info_.epWorldSize, rank, sourceServer_);
                cacheReadInvalidate(lane, layout_.controlStride);
                auto *counts = lane + CountControlDataOffset();
                const uint32_t packetCount = *(__gm__ uint32_t *)counts;
                auto *expertCounts = (__gm__ int32_t *)(counts + sizeof(uint32_t));
                uint32_t memberCount = 0U;
                for (uint32_t expert = 0U; expert < expertsPerServer_; ++expert) {
                    memberCount += static_cast<uint32_t>(expertCounts[expert]);
                }
                ingressBase += AlignUp64(DispatchBatchPayloadBytes(packetCount, memberCount, info_.recordStride));
            }
            auto *lane = in + ControlLaneOffset(layout_, info_.epWorldSize, sourceRank, sourceServer_);
            cacheReadInvalidate(lane, layout_.controlStride);
            auto *counts = lane + CountControlDataOffset();
            const uint32_t packetCount = *(__gm__ uint32_t *)counts;
            auto *expertCounts = (__gm__ int32_t *)(counts + sizeof(uint32_t));
            uint32_t memberCount = 0U;
            for (uint32_t expert = 0U; expert < expertsPerServer_; ++expert) {
                memberCount += static_cast<uint32_t>(expertCounts[expert]);
            }
            const uint64_t payloadBytes = DispatchBatchPayloadBytes(packetCount, memberCount, info_.recordStride);
            StorePlanValue(p.recvIngressBase + 2U * sourceServer, static_cast<int32_t>(ingressBase));
            StorePlanValue(p.recvIngressBase + 2U * sourceServer + 1U,
                           static_cast<int32_t>(ingressBase >> 32U));  // 当前AIV处理的server发给本卡的数据偏移
            StorePlanValue(p.recvIngressBytes + 2U * sourceServer, static_cast<int32_t>(payloadBytes));
            StorePlanValue(p.recvIngressBytes + 2U * sourceServer + 1U,
                           static_cast<int32_t>(payloadBytes >> 32U));  // 当前AIV处理的server发给本卡的数据大小
            control.ingressOffsetLo = static_cast<uint32_t>(ingressBase);
            control.ingressOffsetHi = static_cast<uint32_t>(ingressBase >> 32U);
            control.payloadBytesLo = static_cast<uint32_t>(payloadBytes);
            control.payloadBytesHi = static_cast<uint32_t>(payloadBytes >> 32U);
            /*
             *  ack:
             *    BaseAckControlMeta   记录proxy侧接收偏移和长度
             *    [owner][localExpert] 目标专家位置矩阵
             */
            CopyBaseAckControl(ack, control);  // 将上述计算的 BaseAckControlMeta 拷到头32B
            for (uint32_t owner = 0; owner < info_.ranksPerServer; ++owner) {
                // [owner][sourceServer][localExpert]
                auto *slice = in + BaseAckFanoutSliceOffset(layout_, owner, sourceServer);
                const uint64_t dstOffset = static_cast<uint64_t>(owner) * localExperts_ * sizeof(int32_t);
                CopyWindowBytes(slice, ack + sizeof(BaseAckControlMeta) + dstOffset,
                                static_cast<uint64_t>(localExperts_) *
                                    sizeof(int32_t));  // 将每个本地rank专家计算的 BaseAckStage 拷到这里
            }
        }
        TracePoint(1202U, 0U);
    }

    __aicore__ inline void PublishBaseAcksFromProxies()
    {
        auto *out = (__gm__ uint8_t *)comm_.WindowOut(info_.epRankId);
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        for (uint32_t sourceServer = aiv; sourceServer < info_.serverNum; sourceServer += aivNum) {
            const uint32_t source = sourceServer * info_.ranksPerServer + info_.rankInServer;
            auto *ack = out + BaseAckLaneOffset(layout_, info_.epWorldSize, source, sourceServer_);
            auto *sin = (__gm__ uint8_t *)comm_.WindowIn(source);
            // [ownerInServer][localExpert]
            auto *dst = sin + BaseAckLaneOffset(layout_, info_.epWorldSize, source, sourceServer_);
            TracePoint(1301U, sourceServer);
            if (source == info_.epRankId) {
                CopyWindowBytes(ack, dst, layout_.baseAckStride);
            } else {
                comm_.PostRdmaSend(ack, dst, source, layout_.baseAckStride, *doorbell_, *head_);
            }
            auto *flag =
                (__gm__ uint64_t *)(out + BaseAckFlagOffset(layout_, info_.epWorldSize, source, sourceServer_));
            *flag = ComposeFlag(info_.epoch, Phase::CONTROL_BASE_READY);
            // PostReadyFlag reads this local WindowOut word through RDMA. Make
            // the scalar store visible before submitting that second WQE.
            cacheWriteThrough((__gm__ uint8_t *)flag, sizeof(uint64_t));
            auto *remoteFlag =
                (__gm__ uint64_t *)(sin + BaseAckFlagOffset(layout_, info_.epWorldSize, source, sourceServer_));
            if (source == info_.epRankId) {
                CopyWindowFlag(remoteFlag, *flag);
            } else {
                comm_.PostReadyFlag((GM_ADDR)flag, (GM_ADDR)remoteFlag, source, *doorbell_, *head_);
            }
        }
        TracePoint(1302U, 0U);
    }
    __aicore__ inline void ApplyAllReturnedRowBases()
    {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = info_.aivNum == 0U ? 1U : info_.aivNum;
        auto *self = (__gm__ uint8_t *)comm_.WindowIn(info_.epRankId);
        if (aiv == 0U) {
            int32_t total = 0;
            for (uint32_t i = 0; i < localExperts_; ++i) {
                total += recv_[i];
            }
            total_[0] = total;
            cacheWriteThrough(reinterpret_cast<__gm__ uint8_t *>(total_), sizeof(int32_t));
            TracePoint(1401U, static_cast<uint32_t>(total));
        }

        // 等待数据收齐
        uint32_t localMaxRows = 0U;
        for (uint32_t target = aiv; target < info_.serverNum; target += aivNum) {
            auto *f = (__gm__ uint64_t *)(self + BaseAckFlagOffset(layout_, info_.epWorldSize, info_.epRankId, target));
            TracePoint(1402U, target);
            Wait(f, ComposeFlag(info_.epoch, Phase::CONTROL_BASE_READY));
            auto *ack = self + BaseAckLaneOffset(layout_, info_.epWorldSize, info_.epRankId, target);
            cacheReadInvalidate(ack, layout_.baseAckStride);
            /*
             *  ack:
             *    BaseAckControlMeta   记录proxy侧接收偏移和长度
             *    [owner][localExpert] 目标专家位置矩阵
             */
            auto *control = (__gm__ BaseAckControlMeta *)ack;
            const auto p = MakePlanLayout(info_.bs, info_.maxBs, info_.k, info_.moeExpertNum, info_.epWorldSize,
                                          info_.serverNum, info_.ranksPerServer, 1U, info_.maxBs);
            StorePlanValue(p.sendIngressBase + 2U * target, static_cast<int32_t>(control->ingressOffsetLo));
            StorePlanValue(p.sendIngressBase + 2U * target + 1U,
                           static_cast<int32_t>(control->ingressOffsetHi));  // 记录发送到对端proxy的偏移
            StorePlanValue(p.sendIngressBytes + 2U * target, static_cast<int32_t>(control->payloadBytesLo));
            StorePlanValue(p.sendIngressBytes + 2U * target + 1U,
                           static_cast<int32_t>(control->payloadBytesHi));  // 记录发送到对端proxy的长度
            localMaxRows = localMaxRows > control->serverMaxRows ? localMaxRows : control->serverMaxRows;
        }
        workspace_[MaxRowsScratch(aiv)] = static_cast<int32_t>(localMaxRows);
        cacheWriteThrough((__gm__ uint8_t *)(workspace_ + MaxRowsScratch(aiv)), sizeof(int32_t));
        SyncAll<true>();
        TracePoint(1403U, localMaxRows);

        if (aiv == 0U) {
            uint32_t globalMaxRows = 0U;
            for (uint32_t worker = 0U; worker < aivNum; ++worker) {
                auto *value = workspace_ + MaxRowsScratch(worker);
                cacheReadInvalidate((__gm__ uint8_t *)value, sizeof(int32_t));
                globalMaxRows =
                    globalMaxRows > static_cast<uint32_t>(*value) ? globalMaxRows : static_cast<uint32_t>(*value);
            }
            auto *header = (__gm__ PlanHeader *)plan_;
            header->globalCombineRows = globalMaxRows;  // 获取全局最大member值
            cacheWriteThrough((__gm__ uint8_t *)&header->globalCombineRows, sizeof(uint32_t));
        }

        const uint32_t packetElements = info_.serverNum * info_.slotsPerServer;
        const uint32_t begin = static_cast<uint32_t>(static_cast<uint64_t>(packetElements) * aiv / aivNum);
        const uint32_t end = static_cast<uint32_t>(static_cast<uint64_t>(packetElements) * (aiv + 1U) / aivNum);
        cacheReadInvalidate(reinterpret_cast<__gm__ uint8_t *>(plan_ + info_.packetDescOffset),
                            static_cast<uint64_t>(packetElements) * sizeof(SourcePacketDesc));
        cacheReadInvalidate(reinterpret_cast<__gm__ uint8_t *>(plan_ + info_.memberDescOffset),
                            static_cast<uint64_t>(packetElements) * info_.k * sizeof(MemberDesc));
        // 每个AIV处理一部分packet
        for (uint32_t packet = begin; packet < end; ++packet) {
            if ((packet & 63U) == 0U) {
                TracePoint(1404U, packet);
            }
            const uint32_t target = (packet / info_.slotsPerServer) % info_.serverNum;
            const uint32_t slot = packet % info_.slotsPerServer;
            const uint32_t token = slot;
            if (token >= info_.bs) {
                continue;
            }
            CopyPlanToUb(info_.packetDescOffset + packet * PacketElements(), *initUb_, sizeof(SourcePacketDesc));
            const uint32_t memberCount = static_cast<uint32_t>(initUb_->GetValue(2U));  // 发给该server的member数量
            // Same bound as in ReduceLocalCounts: a packet holds at most k
            // members, larger values would overrun initUb_ in the copy below.
            if (memberCount > info_.k) {
                continue;
            }
            auto *ack = self + BaseAckLaneOffset(layout_, info_.epWorldSize, info_.epRankId, target);
            cacheReadInvalidate(ack, layout_.baseAckStride);
            const uint32_t expertBegin = target * expertsPerServer_;
            // 对当前 packet 中每个实际topk member，找到目标 server 回传的对应 expert 内 row base,
            // 再加上该 member 在本地的局部序号，得到它在目标 expert 内的最终行号
            for (uint32_t m = 0; m < memberCount; ++m) {
                const uint32_t mb = info_.memberDescOffset + (packet * info_.k + m) * MemberElements();
                CopyPlanToUb(mb, *initUb_, sizeof(MemberDesc));
                const uint32_t expert = static_cast<uint32_t>(initUb_->GetValue(0U));  // 专家id
                const uint32_t expertInServer = expert - expertBegin;
                const uint32_t owner = expertInServer / localExperts_;
                const uint32_t local = expertInServer % localExperts_;
                auto *base = (__gm__ int32_t *)(ack + BaseAckOwnerExpertOffset(owner, local, localExperts_));
                initUb_->SetValue(1U, initUb_->GetValue(1U) + *base);  // 保存该member发送给目标专家的最终行号
                CopyUbToPlan(mb, *initUb_, sizeof(MemberDesc));
            }
        }
        TracePoint(1405U, end);
    }
    __gm__ int32_t *topk_{nullptr};
    __gm__ int32_t *plan_{nullptr};
    __gm__ int32_t *recv_{nullptr};
    __gm__ int32_t *total_{nullptr};
    __gm__ int32_t *workspace_{nullptr};
    CamMoeLayeredInfo info_{};
    WindowLayout layout_{};
    CommAdapter comm_{};
    LocalTensor<uint64_t> *doorbell_{nullptr};
    LocalTensor<uint32_t> *head_{nullptr};
    LocalTensor<int32_t> *initUb_{nullptr};
    LocalTensor<int32_t> *reduceSrcUb_{nullptr};
    LocalTensor<int32_t> *reduceAccUb_{nullptr};
    LocalTensor<uint8_t> *aggregateCopyUb_{nullptr};
    LocalTensor<int32_t> *packetPrefixUb_{nullptr};
    LocalTensor<int32_t> *memberPrefixUb_{nullptr};
    LocalTensor<int32_t> *expertPrefixUb_{nullptr};
    uint32_t localExperts_{0};
    uint32_t expertsPerServer_{0};
    uint32_t sourceServer_{0};
    uint32_t countSlice_{0};
    uint32_t packetSlice_{0};
    uint32_t countSlicePadded_{0};
    uint32_t packetSlicePadded_{0};
    uint32_t memberCountSlicePadded_{0};
};
}  // namespace

// Kernel arguments must follow the op prototype order declared in
// op_host/notify_dispatch_normal_layered.cpp:
//   inputs  : roce_context, topk_idx
//   outputs : recvTokensPerExpert, layeredPlan, expertTotalCount
//   then    : workspace, tiling
// Swapping the two plan outputs makes plan_ point at the tiny
// recvTokensPerExpert tensor, so every plan offset (memberDescOffset, ...)
// lands outside that allocation and the descriptors read back garbage counts.
extern "C" __global__ __aicore__ void notify_dispatch_normal_layered(GM_ADDR roceContext, GM_ADDR topkIdx,
                                                                     GM_ADDR recvTokensPerExpert,
                                                                     GM_ADDR layeredPlan,
                                                                     GM_ADDR expertTotalCount, GM_ADDR workspace,
                                                                     GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NotifyDispatchNormalLayeredTilingData);
    // TPipe must be constructed before GET_TILING_DATA_WITH_STRUCT: the tiling
    // copy resolves its UB staging buffer through GetTPipePtr(). With the
    // buffer pool still uninitialised the copy works on a stale UB address,
    // which is the "scalar access out of internal buffer" (507035) class of
    // fault and is independent of batch/topk/expert/hidden/network. The working
    // cam_moe_dispatch_normal_layered entry orders it the same way.
    TPipe pipe;
#if DEEPEP_LAYERED_DEBUG_LOG
    // Fixed boot marker: proves kernel printf reaches the device log at all,
    // independently of tiling data and of any stage/point being reached.
    PipeBarrier<PIPE_ALL>();
    if (GetBlockIdx() == 0U) {
        LAYERED_LOG("[ln] boot notify_dispatch_normal_layered\n");
    }
    PipeBarrier<PIPE_ALL>();
#endif
    // Must match SetTilingKey(23U) in notify_dispatch_normal_layered_tiling.cc.
    // Declaring the key here is what makes the op compiler emit the _23
    // funcEntry; without it only the default key-0 entry is generated.
    if (TILING_KEY_IS(23)) {
#if DEEPEP_LAYERED_DEBUG_LOG
        PipeBarrier<PIPE_ALL>();
        if (GetBlockIdx() == 0U) {
            LAYERED_LOG("[ln] tiling key 23 entered\n");
        }
        PipeBarrier<PIPE_ALL>();
#endif
        GET_TILING_DATA_WITH_STRUCT(NotifyDispatchNormalLayeredTilingData, data, tiling);
#if DEEPEP_LAYERED_DEBUG_LOG
        PipeBarrier<PIPE_ALL>();
        if (GetBlockIdx() == 0U) {
            LAYERED_LOG("[ln] t23 data ok rank=%u world=%u aiv=%u\n", data.layered.epRankId,
                        data.layered.epWorldSize, data.layered.aivNum);
        }
        PipeBarrier<PIPE_ALL>();
#endif
#ifdef __DAV_C310__
        TBuf hcommUb;
        pipe.InitBuffer(hcommUb, kHcommUbBytes);
#endif
        TBuf rdmaUb;
        pipe.InitBuffer(rdmaUb, 32U);
        TBuf sharedWorkUb;
        pipe.InitBuffer(sharedWorkUb, kSharedWorkUbBytes);
        auto doorbell = rdmaUb.GetWithOffset<uint64_t>(1U, 0U);
        auto head = rdmaUb.GetWithOffset<uint32_t>(1U, 16U);
        // Reduction owns both halves; prefix construction reuses the second half
        // after reduction, with the first half as its copy-in scratch space.
        auto initTensor = sharedWorkUb.GetWithOffset<int32_t>(kInitTileElements, 0U);
        auto reduceSrcTensor = sharedWorkUb.GetWithOffset<int32_t>(kReduceTileElements, 0U);
        auto reduceAccTensor = sharedWorkUb.GetWithOffset<int32_t>(kReduceTileElements, kReduceTileBytes);
        auto aggregateCopyTensor = sharedWorkUb.GetWithOffset<uint8_t>(kAggregateCopyTileBytes, 0U);
        auto expertPrefixTensor = sharedWorkUb.GetWithOffset<int32_t>(kMaxPrefixExperts, kReduceTileBytes);
        auto packetPrefixTensor =
            sharedWorkUb.GetWithOffset<int32_t>(kMaxPrefixServers, kReduceTileBytes + kExpertPrefixBytes);
        auto memberPrefixTensor = sharedWorkUb.GetWithOffset<int32_t>(
            kMaxPrefixServers, kReduceTileBytes + kExpertPrefixBytes + kServerPrefixBytes);
#ifdef __DAV_C310__
        auto hcommTensor = hcommUb.Get<uint8_t>();
#else
        auto hcommTensor = aggregateCopyTensor;
#endif
        NotifyDispatchNormalLayered op;
#if DEEPEP_LAYERED_DEBUG_LOG
        PipeBarrier<PIPE_ALL>();
        if (GetBlockIdx() == 0U) {
            LAYERED_LOG("[ln] t23 pre-init\n");
        }
        PipeBarrier<PIPE_ALL>();
#endif
        op.Init(roceContext, topkIdx, layeredPlan, recvTokensPerExpert, expertTotalCount, workspace, data.layered,
                doorbell, head, initTensor, reduceSrcTensor, reduceAccTensor, aggregateCopyTensor, hcommTensor,
                packetPrefixTensor, memberPrefixTensor, expertPrefixTensor);
#if DEEPEP_LAYERED_DEBUG_LOG
        PipeBarrier<PIPE_ALL>();
        if (GetBlockIdx() == 0U) {
            LAYERED_LOG("[ln] t23 post-init\n");
        }
        PipeBarrier<PIPE_ALL>();
#endif
        op.Process();
#if DEEPEP_LAYERED_DEBUG_LOG
        PipeBarrier<PIPE_ALL>();
        if (GetBlockIdx() == 0U) {
            LAYERED_LOG("[ln] t23 post-process\n");
        }
        PipeBarrier<PIPE_ALL>();
#endif
#if DEEPEP_LAYERED_DEBUG_LOG
        PipeBarrier<PIPE_ALL>();
        if (GetBlockIdx() == 0U) {
            LAYERED_LOG("[ln] done notify_dispatch_normal_layered\n");
        }
        PipeBarrier<PIPE_ALL>();
#endif
    }
}

#endif  // DEEPEP_CANN_VERSION_NUM >= DEEPEP_LAYERED_MIN_CANN_VERSION
