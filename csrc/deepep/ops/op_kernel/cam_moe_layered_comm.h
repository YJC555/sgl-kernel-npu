#ifndef CAM_MOE_LAYERED_COMM_H
#define CAM_MOE_LAYERED_COMM_H

#include "kernel_operator.h"
#ifdef __DAV_C310__
#include "adv_api/hcomm/hcomm.h"
#endif
#include "cam_moe_layered_comm_context.h"
#include "moe_distribute_base.h"
#include "cam_moe_layered_plan.h"
#include "window_layout.h"

namespace CamMoeLayered {

using namespace AscendC;

// 910_93: the op is declared with AIV_ENGINE (same as the shipped
// notify_dispatch tiling), so no AICPU/KFC server is launched and the peer
// windows come from the launch context's raw relation table
// (remoteRes[].nextDevicePtr->windowsIn). Hccl<AICPU>::GetWindowsInAddr is NOT
// usable here: it returns 0 for every cross-node rank on this chip.

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    const int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

class CommAdapter
{
public:
    __aicore__ inline void Init(uint32_t rankId, GM_ADDR roceContext, AscendC::LocalTensor<uint8_t> &hcommUb,
                                bool isHybridDeployment = false)
    {
        rankId_ = rankId;
        (void)isHybridDeployment;
        context_ = (__gm__ HcclOpParam *)AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
#ifndef __DAV_C310__  // A3 transport
        // The launch context is the MC2 op param; peer windows are read from
        // its raw relation table. HCCL never publishes an AI-RMA QP table for
        // an AlltoAll MC2 op on 910_93, so the data plane is window writes.
        rdmaInfo_ = (__gm__ HcclAiRMAInfo *)context_->aiRMAInfo;
        (void)roceContext;
#else  // A5 Hcomm transport
        rdmaContext_ = (__gm__ CamMoeLayeredCommContext *)roceContext;
        if (rdmaContext_->protocol == CAM_MOE_LAYERED_PROTOCOL_ROCE) {
            hcommInitOk_ = roce_.Init(hcommUb, AscendC::HCOMM_UB_BUF_SIZE) == 0;
        } else {
            hcommInitOk_ = uboe_.Init(hcommUb, AscendC::HCOMM_UB_BUF_SIZE) == 0;
        }
#endif
    }

    __aicore__ inline uint32_t RankId() const
    {
        return rankId_;
    }
    __aicore__ inline uint32_t RankSize() const
    {
        return GetRankDim(context_);
    }
    __aicore__ inline uint64_t RegisteredWindowBytes() const
    {
#ifdef __DAV_C310__
        return rdmaContext_->windowBytes;
#else
        return 0U;
#endif
    }
    __aicore__ inline GM_ADDR GetBufferAddrByRankId(uint32_t rank) const
    {
#ifdef __DAV_C310__
        return (GM_ADDR)rdmaContext_->ipcBuffer[rank];
#else
        if (rank == rankId_) {
            return (GM_ADDR)context_->localWindowsIn;
        }
        // Cross-rank: HCCL mirrors the peer window into this device's relation
        // table. A null nextDevicePtr (or a zero windowsIn) means the mapping
        // was never published for that rank; callers must not dereference it.
        auto *relation = (__gm__ HcclRankRelationResV2 *)context_->remoteRes[rank].nextDevicePtr;
        if (relation == nullptr) {
            return nullptr;
        }
        return (GM_ADDR)relation->windowsIn;
#endif
    }
    __aicore__ inline GM_ADDR WindowIn(uint32_t rank) const
    {
        return GetBufferAddrByRankId(rank);
    }
    __aicore__ inline GM_ADDR WindowOut(uint32_t rank) const
    {
#ifdef __DAV_C310__
        return WindowIn(rank);
#else
        if (rank == rankId_) {
            return (GM_ADDR)context_->localWindowsOut;
        }
        auto *relation = (__gm__ HcclRankRelationResV2 *)context_->remoteRes[rank].nextDevicePtr;
        if (relation == nullptr) {
            return nullptr;
        }
        return (GM_ADDR)relation->windowsOut;
#endif
    }
    __aicore__ inline uint64_t WinSizeBytes() const
    {
        return GetWinSize(context_);
    }
    // True when a copy into this rank's window is legal. A3 resolves the peer
    // window from remoteRes; a null nextDevicePtr or zero windowsIn is exactly
    // the case that used to turn into an MTE write to address 0, so the send
    // paths skip it (and trace it) instead of trapping.
    // A5 has no direct peer mapping at all: only the local IPC buffer is
    // addressable, every remote send goes through the hcomm channel.
    __aicore__ inline bool PeerWindowUsable(uint32_t rank) const
    {
#ifdef __DAV_C310__
        return rank == rankId_;
#else
        return WindowIn(rank) != nullptr;
#endif
    }
    // Debug helper: snapshot of a peer relation resource with a null guard, so a
    // missing remote entry reports 0 instead of faulting on the dereference.
    __aicore__ inline void LoadPeerWindow(uint32_t rank, uint64_t &nextDevPtr, uint64_t &windowIn) const
    {
#ifndef __DAV_C310__
        nextDevPtr = reinterpret_cast<uint64_t>(context_->remoteRes[rank].nextDevicePtr);
        windowIn = 0U;
        if (nextDevPtr != 0U) {
            windowIn = ((__gm__ HcclRankRelationResV2 *)nextDevPtr)->windowsIn;
        }
#else
        (void)rank;
        nextDevPtr = 0U;
        windowIn = 0U;
#endif
    }
#ifdef __DAV_C310__
    __aicore__ inline AscendC::ChannelHandle GetChannel(uint32_t dstRank) const
    {
        const uint32_t peerIndex = dstRank < rankId_ ? dstRank : dstRank - 1U;
        return static_cast<AscendC::ChannelHandle>(rdmaContext_->hcommHandle[peerIndex]);
    }
#endif

    // Runtime initialization guarantees the WQE and memory ABI. Callers own
    // one destination rank/QP per AIV during the full submission phase.
    __aicore__ inline bool PostRdmaSend(GM_ADDR src, GM_ADDR dst, uint32_t dstRank, uint32_t bytes,
                                        AscendC::LocalTensor<uint64_t> &doorbellUb,
                                        AscendC::LocalTensor<uint32_t> &headUb, uint64_t timeoutCycles = 0U) const
    {
#ifndef __DAV_C310__  // A3 AiRMA RDMA write
        if (rdmaInfo_ == nullptr) {
            return false;
        }
        auto qpNum = rdmaInfo_->qpNum;
        auto *qp = (__gm__ HcclAiRMAWQ *)(rdmaInfo_->sqPtr +
                                          static_cast<uint64_t>(dstRank) * qpNum * rdmaInfo_->sizeOfAiRMAWQ);
        auto *headAddr = (__gm__ uint32_t *)qp->headAddr;
        auto *tailAddr = (__gm__ uint32_t *)qp->tailAddr;
        cacheWriteThrough((__gm__ uint8_t *)headAddr, sizeof(uint64_t));
        uint64_t head = *headAddr;
        const uint64_t start = AscendC::GetSystemCycle();
        AscendC::PipeBarrier<PIPE_ALL>();
        while (true) {
            cacheWriteThrough((__gm__ uint8_t *)tailAddr, sizeof(uint64_t));
            if (head - *tailAddr < qp->depth - 1U) {
                break;
            }
            if (timeoutCycles != 0U && AscendC::GetSystemCycle() - start > timeoutCycles) return false;
            AscendC::GetSystemCycle();
        }
        auto *wqe = (__gm__ uint8_t *)(qp->bufAddr + qp->wqeSize * (head % qp->depth));
        const uint64_t ownBit = (head >> 15U) & 1U;
        *(__gm__ uint32_t *)(wqe) = 3U | (((~ownBit) << 7U) & (1U << 7U)) | (1U << 8U);
        *(__gm__ uint32_t *)(wqe + 4U) = bytes;
        *(__gm__ uint32_t *)(wqe + 8U) = 0U;
        *(__gm__ uint32_t *)(wqe + 12U) = 1U << 24U;
        *(__gm__ uint32_t *)(wqe + 16U) = 0U;
        auto *remoteMem =
            (__gm__ HcclAiRMAMemInfo *)(rdmaInfo_->memPtr + static_cast<uint64_t>(dstRank) * rdmaInfo_->sizeOfAiRMAMem);
        auto *remoteKey =
            (__gm__ MemDetails1 *)(remoteMem->memDetailPtr +
                                   remoteMem->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::REMOTE_INPUT));
        *(__gm__ uint32_t *)(wqe + 20U) = remoteKey->key;
        *(__gm__ uint64_t *)(wqe + 24U) = reinterpret_cast<uint64_t>(dst);
        auto *sge = wqe + sizeof(hns_roce_rc_sq_wqe);
        *(__gm__ uint32_t *)(sge) = bytes;
        auto *localMem =
            (__gm__ HcclAiRMAMemInfo *)(rdmaInfo_->memPtr + static_cast<uint64_t>(dstRank) * rdmaInfo_->sizeOfAiRMAMem);
        auto *localKey =
            (__gm__ MemDetails1 *)(localMem->memDetailPtr +
                                   localMem->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT));
        *(__gm__ uint32_t *)(sge + sizeof(uint32_t)) = localKey->key;
        *(__gm__ uint64_t *)(sge + 2U * sizeof(uint32_t)) = reinterpret_cast<uint64_t>(src);
        cacheWriteThrough(wqe, sizeof(hns_roce_rc_sq_wqe) + sizeof(hns_roce_lite_wqe_data_seg));
        AscendC::PipeBarrier<PIPE_ALL>();
        ++head;
        uint64_t db = qp->wqn | ((head % 65536ULL) << 32U) | (static_cast<uint64_t>(qp->sl) << 48U);
        AscendC::PipeBarrier<PIPE_ALL>();
        doorbellUb.SetValue(0, db);
        GlobalTensor<uint64_t> dbGm;
        dbGm.SetGlobalBuffer((__gm__ uint64_t *)qp->dbAddr);
        DataCopyPad(dbGm, doorbellUb, DataCopyExtParams{1U, sizeof(uint64_t), 0U, 0U, 0U});
        AscendC::PipeBarrier<PIPE_ALL>();
        headUb.SetValue(0, static_cast<uint32_t>(head));
        GlobalTensor<uint32_t> headGm;
        headGm.SetGlobalBuffer(headAddr);
        DataCopyPad(headGm, headUb, DataCopyExtParams{1U, sizeof(uint32_t), 0U, 0U, 0U});
        AscendC::PipeBarrier<PIPE_ALL>();
        return true;
#else
        (void)doorbellUb;
        (void)headUb;
        (void)timeoutCycles;
        if (!hcommInitOk_) {
            return false;
        }
        if (rdmaContext_->protocol == CAM_MOE_LAYERED_PROTOCOL_ROCE) {
            return roce_.WriteNbi<true, PIPE_S, PIPE_MTE3>(GetChannel(dstRank), dst, src, bytes) == 0;
        }
        return uboe_.WriteNbi<true, PIPE_S, PIPE_MTE3>(GetChannel(dstRank), dst, src, bytes) == 0;
#endif
    }

    // A ready flag is never embedded in a payload WQE.  The caller posts the
    // payload first, then calls this helper with the flag word in its own GM
    // staging address.  Receivers may therefore poll exactly one 64-bit word
    // without relying on PCIe/RoCE write ordering of unrelated cache lines.
    __aicore__ inline bool PostReadyFlag(GM_ADDR localFlag, GM_ADDR remoteFlag, uint32_t dstRank,
                                         AscendC::LocalTensor<uint64_t> &doorbellUb,
                                         AscendC::LocalTensor<uint32_t> &headUb, uint64_t timeoutCycles = 0U) const
    {
        return PostRdmaSend(localFlag, remoteFlag, dstRank, sizeof(uint64_t), doorbellUb, headUb, timeoutCycles);
    }

    // Debug helper: raw binding of the HCCL context that carries the AiRMA
    // table, read without dereferencing a null context. The context is always
    // HcclOpResParam on 910_93, even on the AICPU domain. The three fields
    // separate the failure modes: ctx==0 means GetHcclContext returned nothing
    // for HCCL_GROUP_ID_0, dim==0 or a wrong rank count means the context is not
    // the expected struct, size==0 means the AiRMA section was never built, and
    // size!=0 with info==0 means the table pointer was left unpublished.
    __aicore__ inline void LoadContextDesc(uint64_t &ctx, uint32_t &dim, uint64_t &infoSize, uint64_t &info) const
    {
#ifndef __DAV_C310__
        ctx = reinterpret_cast<uint64_t>(context_);
        dim = 0U;
        infoSize = 0U;
        info = 0U;
        if (ctx != 0U) {
            dim = context_->rankSize;
            infoSize = context_->sizeOfAiRMAInfo;
            info = context_->aiRMAInfo;
        }
#else
        (void)infoSize;
        (void)info;
        ctx = reinterpret_cast<uint64_t>(context_);
        dim = ctx != 0U ? GetRankDim(context_) : 0U;
#endif
    }

    __aicore__ inline bool Drain(uint32_t dstRank, uint64_t timeoutCycles) const
    {
        if (dstRank == rankId_) return true;
#ifndef __DAV_C310__
        if (rdmaInfo_ == nullptr) return false;
        const uint32_t qpNum = rdmaInfo_->qpNum;
        auto *qp = (__gm__ HcclAiRMAWQ *)(rdmaInfo_->sqPtr +
                                          static_cast<uint64_t>(dstRank) * qpNum * rdmaInfo_->sizeOfAiRMAWQ);
        auto *head = (__gm__ uint32_t *)qp->headAddr;
        auto *tail = (__gm__ uint32_t *)qp->tailAddr;
        cacheReadInvalidate((__gm__ uint8_t *)head, sizeof(uint32_t));
        const uint32_t submitted = *head;
        const uint64_t start = AscendC::GetSystemCycle();
        while (true) {
            cacheReadInvalidate((__gm__ uint8_t *)tail, sizeof(uint32_t));
            if (*tail == submitted) return true;
            if (timeoutCycles != 0U && AscendC::GetSystemCycle() - start > timeoutCycles) return false;
        }
#else
        (void)timeoutCycles;
        if (!hcommInitOk_) {
            return false;
        }
        if (rdmaContext_->protocol == CAM_MOE_LAYERED_PROTOCOL_ROCE) {
            return roce_.Drain(GetChannel(dstRank)) == 0;
        }
        return uboe_.Drain(GetChannel(dstRank)) == 0;
#endif
    }

private:
    __gm__ HcclOpParam *context_{nullptr};
#ifdef __DAV_C310__
    __gm__ CamMoeLayeredCommContext *rdmaContext_{nullptr};
    mutable AscendC::Hcomm<AscendC::COMM_PROTOCOL_ROCE> roce_;
    mutable AscendC::Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> uboe_;
    bool hcommInitOk_{false};
#else
    __gm__ HcclAiRMAInfo *rdmaInfo_{nullptr};
#endif
    uint32_t rankId_{0};
};

}  // namespace CamMoeLayered

#endif
