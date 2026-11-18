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
#ifndef __DAV_C310__  // A3 AiRMA transport
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
#endif
        return GetBaseWindAddrByRankId(context_, rank, rankId_);
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
        return (GM_ADDR)relation->windowsOut;
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
        *(__gm__ uint64_t *)(wqe + 24U) = static_cast<uint64_t>(dst);
        auto *sge = wqe + sizeof(hns_roce_rc_sq_wqe);
        *(__gm__ uint32_t *)(sge) = bytes;
        auto *localMem =
            (__gm__ HcclAiRMAMemInfo *)(rdmaInfo_->memPtr + static_cast<uint64_t>(dstRank) * rdmaInfo_->sizeOfAiRMAMem);
        auto *localKey =
            (__gm__ MemDetails1 *)(localMem->memDetailPtr +
                                   localMem->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT));
        *(__gm__ uint32_t *)(sge + sizeof(uint32_t)) = localKey->key;
        *(__gm__ uint64_t *)(sge + 2U * sizeof(uint32_t)) = static_cast<uint64_t>(src);
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

    __aicore__ inline bool Drain(uint32_t dstRank, uint64_t timeoutCycles) const
    {
        if (dstRank == rankId_) return true;
#ifndef __DAV_C310__
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
