#if DEEPEP_CANN_VERSION_NUM >= DEEPEP_LAYERED_MIN_CANN_VERSION

#include "kernel_operator.h"
#include "cam_moe_dispatch_normal_layered_tiling.h"
#include "cam_moe_layered_comm.h"
#include "cam_moe_layered_plan.h"

using namespace AscendC;
using namespace CamMoeLayered;

namespace {
constexpr uint32_t kPeerCopyTileBytes = 16U * 1024U;
constexpr uint32_t kMaxTopk = 16U;

template <typename T>
class CamMoeDispatchNormalLayered
{
public:
    __aicore__ inline void Init(GM_ADDR roceContext, GM_ADDR x, GM_ADDR topkWeights, GM_ADDR topkIdx,
                                GM_ADDR layeredPlan, GM_ADDR recvX, GM_ADDR recvTopkIdx, GM_ADDR recvTopkWeights,
                                GM_ADDR workspace, const CamMoeLayeredInfo &tiling, LocalTensor<uint64_t> &doorbellUb,
                                LocalTensor<uint32_t> &headUb, LocalTensor<uint8_t> &copyUb,
                                LocalTensor<uint8_t> &hcommUb)
    {
        info_ = tiling;
        header_ = (__gm__ PlanHeader *)layeredPlan;
        plan_ = (__gm__ int32_t *)layeredPlan;
        x_ = (__gm__ T *)x;
        weights_ = (__gm__ float *)topkWeights;
        topk_ = (__gm__ int32_t *)topkIdx;
        recv_ = (__gm__ T *)recvX;
        recvTopk_ = (__gm__ int32_t *)recvTopkIdx;
        recvWeights_ = (__gm__ float *)recvTopkWeights;
        doorbellUb_ = &doorbellUb;
        headUb_ = &headUb;
        copyUb_ = &copyUb;
        scratch_ = (__gm__ uint32_t *)workspace;
        sourceServer_ = info_.epRankId / info_.ranksPerServer;
        localExperts_ = info_.moeExpertNum / info_.epWorldSize;
        layout_ = MakeWindowLayout(info_.moeExpertNum, info_.epWorldSize, info_.ranksPerServer, info_.serverNum,
                                   info_.physicalRounds, info_.slotsPerServer, info_.recordStride, info_.partialStride,
                                   info_.k, info_.isHybridDeployment != 0U);
        comm_.Init(info_.epRankId, roceContext, hcommUb, info_.isHybridDeployment != 0U);
        localOut_ = (__gm__ uint8_t *)comm_.WindowOut(info_.epRankId);
        epoch_ = static_cast<uint64_t>(header_->epochLo) | (static_cast<uint64_t>(header_->epochHi) << 32U);
    }

    __aicore__ inline void Process()
    {
        BuildBatchMeta();
        SyncAll<true>();
        BuildRecordsParallel();
        InitPipelineScratch();
        SyncAll<true>();
        RunPipeline();
        SyncAll<true>();
        DrainOutbound();
        SyncAll<true>();
    }

private:
    __aicore__ inline uint32_t ActiveAivNum() const
    {
        return info_.aivNum == 0U ? 1U : info_.aivNum;
    }

    __aicore__ inline uint32_t PacketIndex(uint32_t server, uint32_t slot) const
    {
        return server * info_.slotsPerServer + slot;
    }

    __aicore__ inline uint64_t BatchOffset(uint32_t targetServer) const
    {
        const auto p = MakePlanLayout(info_.bs, info_.maxBs, info_.k, info_.moeExpertNum, info_.epWorldSize,
                                      info_.serverNum, info_.ranksPerServer, 1U, info_.maxBs);
        const uint32_t base = p.outboundBase + 2U * targetServer;
        const uint64_t compactOffset = static_cast<uint32_t>(plan_[base]) |
                                       (static_cast<uint64_t>(static_cast<uint32_t>(plan_[base + 1U])) << 32U);
        return DispatchPayloadAddress(layout_, compactOffset);
    }

    __aicore__ inline uint64_t IngressOffset(uint32_t sourceServer, bool send) const
    {
        const auto p = MakePlanLayout(info_.bs, info_.maxBs, info_.k, info_.moeExpertNum, info_.epWorldSize,
                                      info_.serverNum, info_.ranksPerServer, 1U, info_.maxBs);
        const uint32_t base = (send ? p.sendIngressBase : p.recvIngressBase) + 2U * sourceServer;
        return static_cast<uint32_t>(plan_[base]) |
               (static_cast<uint64_t>(static_cast<uint32_t>(plan_[base + 1U])) << 32U);
    }

    __aicore__ inline uint64_t DispatchReadyAddress(uint32_t sourceServer, uint32_t targetServer) const
    {
        return PhaseFlagOffset(layout_, info_.serverNum, sourceServer, targetServer, Phase::DISPATCH_DATA_READY);
    }

    __aicore__ inline void CopyPeerBytes(__gm__ uint8_t *dst, __gm__ uint8_t *src, uint32_t bytes)
    {
        for (uint32_t offset = 0; offset < bytes;) {
            const uint32_t count = min(kPeerCopyTileBytes, bytes - offset);
            GlobalTensor<uint8_t> srcTensor;
            srcTensor.SetGlobalBuffer(src + offset);
            const DataCopyExtParams params{1U, count, 0U, 0U, 0U};
            const DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
            DataCopyPad(*copyUb_, srcTensor, params, padParams);
            SyncFunc<HardEvent::MTE2_V>();
            GlobalTensor<uint8_t> dstTensor;
            dstTensor.SetGlobalBuffer(dst + offset);
            DataCopyPad(dstTensor, *copyUb_, params);
            SyncFunc<HardEvent::MTE3_S>();
            offset += count;
        }
    }

    __aicore__ inline void CopyPeerFlag(__gm__ uint64_t *dst, uint64_t value)
    {
        doorbellUb_->SetValue(0, value);
        GlobalTensor<uint64_t> dstTensor;
        dstTensor.SetGlobalBuffer(dst);
        SyncFunc<HardEvent::S_MTE3>();
        DataCopyPad(dstTensor, *doorbellUb_, DataCopyExtParams{1U, sizeof(uint64_t), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }

    __aicore__ inline void StoreWords(__gm__ uint32_t *dst, const uint32_t *values, uint32_t count)
    {
        auto words = copyUb_->template ReinterpretCast<uint32_t>();
        for (uint32_t i = 0U; i < count; ++i) {
            words.SetValue(i, values[i]);
        }
        GlobalTensor<uint32_t> destination;
        destination.SetGlobalBuffer(dst);
        SyncFunc<HardEvent::S_MTE3>();
        DataCopyPad(destination, words,
                    DataCopyExtParams{1U, static_cast<uint32_t>(count * sizeof(uint32_t)), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }

    __aicore__ inline void StoreWords(__gm__ uint8_t *dst, const uint32_t *values, uint32_t count)
    {
        auto words = copyUb_->template ReinterpretCast<uint32_t>();
        for (uint32_t i = 0U; i < count; ++i) {
            words.SetValue(i, values[i]);
        }
        GlobalTensor<uint8_t> destination;
        destination.SetGlobalBuffer(dst);
        SyncFunc<HardEvent::S_MTE3>();
        DataCopyPad(destination, *copyUb_,
                    DataCopyExtParams{1U, static_cast<uint32_t>(count * sizeof(uint32_t)), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }

    __aicore__ inline uint32_t LoadWord(__gm__ uint32_t *src)
    {
        GlobalTensor<uint32_t> source;
        source.SetGlobalBuffer(src);
        auto words = copyUb_->template ReinterpretCast<uint32_t>();
        DataCopyPad(words, source, DataCopyExtParams{1U, sizeof(uint32_t), 0U, 0U, 0U},
                    DataCopyPadExtParams<uint32_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_S>();
        return words.GetValue(0);
    }

    __aicore__ inline void LoadPeerBytes(__gm__ uint8_t *src, uint32_t bytes)
    {
        GlobalTensor<uint8_t> srcTensor;
        srcTensor.SetGlobalBuffer(src);
        const DataCopyExtParams params{1U, bytes, 0U, 0U, 0U};
        const DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
        DataCopyPad(*copyUb_, srcTensor, params, padParams);
        SyncFunc<HardEvent::MTE2_S>();
    }

    __aicore__ inline void CopyLocalBytes(__gm__ uint8_t *dst, __gm__ uint8_t *src, uint32_t bytes)
    {
        for (uint32_t offset = 0; offset < bytes;) {
            const uint32_t count = min(kPeerCopyTileBytes, bytes - offset);
            GlobalTensor<uint8_t> srcTensor;
            srcTensor.SetGlobalBuffer(src + offset);
            const DataCopyExtParams params{1U, count, 0U, 0U, 0U};
            const DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
            DataCopyPad(*copyUb_, srcTensor, params, padParams);
            SyncFunc<HardEvent::MTE2_V>();
            GlobalTensor<uint8_t> dstTensor;
            dstTensor.SetGlobalBuffer(dst + offset);
            DataCopyPad(dstTensor, *copyUb_, params);
            SyncFunc<HardEvent::MTE3_S>();
            offset += count;
        }
    }

    // 为每个目标 server 在 RDMA Window 的 dispatch 批次区域（batch）开头构建一个 32B 的头部元数据
    __aicore__ inline void BuildBatchMeta()
    {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = ActiveAivNum();
        // 按 server 分核
        for (uint32_t destinationServer = aiv; destinationServer < info_.serverNum; destinationServer += aivNum) {
            /*
             * batch
                ├── DispatchBatchMeta        (32B 头部)
                ├── PacketMeta[packetCount]  (每个 packet 的槽位/序号/member 范围)
                ├── 32B padding
                ├── TokenRecord[packetCount] (token 数据 h + topk 路由记录)
                ├── 32B padding
                └── MemberMeta[memberCount]  (expertId/expertRow/topkSlot)
             */
            auto *batch = localOut_ + BatchOffset(destinationServer);
            const uint32_t packets = static_cast<uint32_t>(plan_[header_->packetCountOffset + destinationServer]);
            const uint32_t members =
                static_cast<uint32_t>(plan_[header_->dispatchMemberCountOffset + destinationServer]);
            const uint32_t tokenOffset = AlignUp(sizeof(DispatchBatchMeta) + packets * sizeof(DispatchPacketMeta));
            const uint32_t memberOffset = AlignUp(tokenOffset + packets * info_.recordStride);
            const uint32_t headerWords[8] = {packets, members, sizeof(DispatchBatchMeta), tokenOffset, memberOffset, 0U,
                                             0U,      0U};
            StoreWords(batch, headerWords, 8U);
        }
    }

    // 并行填充每个 dispatch batch 的实际内容
    __aicore__ inline void BuildRecordsParallel()
    {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = ActiveAivNum();
        /* [targetServer][token slot] */
        const uint64_t packets = static_cast<uint64_t>(info_.serverNum) * info_.slotsPerServer;
        const uint64_t begin = packets * aiv / aivNum;
        const uint64_t end = packets * (aiv + 1U) / aivNum;
        for (uint64_t index = begin; index < end; ++index) {
            const uint32_t destinationServer = static_cast<uint32_t>(index / info_.slotsPerServer);
            const uint32_t slot = static_cast<uint32_t>(index % info_.slotsPerServer);
            BuildDispatchRecord(destinationServer, slot);
        }
    }

    __aicore__ inline void BuildDispatchRecord(uint32_t destinationServer, uint32_t slot)
    {
        using namespace CamMoeLayered;
        const uint32_t packetIndex = PacketIndex(destinationServer, slot);
        const auto *packetDesc = (__gm__ SourcePacketDesc *)(plan_ + header_->packetDescOffset) + packetIndex;
        const int32_t memberCount = packetDesc->memberCount;
        // 该token不需要发送给该server
        if (memberCount <= 0) {
            return;
        }
        const int32_t sourceToken = packetDesc->sourceToken;
        /* 本地out位置
         * batch
            ├── DispatchBatchMeta
            ├── PacketMeta[0 .. packetCount - 1]
            ├── 32B padding
            ├── TokenRecord[0 .. packetCount - 1]
            ├── 32B padding
            └── MemberMeta[0 .. memberCount - 1]
         */
        const uint32_t ordinal = static_cast<uint32_t>(packetDesc->dispatchOrdinal);  // dispatch时的本地紧凑偏移
        auto *batch = localOut_ + BatchOffset(destinationServer);
        const uint32_t packets = PacketCount(destinationServer);
        const uint32_t packetMetaOffset = sizeof(DispatchBatchMeta);
        const uint32_t tokenRecordOffset = AlignUp(packetMetaOffset + packets * sizeof(DispatchPacketMeta));
        const uint32_t memberMetaOffset = AlignUp(tokenRecordOffset + packets * info_.recordStride);
        // 写 PacketMeta
        auto *packetMeta =
            (__gm__ uint32_t *)(batch + packetMetaOffset + static_cast<uint64_t>(ordinal) * sizeof(DispatchPacketMeta));
        const uint32_t packetWords[3] = {static_cast<uint32_t>(sourceToken),
                                         static_cast<uint32_t>(packetDesc->memberBegin),
                                         static_cast<uint32_t>(memberCount)};
        StoreWords(packetMeta, packetWords, 3U);

        // 搬 token 数据
        auto *record = batch + tokenRecordOffset + static_cast<uint64_t>(ordinal) * info_.recordStride;
        auto *sourceX = (__gm__ uint8_t *)(x_ + static_cast<uint64_t>(sourceToken) * info_.h);
        CopyLocalBytes(record, sourceX, info_.h * static_cast<uint32_t>(sizeof(T)));
        // 搬 top-k ids 和 topk weights
        auto *topkRoutes = (__gm__ DispatchTopkRouteRecord *)(record + DispatchTopkRouteOffset(info_.h, sizeof(T)));
        for (uint32_t topkSlot = 0; topkSlot < info_.k; ++topkSlot) {
            const uint32_t expert =
                static_cast<uint32_t>(topk_[static_cast<uint64_t>(sourceToken) * info_.k + topkSlot]);
            union {
                float value;
                int32_t bits;
            } weight{};
            weight.value = weights_[static_cast<uint64_t>(sourceToken) * info_.k + topkSlot];
            const uint32_t routeWords[2] = {expert, static_cast<uint32_t>(weight.bits)};
            StoreWords((__gm__ uint32_t *)(topkRoutes + topkSlot), routeWords, 2U);
        }

        // 写 MemberMeta
        auto *members = (__gm__ DispatchMemberMeta *)(batch + memberMetaOffset) + packetDesc->memberBegin;
        for (int32_t member = 0; member < memberCount; ++member) {
            const uint32_t memberBase =
                header_->memberDescOffset + (packetIndex * info_.k + static_cast<uint32_t>(member)) * MemberElements();
            const uint32_t memberWords[3] = {static_cast<uint32_t>(plan_[memberBase]),
                                             static_cast<uint32_t>(plan_[memberBase + 1U]),
                                             static_cast<uint32_t>(plan_[memberBase + 2U])};
            StoreWords((__gm__ uint32_t *)(members + member), memberWords, 3U);
        }
    }

    __aicore__ inline uint64_t ScratchStride() const
    {
        /* 发送进度[server]、接收进度[server]、rank消费进度[src×proxy] */
        const uint64_t words =
            static_cast<uint64_t>(info_.serverNum) * (static_cast<uint64_t>(info_.ranksPerServer) + 2U);
        return (words + 15U) / 16U * 16U;
    }

    __aicore__ inline __gm__ uint32_t *Scratch(uint32_t core) const
    {
        return scratch_ + static_cast<uint64_t>(core) * ScratchStride();
    }

    __aicore__ inline void InitPipelineScratch()
    {
        auto *state = Scratch(GetBlockIdx());
        const uint32_t zeros[16] = {};
        for (uint64_t i = 0; i < ScratchStride(); i += 16U) {
            StoreWords(state + i, zeros, 16U);
        }
    }

    // 判断核是否足够
    __aicore__ inline bool DedicatedRoles() const
    {
        return static_cast<uint64_t>(ActiveAivNum()) >= 2ULL * info_.serverNum + info_.ranksPerServer;
    }

    __aicore__ inline uint32_t OwnerWorkerCount(uint32_t proxy) const
    {
        if (!DedicatedRoles()) {
            return 1U;
        }
        const uint32_t ownerCores = ActiveAivNum() - 2U * info_.serverNum;
        return (ownerCores + info_.ranksPerServer - 1U - proxy) / info_.ranksPerServer;
    }

    __aicore__ inline uint32_t OwnerWorkerCore(uint32_t proxy, uint32_t worker) const
    {
        const uint32_t core = 2U * info_.serverNum + proxy + worker * info_.ranksPerServer;
        return DedicatedRoles() ? core : core % ActiveAivNum();
    }

    __aicore__ inline bool HasFlag(uint64_t offset, Phase phase) const
    {
        auto *flag = (__gm__ uint64_t *)((__gm__ uint8_t *)comm_.WindowIn(info_.epRankId) + offset);
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer((__gm__ uint8_t *)flag);
        DataCopyPad(*copyUb_, source, DataCopyExtParams{1U, sizeof(uint64_t), 0U, 0U, 0U},
                    DataCopyPadExtParams<uint8_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_S>();
        return copyUb_->template ReinterpretCast<uint64_t>().GetValue(0) == ComposeFlag(epoch_, phase);
    }

    __aicore__ inline uint64_t SourcePipelineAddress(uint32_t sourceServer, uint32_t targetServer, uint32_t slot) const
    {
        return DispatchSourcePipelineFlagOffset(layout_, info_.serverNum, info_.ranksPerServer, sourceServer,
                                                targetServer, slot);
    }

    __aicore__ inline uint64_t OwnerPipelineAddress(uint32_t sourceServer, uint32_t proxy, uint32_t slot) const
    {
        return DispatchOwnerPipelineFlagOffset(layout_, info_.ranksPerServer, sourceServer, proxy, slot);
    }

    __aicore__ inline void SendFlagToRank(uint32_t rank, uint64_t offset, Phase phase)
    {
        auto *local = (__gm__ uint64_t *)(localOut_ + offset);
        CopyPeerFlag(local, ComposeFlag(epoch_, phase));
        auto *remote = (__gm__ uint64_t *)((__gm__ uint8_t *)comm_.WindowIn(rank) + offset);
        if (rank == info_.epRankId) {
            CopyPeerFlag(remote, ComposeFlag(epoch_, phase));
        } else {
            AscendC::PipeBarrier<PIPE_ALL>();
            (void)comm_.PostReadyFlag((GM_ADDR)local, (GM_ADDR)remote, rank, *doorbellUb_, *headUb_, 0U);
        }
    }

    __aicore__ inline void WriteBatchRange(uint32_t rank, __gm__ uint8_t *dst, __gm__ uint8_t *src, uint32_t bytes)
    {
        if (bytes == 0U) {
            return;
        }
        if (rank == info_.epRankId) {
            CopyPeerBytes(dst, src, bytes);
        } else {
            (void)comm_.PostRdmaSend(src, dst, rank, bytes, *doorbellUb_, *headUb_, 0U);
        }
    }

    __aicore__ inline uint32_t PacketCount(uint32_t destinationServer) const
    {
        return static_cast<uint32_t>(plan_[header_->packetCountOffset + destinationServer]);
    }

    __aicore__ inline uint32_t ChunkCount(uint32_t packets) const
    {
        return packets == 0U ? 0U : (packets - 1U) / info_.dispatchChunkPackets + 1U;
    }

    __aicore__ inline void SendMetadata(uint32_t destinationServer)
    {
        const uint32_t rank = destinationServer * info_.ranksPerServer + info_.rankInServer;  // 同号rank
        auto *batch = localOut_ + BatchOffset(destinationServer);
        const uint32_t packets = PacketCount(destinationServer);
        const uint32_t members = static_cast<uint32_t>(plan_[header_->dispatchMemberCountOffset + destinationServer]);
        const uint32_t tokenOffset = AlignUp(sizeof(DispatchBatchMeta) + packets * sizeof(DispatchPacketMeta));
        const uint32_t memberOffset = AlignUp(tokenOffset + packets * info_.recordStride);
        const uint64_t bytes = DispatchBatchPayloadBytes(packets, members, info_.recordStride);
        const uint64_t ingress = IngressOffset(destinationServer, true);
        if (bytes != 0U) {
            auto *remote = (__gm__ uint8_t *)comm_.WindowIn(rank) + DispatchPayloadAddress(layout_, ingress);
            if (ChunkCount(packets) <= 1U) {
                /* 单chunk，一次将数据全部写过去 */
                WriteBatchRange(rank, remote, batch, static_cast<uint32_t>(bytes));
            } else {
                /* 多chunk，只写 [头+packetMeta] 和 [memberMeta] 两段 */
                WriteBatchRange(rank, remote, batch, tokenOffset);
                WriteBatchRange(rank, remote + memberOffset, batch + memberOffset,
                                members * sizeof(DispatchMemberMeta));
            }
        }
        PublishDispatchReady(destinationServer, rank); /* DISPATCH_DATA_READY */
    }

    __aicore__ inline void SendChunk(uint32_t destinationServer, uint32_t chunk)
    {
        const uint32_t rank = destinationServer * info_.ranksPerServer + info_.rankInServer;
        auto *batch = localOut_ + BatchOffset(destinationServer);
        const uint32_t packets = PacketCount(destinationServer);
        if (ChunkCount(packets) > 1U) {
            const uint32_t begin = chunk * info_.dispatchChunkPackets;
            const uint32_t end = min(packets, begin + info_.dispatchChunkPackets);
            const uint64_t ingress = IngressOffset(destinationServer, true);
            /* 计算chunk的偏移 */
            const uint32_t tokenOffset = AlignUp(sizeof(DispatchBatchMeta) + packets * sizeof(DispatchPacketMeta));
            const uint64_t offset = tokenOffset + static_cast<uint64_t>(begin) * info_.recordStride;
            auto *dst = (__gm__ uint8_t *)comm_.WindowIn(rank) + DispatchPayloadAddress(layout_, ingress) + offset;
            WriteBatchRange(rank, dst, batch + offset, (end - begin) * info_.recordStride);
        }
        SendFlagToRank(rank, SourcePipelineAddress(sourceServer_, destinationServer, chunk),
                       Phase::DISPATCH_CHUNK_READY); /* DISPATCH_CHUNK_READY */
    }

    __aicore__ inline bool TrySendServer(uint32_t server, __gm__ uint32_t *state)
    {
        /*
         * bit31: inbound reusable ACK
         * bit30: outbound reusable ACK
         * bit0-28: 0x1fffffff : 发送进度
         */
        constexpr uint32_t OUTBOUND_ACK = 0x40000000U;
        constexpr uint32_t INBOUND_ACK = 0x80000000U;
        uint32_t current = LoadWord(state + server);
        const uint32_t chunks = ChunkCount(PacketCount(server));  // 切分多少个chunk
        const uint32_t next = current & 0x1fffffffU;              // 获取发送进度
        if (next <= chunks) {
            if (next == 0U) {
                /* 把元数据（头部 packetMeta + memberMeta；单 chunk 时连 tokenRecord 一起）写到对端 ingress，并置
                 * DISPATCH_DATA_READY */
                SendMetadata(server);
            } else {
                /* 把第 next-1 个 chunk 的 tokenRecord 段写到对端，并置 DISPATCH_CHUNK_READY */
                SendChunk(server, next - 1U);
            }
            current = (current & 0xc0000000U) | (next + 1U);  // 保留两个ack位，推进
            StoreWords(state + server, &current, 1U);
        }
        if ((current & INBOUND_ACK) == 0U &&
            HasFlag(SourcePipelineAddress(server, sourceServer_, layout_.dispatchMaxChunks + 1U),
                    Phase::DISPATCH_REVERSE_DONE)) {  // 等本地元数据全部转发完： DISPATCH_REVERSE_DONE
            bool consumed = true;
            for (uint32_t owner = 0U; owner < info_.ranksPerServer;
                 ++owner) {  // 判断本 server 内 所有 rank 都已 DISPATCH_REUSABLE
                consumed &= HasFlag(OwnerPipelineAddress(server, owner, layout_.dispatchMaxChunks + 1U),
                                    Phase::DISPATCH_REUSABLE);
            }
            if (consumed) {
                /* 向对端同号 rank 发 DISPATCH_REUSABLE */
                const uint32_t rank = server * info_.ranksPerServer + info_.rankInServer;
                SendFlagToRank(rank, SourcePipelineAddress(server, sourceServer_, layout_.dispatchMaxChunks),
                               Phase::DISPATCH_REUSABLE);
                current |= INBOUND_ACK;
                StoreWords(state + server, &current, 1U);
            }
        }
        if ((current & OUTBOUND_ACK) == 0U && next > chunks &&
            HasFlag(SourcePipelineAddress(sourceServer_, server, layout_.dispatchMaxChunks),
                    Phase::DISPATCH_REUSABLE)) { /* 等待对方回 DISPATCH_REUSABLE */
            current |= OUTBOUND_ACK;
            StoreWords(state + server, &current, 1U);
        }
        /* 发送核任务是否完成 */
        return (current & (OUTBOUND_ACK | INBOUND_ACK)) == (OUTBOUND_ACK | INBOUND_ACK);
    }

    __aicore__ inline void DrainOutbound()
    {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t aivNum = ActiveAivNum();
        for (uint32_t server = 0U; server < info_.serverNum; ++server) {
            const uint32_t sendCore = DedicatedRoles() ? server : server % aivNum;
            if (aiv != sendCore) {
                continue;
            }
            const uint32_t rank = server * info_.ranksPerServer + info_.rankInServer;
            comm_.Drain(rank, 0U);
        }
    }

    __aicore__ inline void PublishOwnerPipelineFlag(uint32_t sourceServer, uint32_t slot, Phase phase)
    {
        for (uint32_t owner = 0U; owner < info_.ranksPerServer; ++owner) {
            const uint32_t rank = sourceServer_ * info_.ranksPerServer + owner;
            auto *flag = (__gm__ uint64_t *)((__gm__ uint8_t *)comm_.WindowIn(rank) +
                                             OwnerPipelineAddress(sourceServer, info_.rankInServer, slot));
            CopyPeerFlag(flag, ComposeFlag(epoch_, phase));
        }
    }

    __aicore__ inline bool TryReceiveSource(uint32_t sourceServer, __gm__ uint32_t *state)
    {
        uint32_t next = LoadWord(state + info_.serverNum + sourceServer);
        auto *in = (__gm__ uint8_t *)comm_.WindowIn(info_.epRankId);
        const auto p = MakePlanLayout(info_.bs, info_.maxBs, info_.k, info_.moeExpertNum, info_.epWorldSize,
                                      info_.serverNum, info_.ranksPerServer, 1U, info_.maxBs);
        if (next == 0U) {
            if (!HasFlag(DispatchReadyAddress(sourceServer, sourceServer_),
                         Phase::DISPATCH_DATA_READY)) {  // 等待 DISPATCH_DATA_READY
                return false;
            }
            const uint32_t sourceRank = sourceServer * info_.ranksPerServer + info_.rankInServer;
            auto *control = in + ControlLaneOffset(layout_, info_.epWorldSize, sourceRank, sourceServer_);
            LoadPeerBytes(control, sizeof(CountControlMeta));
            const uint32_t returnBase = copyUb_->template ReinterpretCast<uint32_t>().GetValue(0);
            StoreWords((__gm__ uint32_t *)(plan_ + p.reverseReturnBase + sourceServer), &returnBase, 1U);
            auto *entry = control + CountControlDataOffset();
            LoadPeerBytes(entry, sizeof(uint32_t));
            const uint32_t expectedPackets = copyUb_->template ReinterpretCast<uint32_t>().GetValue(0);
            if (expectedPackets == 0U) {  // 源 server 无 token 发来
                const uint32_t zero = 0U;
                StoreWords((__gm__ uint32_t *)(plan_ + p.reverseCount + sourceServer), &zero, 1U);
                for (uint32_t owner = 0U; owner < info_.ranksPerServer; ++owner) {  // 给每个本地rank写空的meta
                    WriteMetadataHeader(MetadataOwnerAddress(sourceServer, owner), 0U, 0U,
                                        IngressOffset(sourceServer, false));
                }
                PublishOwnerPipelineFlag(sourceServer, 0U,
                                         Phase::DISPATCH_METADATA_READY); /* 给本地rank发送DISPATCH_METADATA_READY */
                auto *done = (__gm__ uint64_t *)(in + SourcePipelineAddress(sourceServer, sourceServer_,
                                                                            layout_.dispatchMaxChunks + 1U));
                CopyPeerFlag(done,
                             ComposeFlag(epoch_, Phase::DISPATCH_REVERSE_DONE));  // 写本地：标记DISPATCH_REVERSE_DONE
                const uint32_t initial = 1U;
                StoreWords(state + info_.serverNum + sourceServer, &initial, 1U);
                return true;
            }
            // 读取收到的元数据
            auto *batch = in + DispatchPayloadAddress(layout_, IngressOffset(sourceServer, false));
            LoadPeerBytes(batch, sizeof(DispatchBatchMeta));
            auto words = copyUb_->template ReinterpretCast<uint32_t>();
            DispatchBatchMeta meta{};
            meta.packetCount = words.GetValue(0);
            meta.memberCount = words.GetValue(1);
            meta.packetMetaOffset = words.GetValue(2);
            meta.tokenRecordOffset = words.GetValue(3);
            meta.memberMetaOffset = words.GetValue(4);
            StoreWords((__gm__ uint32_t *)(plan_ + p.reverseCount + sourceServer), &meta.packetCount, 1U);
            for (uint32_t owner = 0U; owner < info_.ranksPerServer; ++owner) {
                FanoutMetadataForOwnerGroup(sourceServer, owner, batch, meta, 0U,
                                            1U);  // 收到跨server的元数据后，给本地每个rank转发
            }
            PublishOwnerPipelineFlag(sourceServer, 0U,
                                     Phase::DISPATCH_METADATA_READY);  // 给本地rank发送DISPATCH_METADATA_READY
            const uint32_t initial = 1U;
            StoreWords(state + info_.serverNum + sourceServer, &initial, 1U);
            return false;
        }
        const uint32_t packets = LoadWord((__gm__ uint32_t *)(plan_ + p.reverseCount + sourceServer));
        const uint32_t chunks = ChunkCount(packets);  // 轮次数
        if (next > chunks) {
            return true;
        }
        const uint32_t chunk = next - 1U;
        if (!HasFlag(SourcePipelineAddress(sourceServer, sourceServer_, chunk),
                     Phase::DISPATCH_CHUNK_READY)) {  // 等待 DISPATCH_CHUNK_READY
            return false;
        }
        // 读取收到的元数据
        auto *batch = in + DispatchPayloadAddress(layout_, IngressOffset(sourceServer, false));
        LoadPeerBytes(batch, sizeof(DispatchBatchMeta));
        auto words = copyUb_->template ReinterpretCast<uint32_t>();
        DispatchBatchMeta meta{};
        meta.packetCount = words.GetValue(0);
        meta.memberCount = words.GetValue(1);
        meta.packetMetaOffset = words.GetValue(2);
        meta.tokenRecordOffset = words.GetValue(3);
        meta.memberMetaOffset = words.GetValue(4);
        PublishOwnerPipelineFlag(sourceServer, chunk + 1U,
                                 Phase::DISPATCH_CHUNK_READY);  // 通知server内每个rank，该chunk ready
        const uint32_t begin = chunk * info_.dispatchChunkPackets;
        const uint32_t end = min(packets, begin + info_.dispatchChunkPackets);
        PersistReverseBatch(sourceServer, batch, meta, begin, end);  // 建立反查项
        ++next;
        StoreWords(state + info_.serverNum + sourceServer, &next, 1U);
        if (next > chunks) {
            auto *done = (__gm__ uint64_t *)(in + SourcePipelineAddress(sourceServer, sourceServer_,
                                                                        layout_.dispatchMaxChunks + 1U));
            CopyPeerFlag(
                done,
                ComposeFlag(epoch_, Phase::DISPATCH_REVERSE_DONE));  // 写本地 DISPATCH_REVERSE_DONE，数据通知全部完成
        }
        return next > chunks;
    }

    __aicore__ inline bool TryConsumeOwner(uint32_t sourceServer, uint32_t proxy, uint32_t worker, uint32_t workers,
                                           __gm__ uint32_t *state)
    {
        const uint64_t index =
            2ULL * info_.serverNum + static_cast<uint64_t>(sourceServer) * info_.ranksPerServer + proxy;
        /*
         * 0: 未开始
         * 1...chunks：已消费 chunk next-1
         * 0xffffffff：所有chunk均消费完
         * 0xfffffffe：首个AIV发布 DISPATCH_REUSABLE，表示本 rank 处理完成
         */
        uint32_t next = LoadWord(state + index);
        if (next == 0xfffffffeU) {
            return true;
        }
        if (next == 0xffffffffU) {
            if (worker != 0U) {  // 第一个AIV负责汇总，其他直接返回
                return true;
            }
            for (uint32_t peerWorker = 1U; peerWorker < workers; ++peerWorker) {  // 其他AIV也全部完成
                auto *peer = Scratch(OwnerWorkerCore(proxy, peerWorker)) + index;
                if (LoadWord(peer) != 0xffffffffU) {
                    return false;
                }
            }
            const uint32_t proxyRank = sourceServer_ * info_.ranksPerServer + proxy;
            auto *flag = (__gm__ uint64_t *)((__gm__ uint8_t *)comm_.WindowIn(proxyRank) +
                                             OwnerPipelineAddress(sourceServer, info_.rankInServer,
                                                                  layout_.dispatchMaxChunks + 1U));
            CopyPeerFlag(
                flag,
                ComposeFlag(epoch_, Phase::DISPATCH_REUSABLE));  // 将 DISPATCH_REUSABLE 通知给proxy，数据全部拉取完成
            const uint32_t notified = 0xfffffffeU;
            StoreWords(state + index, &notified, 1U);
            return true;
        }
        if (!HasFlag(OwnerPipelineAddress(sourceServer, proxy, 0U),
                     Phase::DISPATCH_METADATA_READY)) {  // 等待元数据到达
            return false;
        }
        auto *in = (__gm__ uint8_t *)comm_.WindowIn(info_.epRankId);
        auto *batch = in + DispatchMetadataFanoutOffset(layout_, info_.ranksPerServer, sourceServer, proxy);
        LoadPeerBytes(batch, sizeof(DispatchBatchMeta));
        auto words = copyUb_->template ReinterpretCast<uint32_t>();
        const uint32_t packets = words.GetValue(0);
        const uint32_t packetOffset = words.GetValue(2);
        const uint32_t memberOffset = words.GetValue(4);
        const uint32_t ingressOffsetLo = words.GetValue(DispatchIngressOffsetLoWord());
        const uint32_t ingressOffsetHi = words.GetValue(DispatchIngressOffsetHiWord());
        const uint64_t ingress =
            static_cast<uint64_t>(ingressOffsetLo) | (static_cast<uint64_t>(ingressOffsetHi) << 32U);
        const uint32_t chunks = ChunkCount(packets);  // 获取chunk数
        if (next == 0U) {
            const uint32_t initial = 1U;
            StoreWords(state + index, &initial, 1U);
            return false;
        }
        if (next > chunks) {
            const uint32_t complete = 0xffffffffU;
            StoreWords(state + index, &complete, 1U);
            return false;
        }
        const uint32_t chunk = next - 1U;
        if (!HasFlag(OwnerPipelineAddress(sourceServer, proxy, chunk + 1U),
                     Phase::DISPATCH_CHUNK_READY)) {  // 等待该chunk数据就绪
            return false;
        }
        // 计算本 worker 负责的 packet 子区间
        const uint32_t begin = chunk * info_.dispatchChunkPackets;
        const uint32_t end = min(packets, begin + info_.dispatchChunkPackets);
        const uint32_t myBegin = begin + static_cast<uint64_t>(end - begin) * worker / workers;
        const uint32_t myEnd = begin + static_cast<uint64_t>(end - begin) * (worker + 1U) / workers;
        const uint32_t proxyRank = sourceServer * info_.ranksPerServer + proxy;
        auto *proxyBatch = (__gm__ uint8_t *)comm_.WindowIn(proxyRank) + DispatchPayloadAddress(layout_, ingress);
        const uint32_t tokenOffset = AlignUp(packetOffset + packets * sizeof(DispatchPacketMeta));
        ConsumeMetadataLaneSlice(batch, packetOffset, memberOffset, proxyBatch, tokenOffset, sourceServer, proxy,
                                 myBegin, myEnd);  // 拷贝实际token
        const uint32_t advanced = next + 1U;
        StoreWords(state + index, &advanced, 1U);
        return false;
    }

    __aicore__ inline void RunPipeline()
    {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t n = info_.serverNum;
        const uint32_t aivNum = ActiveAivNum();
        auto *state = Scratch(aiv);  // 状态区
        while (true) {
            bool done = true;  // 所有角色是否都完成
            for (uint32_t server = 0U; server < n; ++server) {
                const uint32_t sendCore = DedicatedRoles() ? server : server % aivNum; /* 每个server一个发送核 */
                const uint32_t recvCore =
                    DedicatedRoles() ? n + server : (n + server) % aivNum; /* 每个server一个接收核 */
                if (aiv == sendCore) {
                    done &= TrySendServer(server, state);  // 跨server发送
                }
                if (aiv == recvCore) {
                    done &= TryReceiveSource(server, state);  // 跨server接收
                }
            }
            for (uint32_t proxy = 0U; proxy < info_.ranksPerServer; ++proxy) {
                const uint32_t workers = OwnerWorkerCount(proxy);  // 计算有效核数
                for (uint32_t worker = 0U; worker < workers; ++worker) {
                    if (aiv != OwnerWorkerCore(proxy, worker)) {  // 按照rank分组
                        continue;
                    }
                    for (uint32_t source = 0U; source < n; ++source) {  // 组内每个核均处理
                        done &= TryConsumeOwner(source, proxy, worker, workers, state);
                    }
                }
            }
            if (done) {  // 完成
                break;
            }
            AscendC::GetSystemCycle();
        }
    }

    __aicore__ inline void PublishDispatchReady(uint32_t destinationServer, uint32_t dstRank)
    {
        using namespace CamMoeLayered;
        const uint64_t readyOffset = DispatchReadyAddress(sourceServer_, destinationServer);
        auto *localFlag = (__gm__ uint64_t *)(localOut_ + readyOffset);
        CopyPeerFlag(localFlag, ComposeFlag(epoch_, Phase::DISPATCH_DATA_READY));
        auto *remoteFlag = (__gm__ uint64_t *)((__gm__ uint8_t *)comm_.WindowIn(dstRank) + readyOffset);
        if (destinationServer == sourceServer_) {
            CopyPeerFlag(remoteFlag, ComposeFlag(epoch_, Phase::DISPATCH_DATA_READY));
            return;
        } else {
            AscendC::PipeBarrier<PIPE_ALL>();
            (void)comm_.PostReadyFlag((GM_ADDR)localFlag, (GM_ADDR)remoteFlag, dstRank, *doorbellUb_, *headUb_, 0U);
        }
    }

    __aicore__ inline uint64_t DispatchMetadataAddress(uint32_t sourceServer) const
    {
        return DispatchMetadataFanoutOffset(layout_, info_.ranksPerServer, sourceServer, info_.rankInServer);
    }

    __aicore__ inline void WriteMetadataHeader(__gm__ uint8_t *dst, uint32_t packetCount, uint32_t memberCount,
                                               uint64_t ingressOffset)
    {
        auto words = copyUb_->template ReinterpretCast<uint32_t>();
        words.SetValue(0, packetCount);                // packet数量
        words.SetValue(1, memberCount);                // member数量
        words.SetValue(2, sizeof(DispatchBatchMeta));  // packetMeta的偏移
        words.SetValue(3, 0U);                         // tokenRecord 偏移
        const uint32_t packetOffset = sizeof(DispatchBatchMeta);
        const uint32_t memberOffset = AlignUp(packetOffset + packetCount * sizeof(DispatchPacketMeta));
        words.SetValue(4, memberOffset);  // memberMeta的偏移
        words.SetValue(5, 0U);
        words.SetValue(DispatchIngressOffsetLoWord(), static_cast<uint32_t>(ingressOffset));
        words.SetValue(DispatchIngressOffsetHiWord(), static_cast<uint32_t>(ingressOffset >> 32U));
        GlobalTensor<uint8_t> dstTensor;
        dstTensor.SetGlobalBuffer(dst);
        SyncFunc<HardEvent::S_MTE3>();
        DataCopyPad(dstTensor, *copyUb_, DataCopyExtParams{1U, sizeof(DispatchBatchMeta), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }

    __aicore__ inline __gm__ uint8_t *MetadataOwnerAddress(uint32_t sourceServer, uint32_t ownerInServer)
    {
        const uint32_t ownerRank = sourceServer_ * info_.ranksPerServer + ownerInServer;
        return (__gm__ uint8_t *)comm_.WindowIn(ownerRank) + DispatchMetadataAddress(sourceServer);
    }

    __aicore__ inline void WriteMetadataHeaderToOwner(uint32_t sourceServer, uint32_t ownerInServer,
                                                      const DispatchBatchMeta &ingressMeta)
    {
        WriteMetadataHeader(MetadataOwnerAddress(sourceServer, ownerInServer), ingressMeta.packetCount,
                            ingressMeta.memberCount, IngressOffset(sourceServer, false));
    }

    __aicore__ inline void CopyPacketMetaRangeToOwner(uint32_t sourceServer, uint32_t ownerInServer,
                                                      __gm__ uint8_t *batch, const DispatchBatchMeta &ingressMeta,
                                                      uint32_t begin, uint32_t end)
    {
        if (begin >= end) {
            return;
        }
        const uint32_t bytes = (end - begin) * sizeof(DispatchPacketMeta);
        auto *dst = MetadataOwnerAddress(sourceServer, ownerInServer) + sizeof(DispatchBatchMeta) +
                    begin * sizeof(DispatchPacketMeta);
        auto *src = batch + ingressMeta.packetMetaOffset + begin * sizeof(DispatchPacketMeta);
        CopyPeerBytes(dst, src, bytes);
    }

    __aicore__ inline void CopyMemberMetaRangeToOwner(uint32_t sourceServer, uint32_t ownerInServer,
                                                      __gm__ uint8_t *batch, const DispatchBatchMeta &ingressMeta,
                                                      uint32_t begin, uint32_t end)
    {
        if (begin >= end) {
            return;
        }
        const uint32_t bytes = (end - begin) * sizeof(DispatchMemberMeta);
        const uint32_t memberOffset =
            AlignUp(sizeof(DispatchBatchMeta) + ingressMeta.packetCount * sizeof(DispatchPacketMeta));
        auto *dst =
            MetadataOwnerAddress(sourceServer, ownerInServer) + memberOffset + begin * sizeof(DispatchMemberMeta);
        auto *src = batch + ingressMeta.memberMetaOffset + begin * sizeof(DispatchMemberMeta);
        CopyPeerBytes(dst, src, bytes);
    }

    __aicore__ inline void FanoutMetadataForOwnerGroup(uint32_t sourceServer, uint32_t ownerInServer,
                                                       __gm__ uint8_t *batch, const DispatchBatchMeta &ingressMeta,
                                                       uint32_t localAiv, uint32_t groupSize)
    {
        if (localAiv == 0U) {
            // 搬运header
            WriteMetadataHeaderToOwner(sourceServer, ownerInServer, ingressMeta);
        }
        const uint32_t packetBegin = ingressMeta.packetCount * localAiv / groupSize;
        const uint32_t packetEnd = ingressMeta.packetCount * (localAiv + 1U) / groupSize;
        const uint32_t memberBegin = ingressMeta.memberCount * localAiv / groupSize;
        const uint32_t memberEnd = ingressMeta.memberCount * (localAiv + 1U) / groupSize;
        // 搬运 packet meta
        CopyPacketMetaRangeToOwner(sourceServer, ownerInServer, batch, ingressMeta, packetBegin, packetEnd);
        // 搬运 member meta
        CopyMemberMetaRangeToOwner(sourceServer, ownerInServer, batch, ingressMeta, memberBegin, memberEnd);
    }

    /*
     * 在目标 server 的 proxy 卡上，把一个源 server 发来的 batch 里的路由信息
     * 解析并持久化成反向索引，供后续 combine 阶段把专家计算结果回送到正确的源 token
     */
    __aicore__ inline void PersistReverseBatch(uint32_t sourceServer, __gm__ uint8_t *batch,
                                               const DispatchBatchMeta &meta, uint32_t packetBegin, uint32_t packetEnd)
    {
        const auto p = MakePlanLayout(info_.bs, info_.maxBs, info_.k, info_.moeExpertNum, info_.epWorldSize,
                                      info_.serverNum, info_.ranksPerServer, 1U, info_.maxBs);
        for (uint32_t ordinal = packetBegin; ordinal < packetEnd; ++ordinal) {
            LoadPeerBytes(batch + meta.packetMetaOffset + ordinal * sizeof(DispatchPacketMeta),
                          sizeof(DispatchPacketMeta));
            auto words = copyUb_->template ReinterpretCast<uint32_t>();
            const uint32_t sourceToken = words.GetValue(0);
            const uint32_t begin = words.GetValue(1);
            const uint32_t count = words.GetValue(2);
            const uint32_t index = sourceServer * info_.maxBs + ordinal;
            /* 建立 PacketDesc 反查项 */
            auto *packet = (__gm__ ReversePacketDesc *)(plan_ + p.reversePacket) + index;
            const uint32_t packetWords[2] = {sourceServer * info_.ranksPerServer + info_.rankInServer, sourceToken};
            StoreWords((__gm__ uint32_t *)packet, packetWords, 2U);
            /* 建立 ReverseMember 反查项 */
            auto *members = (__gm__ ReverseMember *)(plan_ + p.reverseMember) + index * info_.k;
            for (uint32_t slot = 0U; slot < info_.k; ++slot) {
                const uint32_t invalid = static_cast<uint32_t>(INVALID_SLOT);
                StoreWords((__gm__ uint32_t *)(members + slot), &invalid, 1U);
            }
            for (uint32_t member = 0U; member < count; ++member) {
                LoadPeerBytes(batch + meta.memberMetaOffset + (begin + member) * sizeof(DispatchMemberMeta),
                              sizeof(DispatchMemberMeta));
                auto memberWords = copyUb_->template ReinterpretCast<int32_t>();
                const int32_t expert = memberWords.GetValue(0);
                const int32_t row = memberWords.GetValue(1);
                const int32_t slot = memberWords.GetValue(2);
                LoadPeerBytes(batch + meta.tokenRecordOffset + static_cast<uint64_t>(ordinal) * info_.recordStride +
                                  DispatchTopkRouteOffset(info_.h, sizeof(T)) + slot * sizeof(DispatchTopkRouteRecord),
                              sizeof(DispatchTopkRouteRecord));
                const uint32_t weightBits = copyUb_->template ReinterpretCast<uint32_t>().GetValue(1);
                const uint32_t reverseWords[4] = {static_cast<uint32_t>(expert), static_cast<uint32_t>(row),
                                                  static_cast<uint32_t>(slot), weightBits};
                StoreWords((__gm__ uint32_t *)(members + slot), reverseWords, 4U);
            }
        }
    }

    __aicore__ inline void ConsumeMetadataLaneSlice(__gm__ uint8_t *metaBatch, uint32_t packetMetaOffset,
                                                    uint32_t memberMetaOffset, __gm__ uint8_t *proxyBatch,
                                                    uint32_t tokenRecordOffset, uint32_t remoteSourceServer,
                                                    uint32_t sourceRankInServer, uint32_t packetBegin,
                                                    uint32_t packetEnd)
    {
        for (uint32_t ordinal = packetBegin; ordinal < packetEnd; ++ordinal) {
            ConsumeMetadataPacket(metaBatch, packetMetaOffset, memberMetaOffset, proxyBatch, tokenRecordOffset,
                                  ordinal);
        }
    }

    __aicore__ inline void ConsumeMetadataPacket(__gm__ uint8_t *metaBatch, uint32_t packetMetaOffset,
                                                 uint32_t memberMetaOffset, __gm__ uint8_t *proxyBatch,
                                                 uint32_t tokenRecordOffset, uint32_t ordinal)
    {
        LoadPeerBytes(metaBatch + packetMetaOffset + ordinal * sizeof(DispatchPacketMeta), sizeof(DispatchPacketMeta));
        auto packetWords = copyUb_->template ReinterpretCast<uint32_t>();
        const uint32_t memberBegin = packetWords.GetValue(1);
        const uint32_t memberCount = packetWords.GetValue(2);
        if (memberCount == 0U) {  // 空 packet 直接跳过
            return;
        }

        uint32_t compactRows[kMaxTopk];
        const uint32_t localMemberCount =
            CollectLocalPacketMembers(metaBatch, memberMetaOffset, memberBegin, memberCount, compactRows);
        if (localMemberCount == 0U) {  // 不是本 rank 的，跳过
            return;
        }

        auto *tokenRecord = proxyBatch + tokenRecordOffset + static_cast<uint64_t>(ordinal) * info_.recordStride;
        int32_t routeExperts[kMaxTopk];
        int32_t routeWeightBits[kMaxTopk];
        // 获取专家与路由权重
        LoadPacketTopkRoute(tokenRecord, routeExperts, routeWeightBits);
        // 拷贝路由专家权重到recvWeights_
        CopyPacketTopkRouteToRows(compactRows, localMemberCount, routeExperts, routeWeightBits);
        // 拷贝payload到recv_
        CopyPacketPayloadToRows(tokenRecord, compactRows, localMemberCount);
    }

    __aicore__ inline uint32_t CollectLocalPacketMembers(__gm__ uint8_t *metaBatch, uint32_t memberMetaOffset,
                                                         uint32_t memberBegin, uint32_t memberCount,
                                                         uint32_t *compactRows)
    {
        uint32_t localMemberCount = 0U;
        for (uint32_t member = 0U; member < memberCount; ++member) {
            const uint32_t memberIndex = memberBegin + member;
            LoadPeerBytes(metaBatch + memberMetaOffset + memberIndex * sizeof(DispatchMemberMeta),
                          sizeof(DispatchMemberMeta));
            auto memberWords = copyUb_->template ReinterpretCast<int32_t>();
            const int32_t expert = memberWords.GetValue(0);
            const int32_t expertRow = memberWords.GetValue(1);
            const uint32_t ownerInServer = (static_cast<uint32_t>(expert) / localExperts_) % info_.ranksPerServer;
            if (ownerInServer != info_.rankInServer) {  // 不是本 rank 的，跳过
                continue;
            }

            const uint32_t localExpert = static_cast<uint32_t>(expert) % localExperts_;
            const int32_t localOffset = plan_[header_->localExpertOffset + localExpert];
            compactRows[localMemberCount] = static_cast<uint32_t>(localOffset + expertRow);
            ++localMemberCount;
        }
        return localMemberCount;
    }

    __aicore__ inline void LoadPacketTopkRoute(__gm__ uint8_t *record, int32_t *routeExperts, int32_t *routeWeightBits)
    {
        using namespace CamMoeLayered;
        for (uint32_t topkSlot = 0; topkSlot < info_.k; ++topkSlot) {
            LoadPeerBytes(
                record + DispatchTopkRouteOffset(info_.h, sizeof(T)) + topkSlot * sizeof(DispatchTopkRouteRecord),
                sizeof(DispatchTopkRouteRecord));
            auto routeWords = copyUb_->template ReinterpretCast<int32_t>();
            routeExperts[topkSlot] = routeWords.GetValue(0);
            routeWeightBits[topkSlot] = routeWords.GetValue(1);
        }
    }

    __aicore__ inline void CopyPacketTopkRouteToRows(const uint32_t *compactRows, uint32_t localMemberCount,
                                                     const int32_t *routeExperts, const int32_t *routeWeightBits)
    {
        uint32_t expertWords[kMaxTopk];
        uint32_t weightWords[kMaxTopk];
        for (uint32_t topkSlot = 0U; topkSlot < info_.k; ++topkSlot) {
            expertWords[topkSlot] = static_cast<uint32_t>(routeExperts[topkSlot]);
            weightWords[topkSlot] = static_cast<uint32_t>(routeWeightBits[topkSlot]);
        }
        for (uint32_t member = 0U; member < localMemberCount; ++member) {
            const uint64_t rowOffset = static_cast<uint64_t>(compactRows[member]) * info_.k;
            StoreWords((__gm__ uint32_t *)(recvTopk_ + rowOffset), expertWords, info_.k);
            StoreWords((__gm__ uint32_t *)(recvWeights_ + rowOffset), weightWords, info_.k);
        }
    }

    __aicore__ inline void CopyPacketPayloadToRows(__gm__ uint8_t *record, const uint32_t *compactRows,
                                                   uint32_t localMemberCount)
    {
        const uint32_t payloadBytes = info_.h * static_cast<uint32_t>(sizeof(T));
        for (uint32_t byteOffset = 0; byteOffset < payloadBytes; byteOffset += kPeerCopyTileBytes) {
            const uint32_t bytes = min(kPeerCopyTileBytes, payloadBytes - byteOffset);
            LoadPeerBytes(record + byteOffset, bytes);
            for (uint32_t member = 0U; member < localMemberCount; ++member) {
                GlobalTensor<uint8_t> recvTensor;
                recvTensor.SetGlobalBuffer((__gm__ uint8_t *)recv_ +
                                           static_cast<uint64_t>(compactRows[member]) * info_.h * sizeof(T) +
                                           byteOffset);
                DataCopyPad(recvTensor, *copyUb_, DataCopyExtParams{1U, bytes, 0U, 0U, 0U});
                SyncFunc<HardEvent::MTE3_S>();
            }
        }
    }

    __gm__ PlanHeader *header_{nullptr};
    __gm__ int32_t *plan_{nullptr};
    __gm__ T *x_{nullptr};
    __gm__ float *weights_{nullptr};
    __gm__ int32_t *topk_{nullptr};
    __gm__ T *recv_{nullptr};
    __gm__ int32_t *recvTopk_{nullptr};
    __gm__ float *recvWeights_{nullptr};
    __gm__ uint8_t *localOut_{nullptr};
    __gm__ uint32_t *scratch_{nullptr};
    CamMoeLayeredInfo info_{};
    WindowLayout layout_{};
    CommAdapter comm_{};
    LocalTensor<uint64_t> *doorbellUb_{nullptr};
    LocalTensor<uint32_t> *headUb_{nullptr};
    LocalTensor<uint8_t> *copyUb_{nullptr};
    uint64_t epoch_{0};
    uint32_t sourceServer_{0};
    uint32_t localExperts_{0};
};
}  // namespace

extern "C" __global__ __aicore__ void cam_moe_dispatch_normal_layered(GM_ADDR roceContext, GM_ADDR x,
                                                                      GM_ADDR topkWeights, GM_ADDR topkIdx,
                                                                      GM_ADDR layeredPlan, GM_ADDR recvX,
                                                                      GM_ADDR recvTopkIdx, GM_ADDR recvTopkWeights,
                                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CamMoeDispatchNormalLayeredTilingData);
    TPipe pipe;
    // Must match SetTilingKey(13000U) in cam_moe_dispatch_normal_layered_tiling.cc.
    // Declaring the key here is what makes the op compiler emit the _13000
    // funcEntry; without it only the default key-0 entry is generated.
    if (TILING_KEY_IS(13000)) {
#if (ORIG_DTYPE_X == DT_BF16 || ORIG_DTYPE_X == DT_FLOAT16)
        GET_TILING_DATA_WITH_STRUCT(CamMoeDispatchNormalLayeredTilingData, data, tiling);
        TBuf rdmaUb;
        pipe.InitBuffer(rdmaUb, 32U);
        TBuf copyUb;
        pipe.InitBuffer(copyUb, kPeerCopyTileBytes);
#ifdef __DAV_C310__
        TBuf hcommUb;
        pipe.InitBuffer(hcommUb, HCOMM_UB_BUF_SIZE);
#endif
        auto doorbellUb = rdmaUb.GetWithOffset<uint64_t>(1U, 0U);
        auto headUb = rdmaUb.GetWithOffset<uint32_t>(1U, 16U);
        auto copyTensor = copyUb.Get<uint8_t>();
#ifdef __DAV_C310__
        auto hcommTensor = hcommUb.Get<uint8_t>();
#else
        auto hcommTensor = copyTensor;
#endif
        CamMoeDispatchNormalLayered<DTYPE_X> op;
        op.Init(roceContext, x, topkWeights, topkIdx, layeredPlan, recvX, recvTopkIdx, recvTopkWeights, workspace,
                data.layered, doorbellUb, headUb, copyTensor, hcommTensor);
        op.Process();
#endif
    }
}

#endif  // DEEPEP_CANN_VERSION_NUM >= DEEPEP_LAYERED_MIN_CANN_VERSION
