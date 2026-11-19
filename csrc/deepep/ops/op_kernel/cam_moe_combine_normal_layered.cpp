#if DEEPEP_CANN_VERSION_NUM >= DEEPEP_LAYERED_MIN_CANN_VERSION

#include "kernel_operator.h"
#include "cam_moe_combine_normal_layered_tiling.h"
#include "cam_moe_layered_comm.h"
#include "cam_moe_layered_plan.h"

using namespace AscendC;
using namespace CamMoeLayered;

namespace {
constexpr uint32_t kCopyTileBytes = 4096U;
constexpr uint32_t kMaxTopk = 16U;
constexpr uint32_t kMaxServers = 32U;
constexpr uint32_t kProgressStride = static_cast<uint32_t>(COMBINE_FLAG_STRIDE);

template <typename T>
class CamMoeCombineNormalLayered
{
public:
    __aicore__ inline void Init(GM_ADDR roceContext, GM_ADDR recvX, GM_ADDR layeredPlan, GM_ADDR xOut,
                                GM_ADDR workspace, const CamMoeLayeredInfo &tiling, uint32_t ownerCapacity,
                                uint32_t chunkPackets, LocalTensor<uint64_t> &doorbellUb, LocalTensor<uint32_t> &headUb,
                                LocalTensor<uint32_t> &expertBasesUb, LocalTensor<uint8_t> &copyUb,
                                LocalTensor<uint32_t> &progressUb, LocalTensor<float> &partialUb,
                                LocalTensor<float> &weightedUb, LocalTensor<T> &ownerRowUb,
                                LocalTensor<uint8_t> &hcommUb)
    {
        info_ = tiling;
        header_ = (__gm__ PlanHeader *)layeredPlan;
        plan_ = (__gm__ int32_t *)layeredPlan;
        recv_ = (__gm__ T *)recvX;
        progress_ = (__gm__ uint8_t *)workspace;
        chunkPackets_ = chunkPackets;
        out_ = (__gm__ T *)xOut;
        doorbellUb_ = &doorbellUb;
        headUb_ = &headUb;
        expertBasesUb_ = &expertBasesUb;
        copyUb_ = &copyUb;
        progressUb_ = &progressUb;
        partialUb_ = &partialUb;
        weightedUb_ = &weightedUb;
        ownerRowUb_ = &ownerRowUb;
        server_ = info_.epRankId / info_.ranksPerServer;
        localExperts_ = info_.moeExpertNum / info_.epWorldSize;
        const auto base = MakeWindowLayout(info_.moeExpertNum, info_.epWorldSize, info_.ranksPerServer, info_.serverNum,
                                           info_.physicalRounds, info_.slotsPerServer, info_.recordStride,
                                           info_.partialStride, info_.k, info_.isHybridDeployment != 0U);
        layout_ = MakeCombineWindowLayout(base, info_.serverNum, info_.ranksPerServer, localExperts_, ownerCapacity,
                                          info_.realMaxBs, info_.h, info_.elementBytes);
        comm_.Init(info_.epRankId, roceContext, hcommUb, info_.isHybridDeployment != 0U);
        in_ = (__gm__ uint8_t *)comm_.WindowIn(info_.epRankId);
        outWindow_ = (__gm__ uint8_t *)comm_.WindowOut(info_.epRankId);
        epoch_ = info_.epoch;
    }

    __aicore__ inline void Process()
    {
        StageOwnerRowsAndBases();  // 搬运token到本地通信窗口，并构造专家偏移
        SyncAll<true>();
        FanoutExpertBases();  // 专家偏移发送给本地其他rank
        SyncAll<true>();
        if (GetBlockIdx() == 0U) {  // 全卡barrier
            PublishOwnerReady();
            WaitOwnerReady();
            PublishServerReady();
            WaitServerReady();
        }
        SyncAll<true>();

        RunPartialPipeline();  // 发送
        SyncAll<true>();
        if (GetBlockIdx() == 0U) {
            PublishOwnerConsumed();  // proxy已全部读取，对端stage可复用
        }
        WaitSourcePartials();  // 等待全部server
        SyncAll<true>();

        FinalizeOutput();  // 对每个源 token 跨 target server 求和
        SyncAll<true>();
        // 同步 + 释放
        PublishSourceConsumed();
        WaitConsumed();
        DrainOutbound();
    }

private:
    __aicore__ inline uint32_t AivNum() const
    {
        return info_.aivNum;
    }
    __aicore__ inline uint64_t Expected(Phase phase) const
    {
        return ComposeFlag(epoch_, phase);
    }

    __aicore__ inline void WaitFlag(uint64_t offset, Phase phase)
    {
        auto *flag = (__gm__ uint64_t *)(in_ + offset);
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer((__gm__ uint8_t *)flag);
        auto words = copyUb_->template ReinterpretCast<uint64_t>();
        while (true) {
            cacheReadInvalidate((__gm__ uint8_t *)flag, sizeof(uint64_t));
            DataCopyPad(*copyUb_, source, DataCopyExtParams{1U, sizeof(uint64_t), 0U, 0U, 0U},
                        DataCopyPadExtParams<uint8_t>{false, 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE2_S>();
            if (words.GetValue(0) == Expected(phase)) {
                break;
            }
            AscendC::GetSystemCycle();
        }
    }

    __aicore__ inline void CopyBytes(__gm__ uint8_t *dst, __gm__ uint8_t *src, uint32_t bytes)
    {
        if (dst == src) {
            return;
        }
        for (uint32_t offset = 0U; offset < bytes;) {
            const uint32_t size = min(kCopyTileBytes, bytes - offset);
            GlobalTensor<uint8_t> source;
            source.SetGlobalBuffer(src + offset);
            DataCopyPad(*copyUb_, source, DataCopyExtParams{1U, size, 0U, 0U, 0U},
                        DataCopyPadExtParams<uint8_t>{false, 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE2_V>();
            GlobalTensor<uint8_t> destination;
            destination.SetGlobalBuffer(dst + offset);
            DataCopyPad(destination, *copyUb_, DataCopyExtParams{1U, size, 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE3_S>();
            offset += size;
        }
    }

    __aicore__ inline void CopyFlag(__gm__ uint64_t *dst, uint64_t value)
    {
        doorbellUb_->SetValue(0, value);
        GlobalTensor<uint64_t> destination;
        destination.SetGlobalBuffer(dst);
        SyncFunc<HardEvent::S_MTE3>();
        DataCopyPad(destination, *doorbellUb_, DataCopyExtParams{1U, sizeof(uint64_t), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }

    __aicore__ inline void SendFlag(uint32_t rank, uint64_t destinationOffset, uint64_t stagingOffset, Phase phase)
    {
        auto *dst = (__gm__ uint64_t *)((__gm__ uint8_t *)comm_.WindowIn(rank) + destinationOffset);
        if (rank / info_.ranksPerServer == server_) {
            CopyFlag(dst, Expected(phase));
        } else {
            auto *src = (__gm__ uint64_t *)(outWindow_ + stagingOffset);
            CopyFlag(src, Expected(phase));
            comm_.PostReadyFlag((GM_ADDR)src, (GM_ADDR)dst, rank, *doorbellUb_, *headUb_);
        }
    }

    __aicore__ inline void SendPayload(uint32_t rank, __gm__ uint8_t *source, __gm__ uint8_t *destination,
                                       uint32_t bytes)
    {
        if (rank / info_.ranksPerServer == server_) {
            CopyBytes(destination, source, bytes);
        } else {
            for (uint32_t offset = 0U; offset < bytes;) {
                const uint32_t size = min(bytes - offset, 1U << 20U);
                comm_.PostRdmaSend(source + offset, destination + offset, rank, size, *doorbellUb_, *headUb_);
                offset += size;
            }
        }
    }

    __aicore__ inline void StageOwnerRowsAndBases()
    {
        uint32_t prefix = 0U;
        const bool publishBases = GetBlockIdx() == 0U;
        for (uint32_t expert = 0U; expert < localExperts_; ++expert) {
            if (publishBases) {
                expertBasesUb_->SetValue(expert, prefix);  // 专家token前缀和
            }
            const uint32_t count =
                static_cast<uint32_t>(plan_[header_->localExpertCountOffset + expert]);  // 该专家处理的token数量
            const uint32_t physicalBase =
                static_cast<uint32_t>(plan_[header_->localExpertOffset + expert]);  // 该专家在recv_中的偏移
            for (uint32_t row = GetBlockIdx(); row < count; row += AivNum()) {      // 分核搬token到通信buffer
                CopyBytes(
                    in_ + layout_.ownerStageOffset + static_cast<uint64_t>(prefix + row) * layout_.ownerStageStride,
                    (__gm__ uint8_t *)(recv_ + static_cast<uint64_t>(physicalBase + row) * info_.h),
                    info_.h * sizeof(T));
            }
            prefix += count;
        }
        if (publishBases) {
            expertBasesUb_->SetValue(localExperts_, prefix);
            auto *bases = (__gm__ uint32_t *)(in_ + layout_.expertBaseOffset +
                                              static_cast<uint64_t>(info_.rankInServer) * layout_.expertBaseStride);
            GlobalTensor<uint32_t> destination;
            destination.SetGlobalBuffer(bases);
            SyncFunc<HardEvent::S_MTE3>();
            DataCopyPad(
                destination, *expertBasesUb_,
                DataCopyExtParams{1U, static_cast<uint32_t>((localExperts_ + 1U) * sizeof(uint32_t)), 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE3_S>();
            InitPartialProgress();  // 初始化并行状态区
        }
    }

    __aicore__ inline void FanoutExpertBases()
    {
        const uint32_t bytes = (localExperts_ + 1U) * sizeof(uint32_t);
        auto *source =
            in_ + layout_.expertBaseOffset + static_cast<uint64_t>(info_.rankInServer) * layout_.expertBaseStride;
        for (uint32_t proxy = GetBlockIdx(); proxy < info_.ranksPerServer; proxy += AivNum()) {
            if (proxy == info_.rankInServer) {
                continue;
            }
            const uint32_t rank = server_ * info_.ranksPerServer + proxy;
            auto *destination = (__gm__ uint8_t *)comm_.WindowIn(rank) + layout_.expertBaseOffset +
                                static_cast<uint64_t>(info_.rankInServer) * layout_.expertBaseStride;
            CopyBytes(destination, source, bytes);
        }
    }

    __aicore__ inline void PublishOwnerReady()
    {
        for (uint32_t proxy = 0U; proxy < info_.ranksPerServer; ++proxy) {
            const uint32_t rank = server_ * info_.ranksPerServer + proxy;
            SendFlag(rank, layout_.ownerReadyOffset + info_.rankInServer * COMBINE_FLAG_STRIDE, 0U,
                     Phase::COMBINE_OWNER_READY);
        }
    }

    __aicore__ inline void WaitOwnerReady()
    {
        for (uint32_t owner = 0U; owner < info_.ranksPerServer; ++owner) {
            WaitFlag(layout_.ownerReadyOffset + owner * COMBINE_FLAG_STRIDE, Phase::COMBINE_OWNER_READY);
        }
    }

    __aicore__ inline void PublishServerReady()
    {
        for (uint32_t targetServer = 0U; targetServer < info_.serverNum; ++targetServer) {
            const uint32_t rank = targetServer * info_.ranksPerServer + info_.rankInServer;
            SendFlag(rank, layout_.serverReadyOffset + server_ * COMBINE_FLAG_STRIDE,
                     layout_.serverFlagStageOffset + targetServer * COMBINE_FLAG_STRIDE, Phase::COMBINE_GLOBAL_READY);
        }
    }

    __aicore__ inline void WaitServerReady()
    {
        for (uint32_t sourceServer = 0U; sourceServer < info_.serverNum; ++sourceServer) {
            WaitFlag(layout_.serverReadyOffset + sourceServer * COMBINE_FLAG_STRIDE, Phase::COMBINE_GLOBAL_READY);
        }
    }

    __aicore__ inline void BuildPartialPacket(uint32_t source, uint32_t ordinal, uint64_t base, const PlanLayout &p)
    {
        const uint32_t packetIndex = source * info_.maxBs + ordinal;
        auto *members = (__gm__ uint8_t *)(plan_ + p.reverseMember) +
                        static_cast<uint64_t>(packetIndex) * info_.k * sizeof(ReverseMember);
        GlobalTensor<uint8_t> memberSource;
        memberSource.SetGlobalBuffer(members);
        const uint32_t memberBytes = static_cast<uint32_t>(info_.k * sizeof(ReverseMember));
        DataCopyPad(*copyUb_, memberSource, DataCopyExtParams{1U, memberBytes, 0U, 0U, 0U},
                    DataCopyPadExtParams<uint8_t>{false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_S>();
        auto memberWords = copyUb_->template ReinterpretCast<int32_t>();
        auto *partial = (__gm__ float *)(outWindow_ + layout_.proxyPartialOffset +
                                         static_cast<uint64_t>(base + ordinal) *
                                             layout_.partialStride);  // 拷贝到最终目的地，准备发给server
        __gm__ T *ownerRows[kMaxTopk];
        float weights[kMaxTopk];
        uint32_t validMembers = 0U;
        for (uint32_t slot = 0U; slot < info_.k; ++slot) {  // 获取有效专家的token
            const uint32_t memberOffset = slot * ReverseMemberElements();
            const int32_t expertId = memberWords.GetValue(memberOffset);
            if (expertId < 0) {
                continue;
            }
            const uint32_t expert = static_cast<uint32_t>(expertId);
            const int32_t expertRow = memberWords.GetValue(memberOffset + 1U);
            const int32_t weightBits = memberWords.GetValue(memberOffset + 3U);
            const uint32_t ownerRank = expert / localExperts_;
            const uint32_t localExpert = expert % localExperts_;
            auto *ownerWindow = (__gm__ uint8_t *)comm_.WindowIn(ownerRank);
            auto *expertBases =
                (__gm__ uint32_t *)(in_ + layout_.expertBaseOffset +
                                    static_cast<uint64_t>(ownerRank % info_.ranksPerServer) * layout_.expertBaseStride);
            const uint32_t ownerRow = expertBases[localExpert] + static_cast<uint32_t>(expertRow);
            ownerRows[validMembers] = (__gm__ T *)(ownerWindow + layout_.ownerStageOffset +
                                                   static_cast<uint64_t>(ownerRow) * layout_.ownerStageStride);
            union {
                int32_t bits;
                float value;
            } weight{};
            weight.bits = weightBits;
            weights[validMembers] = weight.value;
            ++validMembers;
        }
        AscendC::Duplicate(*partialUb_, 0.0F, static_cast<uint32_t>(layout_.partialStride / sizeof(float)));
        PipeBarrier<PIPE_V>();
        for (uint32_t member = 0U; member < validMembers; ++member) {  // 远端读+加权求和
            GlobalTensor<T> sourceRow;
            sourceRow.SetGlobalBuffer(ownerRows[member]);
            DataCopyPad(*ownerRowUb_, sourceRow,
                        DataCopyExtParams{1U, static_cast<uint32_t>(info_.h * sizeof(T)), 0U, 0U, 0U},
                        DataCopyPadExtParams<T>{false, 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE2_V>();
            AscendC::Cast(*weightedUb_, *ownerRowUb_, RoundMode::CAST_NONE, info_.h);
            PipeBarrier<PIPE_V>();
            AscendC::Muls(*weightedUb_, *weightedUb_, weights[member], info_.h);
            PipeBarrier<PIPE_V>();
            AscendC::Add(*partialUb_, *partialUb_, *weightedUb_, info_.h);
            PipeBarrier<PIPE_V>();
        }
        GlobalTensor<float> destination;
        destination.SetGlobalBuffer(partial);
        SyncFunc<HardEvent::V_MTE3>();
        DataCopyPad(destination, *partialUb_,
                    DataCopyExtParams{1U, static_cast<uint32_t>(layout_.partialStride), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }

    __aicore__ inline void PublishOwnerConsumed()
    {
        for (uint32_t owner = 0U; owner < info_.ranksPerServer; ++owner) {
            const uint32_t rank = server_ * info_.ranksPerServer + owner;
            SendFlag(rank, layout_.ownerConsumedOffset + info_.rankInServer * COMBINE_FLAG_STRIDE, 0U,
                     Phase::COMBINE_OWNER_CONSUMED);
        }
    }

    __aicore__ inline uint32_t WorkerCount() const
    {
        return AivNum() > info_.serverNum ? AivNum() - info_.serverNum : AivNum();
    }

    __aicore__ inline bool IsWorker() const
    {
        return AivNum() <= info_.serverNum || GetBlockIdx() >= info_.serverNum;
    }

    __aicore__ inline uint32_t WorkerIndex() const
    {
        return AivNum() > info_.serverNum ? GetBlockIdx() - info_.serverNum : GetBlockIdx();
    }

    __aicore__ inline __gm__ uint32_t *CompletedChunks(uint32_t source, uint32_t worker) const
    {
        return (__gm__ uint32_t *)(progress_ +
                                   (static_cast<uint64_t>(source) * WorkerCount() + worker) * kProgressStride);
    }

    __aicore__ inline void InitPartialProgress()
    {
        const uint32_t bytes = info_.serverNum * WorkerCount() * kProgressStride;
        Duplicate(*progressUb_, 0U, bytes / sizeof(uint32_t));
        SyncFunc<HardEvent::V_MTE3>();
        GlobalTensor<uint32_t> destination;
        destination.SetGlobalBuffer((__gm__ uint32_t *)progress_);
        DataCopyPad(destination, *progressUb_, DataCopyExtParams{1U, bytes, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }

    __aicore__ inline void CopyProgress(__gm__ uint32_t *progress, uint32_t value)
    {
        auto words = copyUb_->template ReinterpretCast<uint32_t>();
        words.SetValue(0, value);
        GlobalTensor<uint32_t> destination;
        destination.SetGlobalBuffer(progress);
        SyncFunc<HardEvent::S_MTE3>();
        DataCopyPad(destination, words, DataCopyExtParams{1U, sizeof(uint32_t), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }

    __aicore__ inline void PublishChunkComplete(uint32_t source, uint32_t worker, uint32_t completed)
    {
        CopyProgress(CompletedChunks(source, worker), completed);
    }

    __aicore__ inline bool ChunkComplete(uint32_t source, uint32_t completed, uint32_t workers) const
    {
        for (uint32_t worker = 0U; worker < workers; ++worker) {  // 遍历worker是否均已完成
            __gm__ uint32_t *progress = CompletedChunks(source, worker);
            GlobalTensor<uint32_t> source;
            source.SetGlobalBuffer(progress);
            auto words = copyUb_->template ReinterpretCast<uint32_t>();
            cacheReadInvalidate((__gm__ uint8_t *)progress, sizeof(uint32_t));
            DataCopyPad(words, source, DataCopyExtParams{1U, sizeof(uint32_t), 0U, 0U, 0U},
                        DataCopyPadExtParams<uint32_t>{false, 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE2_S>();
            if (words.GetValue(0) < completed) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint32_t ChunkCount(uint32_t packetCount) const
    {
        return packetCount == 0U ? 0U : 1U + (packetCount - 1U) / chunkPackets_;
    }

    __aicore__ inline void BuildNextChunkPacket(uint32_t source, uint32_t count, uint64_t base, const PlanLayout &p,
                                                uint32_t worker, uint32_t workers, uint32_t &chunk, uint32_t &ordinal)
    {
        // 当前chunk范围
        const uint32_t begin = chunk * chunkPackets_;
        const uint32_t end = begin + min(chunkPackets_, count - begin);
        if (ordinal < end) {
            BuildPartialPacket(source, ordinal, base, p);
            ordinal += workers;
        }
        if (ordinal >= end) {
            PublishChunkComplete(source, worker, chunk + 1U);  // 当前chunk完成，通知server可发送
            ++chunk;
            if (chunk < ChunkCount(count)) {
                const uint32_t nextBegin = chunk * chunkPackets_;
                ordinal = worker < count - nextBegin ? nextBegin + worker : count;
            }
        }
    }

    __aicore__ inline bool TrySendPartialChunk(uint32_t source, uint32_t count, uint64_t base, const PlanLayout &p,
                                               uint32_t workers, uint32_t &nextChunk)
    {
        const uint32_t chunks = ChunkCount(count);                                    // chunk的数量
        if (nextChunk < chunks && !ChunkComplete(source, nextChunk + 1U, workers)) {  // 等待worker收齐
            return false;
        }
        const uint32_t rank = source * info_.ranksPerServer + info_.rankInServer;
        if (nextChunk < chunks) {
            const uint32_t begin = nextChunk * chunkPackets_;
            const uint32_t end = begin + min(chunkPackets_, count - begin);
            const uint32_t returnBase = static_cast<uint32_t>(plan_[p.reverseReturnBase + source]);
            auto *payload =
                outWindow_ + layout_.proxyPartialOffset + static_cast<uint64_t>(base + begin) * layout_.partialStride;
            auto *destination = (__gm__ uint8_t *)comm_.WindowIn(rank) + layout_.sourcePartialOffset +
                                (static_cast<uint64_t>(returnBase) + begin) * layout_.partialStride;
            SendPayload(rank, payload, destination,
                        static_cast<uint32_t>((end - begin) * layout_.partialStride));  // 跨server发送
            ++nextChunk;
        }
        if (nextChunk == chunks) {
            SendFlag(rank, layout_.sourceReadyOffset + server_ * COMBINE_FLAG_STRIDE,
                     layout_.sourceFlagStageOffset + source * COMBINE_FLAG_STRIDE,
                     Phase::COMBINE_SOURCE_READY);  // 发送完成
            return true;
        }
        return false;
    }

    __aicore__ inline void RunGroupedPipeline(const PlanLayout &p, const uint32_t *counts, const uint64_t *bases)
    {
        const uint32_t aiv = GetBlockIdx();
        const uint32_t workers = AivNum() - info_.serverNum;
        const uint32_t groupWidth = workers / info_.serverNum;
        const uint32_t extraWorkers = workers % info_.serverNum;
        // sender核逻辑
        if (aiv < info_.serverNum) {
            const uint32_t source = aiv;
            const uint32_t groupSize = groupWidth + (source < extraWorkers ? 1U : 0U);
            uint32_t nextChunk = 0U;
            while (!TrySendPartialChunk(source, counts[source], bases[source], p, groupSize,
                                        nextChunk)) {  // 等待chunk完成并发送，直到发完为止
                AscendC::GetSystemCycle();
            }
            return;
        }

        const uint32_t worker = aiv - info_.serverNum;
        uint32_t groupBegin = 0U;
        // worker核逻辑
        for (uint32_t source = 0U; source < info_.serverNum; ++source) {
            const uint32_t groupSize = groupWidth + (source < extraWorkers ? 1U : 0U);
            if (worker >= groupBegin + groupSize) {  // 判断当前AIV处理哪个server
                groupBegin += groupSize;
                continue;
            }
            const uint32_t localWorker = worker - groupBegin;
            uint32_t chunk = 0U;
            uint32_t ordinal = localWorker;
            const uint32_t chunks = ChunkCount(counts[source]);  // 获取需要处理的chunk数量
            while (chunk < chunks) {                             // 逐chunk处理
                BuildNextChunkPacket(source, counts[source], bases[source], p, localWorker, groupSize, chunk,
                                     ordinal);  // 处理该chunk的收齐，并通知sender
            }
            return;
        }
    }

    __aicore__ inline void RunRoundRobinPipeline(const PlanLayout &p, const uint32_t *counts, const uint64_t *bases)
    {
        const bool worker = IsWorker();
        const uint32_t workerIndex = worker ? WorkerIndex() : 0U;
        const uint32_t workers = WorkerCount();
        uint32_t buildChunk[kMaxServers];    // 该源已生产到第几个 chunk
        uint32_t buildOrdinal[kMaxServers];  // 该源当前 chunk 内已生产到第几个 packet
        uint32_t sendChunk[kMaxServers];     // 该源已发送到第几个 chunk
        uint32_t sendRemaining = 0U;         // 本sender核还要处理多少server
        uint32_t buildRemaining = 0U;        // 本worker核还要处理多少server
        for (uint32_t source = 0U; source < info_.serverNum; ++source) {
            buildChunk[source] = 0U;
            buildOrdinal[source] = workerIndex;  // 各worker交错开始
            sendChunk[source] = 0U;
            if (worker && counts[source] != 0U) {  // 多少个server有token
                ++buildRemaining;
            }
            if ((AivNum() > info_.serverNum && GetBlockIdx() == source) ||
                (AivNum() <= info_.serverNum && source % AivNum() == GetBlockIdx())) {
                ++sendRemaining;  // 对于sender的AIV，每个AIV仅处理一个server的通信
            }
        }
        uint32_t sendCursor = 0U;
        uint32_t buildCursor = 0U;
        while (sendRemaining != 0U || buildRemaining != 0U) {
            bool advanced = false;
            // sender核逻辑
            for (uint32_t step = 0U; step < info_.serverNum; ++step) {
                const uint32_t source = (sendCursor + step) % info_.serverNum;
                const bool ownsSend =
                    AivNum() > info_.serverNum ? GetBlockIdx() == source : source % AivNum() == GetBlockIdx();
                if (!ownsSend || sendChunk[source] > ChunkCount(counts[source])) {  // 非本AIV处理的server，跳过
                    continue;
                }
                const uint32_t previousChunk = sendChunk[source];
                if (TrySendPartialChunk(source, counts[source], bases[source], p, workers,
                                        sendChunk[source])) {             // 跨server发送
                    sendChunk[source] = ChunkCount(counts[source]) + 1U;  // 标记该server已发送完毕
                    --sendRemaining;
                    sendCursor = (source + 1U) % info_.serverNum;
                    advanced = true;
                    break;
                }
                if (sendChunk[source] != previousChunk) {  // 发了但没发完
                    sendCursor = (source + 1U) % info_.serverNum;
                    advanced = true;
                    break;
                }
            }
            // worker核逻辑
            if (worker && buildRemaining != 0U) {
                for (uint32_t step = 0U; step < info_.serverNum; ++step) {  // 遍历server
                    const uint32_t source = (buildCursor + step) % info_.serverNum;
                    const uint32_t chunks = ChunkCount(counts[source]);  // 该server发送的packet总数
                    if (buildChunk[source] >= chunks) {                  // 该server的packet已拉取完毕
                        continue;
                    }
                    BuildNextChunkPacket(source, counts[source], bases[source], p, workerIndex, workers,
                                         buildChunk[source], buildOrdinal[source]);
                    if (buildChunk[source] == chunks) {
                        --buildRemaining;
                    }
                    buildCursor = (source + 1U) % info_.serverNum;
                    advanced = true;
                    break;
                }
            }
            if (!advanced) {
                AscendC::GetSystemCycle();
            }
        }
    }

    __aicore__ inline void RunPartialPipeline()
    {
        const auto p = MakePlanLayout(info_.bs, info_.maxBs, info_.k, info_.moeExpertNum, info_.epWorldSize,
                                      info_.serverNum, info_.ranksPerServer, 1U, info_.maxBs);
        uint32_t counts[kMaxServers];  // 源 server 的 packet 数
        uint64_t bases[kMaxServers];   // 源 server 在本 rank 本地的起始偏移
        uint64_t base = 0U;
        for (uint32_t source = 0U; source < info_.serverNum; ++source) {
            counts[source] = static_cast<uint32_t>(plan_[p.reverseCount + source]);
            bases[source] = base;
            base += counts[source];
        }

        /*
         * 前 serverNum 个AIV核负责server间发送；
         * 其余AIV平均分组，每组固定处理一个server的packet。
         */
        const uint32_t workers = AivNum() > info_.serverNum ? AivNum() - info_.serverNum : AivNum();
        if (AivNum() <= info_.serverNum || workers < info_.serverNum) {  // 如果剩余的AIV数小于server数，回退算法
            RunRoundRobinPipeline(p, counts, bases);
            return;
        }
        // 将AIV分组处理不同server的数据
        RunGroupedPipeline(p, counts, bases);
    }

    __aicore__ inline void WaitSourcePartials()
    {
        for (uint32_t target = GetBlockIdx(); target < info_.serverNum; target += AivNum()) {
            WaitFlag(layout_.sourceReadyOffset + target * COMBINE_FLAG_STRIDE, Phase::COMBINE_SOURCE_READY);  // 等待
        }
    }

    __aicore__ inline void FinalizeOutput()
    {
        const auto p = MakePlanLayout(info_.bs, info_.maxBs, info_.k, info_.moeExpertNum, info_.epWorldSize,
                                      info_.serverNum, info_.ranksPerServer, 1U, info_.maxBs);
        // 按照token分核
        for (uint32_t token = GetBlockIdx(); token < info_.bs; token += AivNum()) {
            AscendC::Duplicate(*partialUb_, 0.0F, static_cast<uint32_t>(layout_.partialStride / sizeof(float)));
            PipeBarrier<PIPE_V>();
            for (uint32_t target = 0U; target < info_.serverNum; ++target) {
                const int32_t slot = plan_[header_->packetSlotOffset + token * info_.serverNum + target];
                if (slot < 0) {
                    continue;
                }
                const uint32_t packetIndex = target * info_.maxBs + static_cast<uint32_t>(slot);
                const auto *packet = (__gm__ SourcePacketDesc *)(plan_ + header_->packetDescOffset) + packetIndex;
                const uint32_t base = static_cast<uint32_t>(plan_[p.sourceReturnBase + target]);
                auto *partial =
                    (__gm__ float *)(in_ + layout_.sourcePartialOffset +
                                     (static_cast<uint64_t>(base) + static_cast<uint32_t>(packet->dispatchOrdinal)) *
                                         layout_.partialStride);
                GlobalTensor<float> source;
                source.SetGlobalBuffer(partial);
                DataCopyPad(*weightedUb_, source,
                            DataCopyExtParams{1U, static_cast<uint32_t>(info_.h * sizeof(float)), 0U, 0U, 0U},
                            DataCopyPadExtParams<float>{false, 0U, 0U, 0.0F});
                SyncFunc<HardEvent::MTE2_V>();
                AscendC::Add(*partialUb_, *partialUb_, *weightedUb_, info_.h);  // topk累加求和
                PipeBarrier<PIPE_V>();
                SyncFunc<HardEvent::V_MTE2>();
            }
            AscendC::Cast(*ownerRowUb_, *partialUb_, RoundMode::CAST_RINT, info_.h);  // 转回原始数据类型
            GlobalTensor<T> destination;
            destination.SetGlobalBuffer(out_ + static_cast<uint64_t>(token) * info_.h);
            SyncFunc<HardEvent::V_MTE3>();
            DataCopyPad(destination, *ownerRowUb_,
                        DataCopyExtParams{1U, static_cast<uint32_t>(info_.h * sizeof(T)), 0U, 0U, 0U});  // 输出位置
            SyncFunc<HardEvent::MTE3_S>();
        }
    }

    __aicore__ inline void PublishSourceConsumed()
    {
        for (uint32_t target = GetBlockIdx(); target < info_.serverNum; target += AivNum()) {
            const uint32_t rank = target * info_.ranksPerServer + info_.rankInServer;
            SendFlag(rank, layout_.sourceConsumedOffset + server_ * COMBINE_FLAG_STRIDE,
                     layout_.consumedFlagStageOffset + target * COMBINE_FLAG_STRIDE, Phase::COMBINE_SOURCE_CONSUMED);
        }
    }

    __aicore__ inline void WaitConsumed()
    {
        for (uint32_t proxy = GetBlockIdx(); proxy < info_.ranksPerServer; proxy += AivNum()) {
            WaitFlag(layout_.ownerConsumedOffset + proxy * COMBINE_FLAG_STRIDE, Phase::COMBINE_OWNER_CONSUMED);
        }
        for (uint32_t source = GetBlockIdx(); source < info_.serverNum; source += AivNum()) {
            WaitFlag(layout_.sourceConsumedOffset + source * COMBINE_FLAG_STRIDE, Phase::COMBINE_SOURCE_CONSUMED);
        }
    }

    __aicore__ inline void DrainOutbound()
    {
        for (uint32_t server = GetBlockIdx(); server < info_.serverNum; server += AivNum()) {
            const uint32_t rank = server * info_.ranksPerServer + info_.rankInServer;
            comm_.Drain(rank, 0U);
        }
    }

    __gm__ PlanHeader *header_{nullptr};
    __gm__ int32_t *plan_{nullptr};
    __gm__ T *recv_{nullptr};
    __gm__ T *out_{nullptr};
    __gm__ uint8_t *in_{nullptr};
    __gm__ uint8_t *outWindow_{nullptr};
    __gm__ uint8_t *progress_{nullptr};
    CamMoeLayeredInfo info_{};
    CombineWindowLayout layout_{};
    CommAdapter comm_{};
    LocalTensor<uint64_t> *doorbellUb_{nullptr};
    LocalTensor<uint32_t> *headUb_{nullptr};
    LocalTensor<uint32_t> *expertBasesUb_{nullptr};
    LocalTensor<uint8_t> *copyUb_{nullptr};
    LocalTensor<uint32_t> *progressUb_{nullptr};
    LocalTensor<float> *partialUb_{nullptr};
    LocalTensor<float> *weightedUb_{nullptr};
    LocalTensor<T> *ownerRowUb_{nullptr};
    uint64_t epoch_{0U};
    uint32_t server_{0U};
    uint32_t localExperts_{0U};
    uint32_t chunkPackets_{0U};
};
}  // namespace

extern "C" __global__ __aicore__ void cam_moe_combine_normal_layered(GM_ADDR roceContext, GM_ADDR recvX,
                                                                     GM_ADDR layeredPlan, GM_ADDR xOut,
                                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CamMoeCombineNormalLayeredTilingData);
    TPipe pipe;
    // Must match SetTilingKey(13000U) in cam_moe_combine_normal_layered_tiling.cc.
    // Declaring the key here is what makes the op compiler emit the _13000
    // funcEntry; without it only the default key-0 entry is generated.
    if (TILING_KEY_IS(13000)) {
#if (ORIG_DTYPE_RECV_X == DT_BF16 || ORIG_DTYPE_RECV_X == DT_FLOAT16)
        GET_TILING_DATA_WITH_STRUCT(CamMoeCombineNormalLayeredTilingData, data, tiling);
        TBuf rdmaUb;
        pipe.InitBuffer(rdmaUb, 32U);
        TBuf copyUb;
        const uint32_t localExperts = data.layered.moeExpertNum / data.layered.epWorldSize;
        const uint32_t expertBasesUbBytes = static_cast<uint32_t>(CombineExpertBasesUbBytes(localExperts));
        pipe.InitBuffer(copyUb, expertBasesUbBytes + kCopyTileBytes);
        TBuf progressUb;
        const uint32_t workers = data.layered.aivNum > data.layered.serverNum
                                     ? data.layered.aivNum - data.layered.serverNum
                                     : data.layered.aivNum;
        const uint32_t progressBytes = data.layered.serverNum * workers * kProgressStride;
        pipe.InitBuffer(progressUb, progressBytes);
        TBuf partialUb;
        pipe.InitBuffer(partialUb, CombinePartialUbBytes(data.layered.h));
        TBuf weightedUb;
        pipe.InitBuffer(weightedUb, CombinePartialUbBytes(data.layered.h));
        TBuf ownerRowUb;
        pipe.InitBuffer(ownerRowUb, CombineOwnerRowUbBytes(data.layered.h, sizeof(DTYPE_RECV_X)));
#ifdef __DAV_C310__
        TBuf hcommUb;
        pipe.InitBuffer(hcommUb, HCOMM_UB_BUF_SIZE);
#endif
        auto doorbellTensor = rdmaUb.GetWithOffset<uint64_t>(1U, 0U);
        auto headTensor = rdmaUb.GetWithOffset<uint32_t>(1U, 16U);
        auto expertBasesTensor = copyUb.GetWithOffset<uint32_t>(localExperts + 1U, 0U);
        auto copyTensor = copyUb.GetWithOffset<uint8_t>(kCopyTileBytes, expertBasesUbBytes);
        auto progressTensor = progressUb.Get<uint32_t>();
        auto partialTensor = partialUb.Get<float>();
        auto weightedTensor = weightedUb.Get<float>();
        auto ownerRowTensor = ownerRowUb.Get<DTYPE_RECV_X>();
#ifdef __DAV_C310__
        auto hcommTensor = hcommUb.Get<uint8_t>();
#else
        auto hcommTensor = copyTensor;
#endif
        CamMoeCombineNormalLayered<DTYPE_RECV_X> op;
        op.Init(roceContext, recvX, layeredPlan, xOut, workspace, data.layered, data.ownerCapacity,
                data.chunkPackets, doorbellTensor, headTensor, expertBasesTensor, copyTensor, progressTensor,
                partialTensor, weightedTensor, ownerRowTensor, hcommTensor);
        op.Process();
#endif
    }
}

#endif  // DEEPEP_CANN_VERSION_NUM >= DEEPEP_LAYERED_MIN_CANN_VERSION
