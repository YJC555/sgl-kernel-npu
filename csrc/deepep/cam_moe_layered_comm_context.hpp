#ifndef CAM_MOE_LAYERED_COMM_HOST_CONTEXT_HPP
#define CAM_MOE_LAYERED_COMM_HOST_CONTEXT_HPP

#include <cstdint>
#include <string>

#include "hccl/hccl.h"

#include "cam_moe_layered_comm_context.h"

namespace deep_ep {

struct LayeredCommResource {
    void *engineContext = nullptr;
    uint64_t windowBytes = 0U;
};

// Buffer-owned A5 manager. HCCL owns the device context; registered windows
// remain live for the communicator lifetime.
class LayeredCommContextManager
{
public:
    LayeredCommContextManager() = default;
    ~LayeredCommContextManager();
    LayeredCommContextManager(const LayeredCommContextManager &) = delete;
    LayeredCommContextManager &operator=(const LayeredCommContextManager &) = delete;

    HcclResult Create(HcclComm comm, const std::string &contextTag, CamMoeLayeredProtocol protocol,
                      uint32_t ranksPerServer, uint64_t minimumWindowBytes, LayeredCommResource &resource);
};

}  // namespace deep_ep

#endif
