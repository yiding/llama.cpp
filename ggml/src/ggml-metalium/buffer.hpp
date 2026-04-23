#pragma once

#include "ggml-backend-impl.h"

#include <vector>
#include <memory>
#include <string>

namespace ggml_backend_metalium {

struct ggml_tensor_extra_metalium {
    tt::tt_metal::Tensor tensor;
    bool is_pretransposed = false;
};

struct ggml_backend_metalium_buffer_context {
    size_t ggml_buffer_size_bytes = 0;
    std::string name;
    std::shared_ptr<ttnn::MeshDevice> device = nullptr;
    size_t base_offset = 0;

    // Tracking our own allocations because Metalium limitations and GGML assuming them
    std::vector<std::unique_ptr<ggml_tensor_extra_metalium>> metadata_to_free;
};

extern struct ggml_backend_buffer_i ggml_backend_metalium_buffer_interface;

} // namespace ggml_backend_metalium
