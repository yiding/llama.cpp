#pragma once

#include "ggml-backend-impl.h"

#include <vector>
#include <memory>
#include <string>

namespace ggml_backend_metalium {

struct tensor_extra {
    tt::tt_metal::Tensor tensor;
};

const tt::tt_metal::Tensor& get_tt_tensor(const ggml_tensor * tensor);
tt::tt_metal::Tensor& get_tt_tensor(ggml_tensor * tensor);


struct ggml_backend_metalium_buffer_context {
    size_t ggml_buffer_size_bytes = 0;
    std::string name;
    std::shared_ptr<ttnn::MeshDevice> device = nullptr;
    size_t base_offset = 0;

    // Tracking our own allocations because Metalium limitations and GGML assuming them
    std::vector<std::unique_ptr<tensor_extra>> metadata_to_free;
};

extern struct ggml_backend_buffer_i ggml_backend_metalium_buffer_interface;

} // namespace ggml_backend_metalium
