#pragma once

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ttnn/distributed/distributed_tensor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ggml_backend_metalium {

struct tensor_extra {
    /// TT tensor associated with this GGML tensor.
    /// Note: This value might not be initialized until an operation is run and
    /// computes the value, or when set_tensor is called to set it.
    tt::tt_metal::Tensor tensor;

    /// Whether this tensor is always used transposed (i.e. for weights).
    bool is_pretransposed;
    std::optional<size_t> tp_dim;
    std::optional<ttnn::distributed::TensorToMesh> mesh_mapper;

    static tensor_extra * from(const ggml_tensor * tensor) {
        return static_cast<tensor_extra *>(tensor->extra);
    }
};

const tt::tt_metal::Tensor & get_tt_tensor(const ggml_tensor * tensor);
tt::tt_metal::Tensor &       get_tt_tensor(ggml_tensor * tensor);

struct ggml_backend_metalium_buffer_context {
    size_t                            ggml_buffer_size_bytes = 0;
    std::string                       name;
    std::shared_ptr<ttnn::MeshDevice> device;
    size_t                            base_offset = 0;

    // Tracking our own allocations because Metalium limitations and GGML assuming them
    std::vector<std::unique_ptr<tensor_extra>> metadata_to_free;

    ssize_t tp_axis() const { return device->shape().dims() - 1; }

    std::optional<size_t> tp_ne() const {
        size_t shard_devs = device->shape()[tp_axis()];
        return (shard_devs > 1) ? std::make_optional(shard_devs) : std::nullopt;
    }

    static ggml_backend_metalium_buffer_context * get(ggml_backend_buffer_t b) {
        return static_cast<ggml_backend_metalium_buffer_context *>(b->context);
    }
};

extern struct ggml_backend_buffer_i ggml_backend_metalium_buffer_interface;

}  // namespace ggml_backend_metalium
