#pragma once

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"

namespace ggml_backend_metalium {

using CoreSpec = std::variant<tt::tt_metal::CoreCoord, tt::tt_metal::CoreRange, tt::tt_metal::CoreRangeSet>;
tt::tt_metal::CBHandle MakeCircularBuffer(
    tt::tt_metal::Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format);
tt::tt_metal::CBHandle MakeCircularBuffer(tt::tt_metal::Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles, tt::tt_metal::DataType dtype);

tt::tt_metal::KernelHandle CreateMetaliumKernel(
    tt::tt_metal::Program& program,
    const std::string& str, // could be path or actual kenrel
    const CoreSpec& core_spec,
    const std::variant<tt::tt_metal::DataMovementConfig, tt::tt_metal::ComputeConfig>& config);

std::string to_string_precise(float value);

template <typename T>
std::optional<T> at_index(const std::vector<T>& vec, size_t index) {
    if (index < vec.size()) {
        return vec[index];
    }
    return std::nullopt;
}

bool ggml_tt_tensors_shape_equal(const ggml_tensor* ggtensor, const tt::tt_metal::Tensor& ttensor);

tt::tt_metal::DataType ggml2tt_type(ggml_type ggtype, tt::ARCH arch);
bool is_ggml_type_supported_by_metalium(ggml_type ggtype, tt::ARCH arch);

bool is_view(const ggml_tensor* tensor);
tt::tt_metal::Tensor realize_ggml_view(const ggml_tensor* tensor);
tt::tt_metal::Tensor reshape_tt_tensor_into_ggml(const tt::tt_metal::Tensor& tensor, const struct ggml_tensor * node);

// Debug flags that can be enabled at runtime. Because recompiling the backend takes forever
// this enables faster iteration on debugging. Eventually these should be removed
// NOTE: DO NOT invent more _hack flags. Else it devolves into a mess like what BUDA did
struct ggml_backend_metalium_debug_flags {
    bool print_rejected_ops = false;        // Print ops that the backend rejects
    bool print_view = false;                // Print details when a VIEW op is being realized
    bool cache_mm_transpose = false;        // Cache the transpose kernel for matmul
    bool disable_program_cache = false;     // Disables the program cache
    bool experimental_ops = false;          // Enable experimental ops that is known to cause trouble
};

extern const ggml_backend_metalium_debug_flags g_debug_flags;

tt::tt_metal::Shape tt_shape_of(const ggml_tensor* ggtensor);

} // namespace ggml_backend_metalium
