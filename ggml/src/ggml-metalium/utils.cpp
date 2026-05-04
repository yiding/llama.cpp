#include "utils.hpp"

#include "buffer.hpp"
#include "ggml.h"

#include <tt-metalium/buffer.hpp>

#include <filesystem>
#include <unordered_map>

#ifdef GGML_METALIUM_EMBED_KERNELS
std::unordered_map<std::string, std::string> & ggml_metalium_get_kernel_map();
#endif

using namespace tt::tt_metal;

namespace ggml_backend_metalium {

CBHandle MakeCircularBuffer(Program &        program,
                            const CoreSpec & core,
                            tt::CBIndex      cb,
                            uint32_t         size,
                            uint32_t         page_size,
                            tt::DataFormat   format) {
    CircularBufferConfig cb_src0_config = CircularBufferConfig(size,
                                                               {
                                                                   { cb, format }
    })
                                              .set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

CBHandle MakeCircularBuffer(Program &              program,
                            const CoreSpec &       core,
                            tt::CBIndex            cb,
                            uint32_t               n_tiles,
                            tt::tt_metal::DataType dtype) {
    auto df2dt = [](tt::tt_metal::DataType dt) {
        switch (dt) {
            case tt::tt_metal::DataType::FLOAT32:
                return tt::DataFormat::Float32;
            case tt::tt_metal::DataType::BFLOAT16:
                return tt::DataFormat::Float16_b;
            case tt::tt_metal::DataType::INT32:
                return tt::DataFormat::Int32;
            case tt::tt_metal::DataType::BFLOAT8_B:
                return tt::DataFormat::Bfp8_b;
            case tt::tt_metal::DataType::BFLOAT4_B:
                return tt::DataFormat::Bfp4_b;
            case tt::tt_metal::DataType::UINT8:
                return tt::DataFormat::UInt8;
            case tt::tt_metal::DataType::UINT16:
                return tt::DataFormat::UInt16;
            case tt::tt_metal::DataType::UINT32:
                return tt::DataFormat::UInt32;
            default:
                TT_FATAL(false, "Unsupported data type: {}", static_cast<int>(dt));
        }
    };

    auto tile_size = tt::tile_size(df2dt(dtype));
    return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, df2dt(dtype));
}

KernelHandle CreateMetaliumKernel(Program &           program,
                                  const std::string & str,  // could be path or actual kenrel
                                  const CoreSpec &    core_spec,
                                  const std::variant<DataMovementConfig, ComputeConfig> & config) {
    if (str.find_first_of(" \n\t") != std::string::npos) {
        return tt::tt_metal::CreateKernelFromString(program, str, core_spec, config);
    }
#ifdef GGML_METALIUM_EMBED_KERNELS
    // if it looks like a name
    if (str.find_first_of(" \n\t") == std::string::npos) {
        auto & kernel_map = ggml_metalium_get_kernel_map();
        auto   it         = kernel_map.find(str);
        if (it != kernel_map.end()) {
            return tt::tt_metal::CreateKernelFromString(program, it->second, core_spec, config);
        }
    }
#endif

    namespace fs = std::filesystem;
    if (fs::exists(str)) {
        return tt::tt_metal::CreateKernel(program, str, core_spec, config);
    }

    if (!fs::path(str).is_absolute()) {
        // We are at root of GGML dir
        fs::path p = fs::current_path() / "ggml/src/ggml-metalium/kernels/" / str;
        if (fs::exists(p)) {
            return tt::tt_metal::CreateKernel(program, p.string(), core_spec, config);
        }
        if (p.extension() != ".cpp") {
            p = (fs::current_path() / "ggml/src/ggml-metalium/kernels/" / str).generic_string() + ".cpp";
            if (fs::exists(p)) {
                return tt::tt_metal::CreateKernel(program, p.string(), core_spec, config);
            }
        }

        // we are at the build folder of llama.cpp
        p = fs::current_path() / "../ggml/src/ggml-metalium/kernels/" / str;
        if (fs::exists(p)) {
            return tt::tt_metal::CreateKernel(program, p.string(), core_spec, config);
        }
        if (p.extension() != ".cpp") {
            p = (fs::current_path() / "../ggml/src/ggml-metalium/kernels/" / str).generic_string() + ".cpp";
            if (fs::exists(p)) {
                return tt::tt_metal::CreateKernel(program, p.string(), core_spec, config);
            }
        }

        const char * metalium_kernel_root = getenv("GGML_METALIUM_KERNEL_ROOT");
        if (metalium_kernel_root != nullptr) {
            p = fs::path(metalium_kernel_root) / str;
            if (fs::exists(p)) {
                return tt::tt_metal::CreateKernel(program, p.string(), core_spec, config);
            }
        }
    }

    throw std::runtime_error("Kernel " + str + " not found in any search path nor itself looks like a kernel");
}

std::string to_string_precise(float value) {
    std::stringstream ss;
    ss << std::hexfloat << value;
    return ss.str();
}

bool ggml_tt_tensors_shape_equal(const ggml_tensor * ggtensor, const tt::tt_metal::Tensor & ttensor) {
    tensor_extra * meta = tensor_extra::from(ggtensor);
    ttnn::Shape shape = ttensor.logical_shape();
    if (meta->is_pretransposed) {
        size_t h = shape[-1];
        size_t w = shape[-2];
        shape[-1] = w;
        shape[-2] = h;
    }
    for (size_t i = 0; i < std::min<size_t>(GGML_MAX_DIMS, shape.size()); i++) {
        if (ggtensor->ne[GGML_MAX_DIMS - i - 1] != shape[i]) {
            return false;
        }
    }

    if (shape.size() > GGML_MAX_DIMS) {
        for (size_t i = GGML_MAX_DIMS; i < shape.size(); i++) {
            if (shape[i] != 1) {
                return false;
            }
        }
    } else if (shape.size() < GGML_MAX_DIMS) {
        for (size_t i = shape.size(); i < GGML_MAX_DIMS; i++) {
            if (ggtensor->ne[GGML_MAX_DIMS - i - 1] != 1) {
                return false;
            }
        }
    }
    return true;
}

static tt::tt_metal::DataType ggml2tt_type_internal(ggml_type ggtype, tt::ARCH arch) {
    // This table is consulted to map GGML types to TT types doing tensor creation
    if (arch == tt::ARCH::WORMHOLE_B0 || arch == tt::ARCH::BLACKHOLE) {
        static constexpr std::array<tt::tt_metal::DataType, GGML_TYPE_COUNT> table = {
            /*GGML_TYPE_F32        = */ tt::tt_metal::DataType::BFLOAT16,
            /*GGML_TYPE_F16        = */ tt::tt_metal::DataType::BFLOAT16,
            /*GGML_TYPE_Q4_0       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q4_1       = */ tt::tt_metal::DataType::BFLOAT8_B,
            tt::tt_metal::DataType::INVALID,
            tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_Q5_0       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q5_1       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q8_0       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q8_1       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q2_K       = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_Q3_K       = */ tt::tt_metal::DataType::BFLOAT4_B,
            /*GGML_TYPE_Q4_K       = */ tt::tt_metal::DataType::BFLOAT4_B,
            /*GGML_TYPE_Q5_K       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q6_K       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q8_K       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_IQ2_XXS    = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ2_XS     = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ3_XXS    = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ1_S      = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ4_NL     = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ3_S      = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ2_S      = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ4_XS     = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_I8         = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_I16        = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_I32        = */
            tt::tt_metal::DataType::UINT32,  // Yeah not ideal. but don't have support for tilizing int32 on device
            /*GGML_TYPE_I64        = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_F64        = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ1_M      = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_BF16       = */ tt::tt_metal::DataType::BFLOAT16,
            /*GGML_TYPE_Q4_0_4_4   = */ tt::tt_metal::DataType::INVALID,  // Support removed from GGML
            /*GGML_TYPE_Q4_0_4_8   = */ tt::tt_metal::DataType::INVALID,  // Support removed from GGML
            /*GGML_TYPE_Q4_0_8_8   = */ tt::tt_metal::DataType::INVALID,  // Support removed from GGML
            /*GGML_TYPE_TQ1_0      = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_TQ2_0      = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ4_NL_4_4 = */ tt::tt_metal::DataType::INVALID,  // Support removed from GGML
            /*GGML_TYPE_IQ4_NL_4_8 = */ tt::tt_metal::DataType::INVALID,  // Support removed from GGML
            /*GGML_TYPE_IQ4_NL_8_8 = */ tt::tt_metal::DataType::INVALID,  // Support removed from GGML
            // This is a bit wasteful, but MXFP4 can't be perfectly represented as BFLOAT4_B
            /*GGML_TYPE_MXFP4      = */ tt::tt_metal::DataType::BFLOAT8_B,
        };
        // safeguard against OOB read from outdated table
        if (ggtype >= table.size()) {
            return tt::tt_metal::DataType::INVALID;
        }
        tt::tt_metal::DataType type = table[ggtype];
        return type;
    }
    GGML_ASSERT(false && "Unsupported Tenstorrent card architecture");
}

tt::tt_metal::DataType ggml2tt_type(ggml_type ggtype, tt::ARCH arch) {
    tt::tt_metal::DataType type = ggml2tt_type_internal(ggtype, arch);
    if (type == tt::tt_metal::DataType::INVALID) {
        fmt::println(stderr, "Unsupported data type: {}", ggml_type_name(ggtype));
        GGML_ASSERT(false && "Unsupported data type");
    }
    return type;
}

bool is_ggml_type_supported_by_metalium(ggml_type ggtype, tt::ARCH arch) {
    return ggml2tt_type_internal(ggtype, arch) != tt::tt_metal::DataType::INVALID;
}

tt::tt_metal::Tensor realize_ggml_view_impl(const ggml_tensor * tensor) {
    // Since TTNN does not support the traditional view operation, we had to support it ourselves
    // This function, realize, extracts the data from the source tensor and creates a new tensor
    // that is separate from the source tensor. DO NOT eagerly call this function

    ggml_tensor * src0 = tensor->src[0];
    ggml_op       op   = tensor->op;

    // Do we really need to lazy evaluate this? Currently transpose is eagerly evaluated
    if (op == GGML_OP_TRANSPOSE) {
        auto parent = realize_ggml_view(src0);
        return ttnn::transpose(parent, -2, -1);
    }
    if (op == GGML_OP_VIEW) {
        tt::tt_metal::Tensor parent     = realize_ggml_view(tensor->view_src);
        std::array           dst_size   = std::to_array(tensor->ne);
        std::array           dst_stride = std::to_array(tensor->nb);
        std::array           src_size   = std::to_array(src0->ne);
        std::array           src_stride = std::to_array(src0->nb);
        size_t               offset     = tensor->view_offs;
        // ggml_backend_metalium_buffer_context* bufctx = ((ggml_tensor_extra_metalium*)tensor->extra)->bufctx;

        // TODO: Generalize this to use permute instead of transpose
        // FIXME: This is failing views in test-backend-ops
        // std::optional<std::pair<uint32_t, uint32_t>> axisswap;
        // for (int i = 0; i < ggml_n_dims(tensor); ++i) {
        //     size_t expected_stride = tensor->nb[0];
        //     for (int j = 0; j < i; ++j) {
        //         expected_stride *= tensor->ne[j];
        //     }
        //     // std::cout << "  Axis " << i << " stride: " << tensor->nb[i] << " expected: " << expected_stride << std::endl;
        //     if (tensor->nb[i] != expected_stride) {
        //         if (!axisswap) {
        //             axisswap = std::make_pair(i, 1000);
        //         } else if (axisswap->second == 1000) {
        //             axisswap->second = i;
        //         } else {
        //             GGML_ASSERT(false && "More than one axis swap detected");
        //         }
        //     }
        // }
        // TODO: Do something with axisswap. I think some ops needs this but it haven't crashed yet

        // Fast path if we can just return the parent tensor (view is a no-op)
        if (dst_size == src_size && dst_stride == src_stride && offset == 0) {
            return parent;
        }
        //TODO: Handle strided views (seems to be unused in the current codebase)
        std::array<uint32_t, GGML_MAX_DIMS> start;
        std::array<uint32_t, GGML_MAX_DIMS> end;

        // FIXME: Does not work when we are viewing into a permuted tensor. Sucks
        size_t remaining_offset = offset;
        for (size_t i = GGML_MAX_DIMS - 1; i < GGML_MAX_DIMS; i--) {
            start[i]         = remaining_offset / src_stride[i];
            end[i]           = dst_size[i] + start[i];
            remaining_offset = remaining_offset % src_stride[i];
        }
        std::reverse(start.begin(), start.end());
        std::reverse(end.begin(), end.end());
        tt::tt_metal::Tensor res;

        if (g_debug_flags.print_view) {
            // Debug prints to help debug complicated view operations
            std::cout << "\nrealize_ggml_view() OP: " << ggml_op_desc(tensor) << "\n";
            std::cout << "  dst name: " << tensor->name << "\n";
            std::cout << "  dst shape: " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " "
                      << tensor->ne[3] << "\n";
            std::cout << "  dst stride: " << tensor->nb[0] << " " << tensor->nb[1] << " " << tensor->nb[2] << " "
                      << tensor->nb[3] << "\n";
            std::cout << "  dst extra: " << tensor->extra << "\n";
            if (tensor->extra != nullptr) {
                const auto & tt_tensor = get_tt_tensor(tensor);
                std::cout << "  dst tensor: " << tt_tensor << "\n";
                if (tt_tensor.tensor_attributes) {
                    std::cout << "  dst tensor shape: " << tt_tensor.logical_shape() << "\n";
                }
            }
            std::cout << "  dst data: " << tensor->data << "\n";
            std::cout << "  dst view_src: " << tensor->view_src << "\n";
            std::cout << "  dst view_src shape: " << tensor->view_src->ne[0] << " " << tensor->view_src->ne[1] << " "
                      << tensor->view_src->ne[2] << " " << tensor->view_src->ne[3] << "\n";
            std::cout << "  dst view_src stride: " << tensor->view_src->nb[0] << " " << tensor->view_src->nb[1] << " "
                      << tensor->view_src->nb[2] << " " << tensor->view_src->nb[3] << "\n";
            std::cout << "  dst src0: " << src0 << "\n";
            std::cout << "  dst src1: " << tensor->src[1] << "\n";
            std::cout << "  src0 shape: " << src0->ne[0] << " " << src0->ne[1] << " " << src0->ne[2] << " "
                      << src0->ne[3] << "\n";
            std::cout << "  src0 stride: " << src0->nb[0] << " " << src0->nb[1] << " " << src0->nb[2] << " "
                      << src0->nb[3] << "\n";
            std::cout << "  src0 OP: " << ggml_op_desc(src0) << "\n";
            std::cout << "  TT parent shape: " << parent.logical_shape() << "\n";
            std::cout << "  TT slice start: " << start[0] << " " << start[1] << " " << start[2] << " " << start[3]
                      << "\n";
            std::cout << "  TT slice end: " << end[0] << " " << end[1] << " " << end[2] << " " << end[3] << "\n";
            std::cout << std::flush;
        }

        // Actually a reshape written as a view
        if (offset == 0 && ggml_nelements(src0) == ggml_nelements(tensor)) {
            res = reshape_tt_tensor_into_ggml(parent, tensor);
        }
        // Trying to convert a flat 1D tensor to N-D tensor (with an offset, else's it's the above case)
        else if (ggml_n_dims(src0) == 1 && ggml_n_dims(tensor) > 1) {
            // grab the source tensor, slice out the relevant part, and reshape it
            uint32_t                            offset_elements = offset / ggml_type_size(src0->type);
            uint32_t                            dst_volume      = (uint32_t) ggml_nelements(tensor);
            std::array<uint32_t, GGML_MAX_DIMS> start{ 0, 0, 0, offset_elements };
            std::array<uint32_t, GGML_MAX_DIMS> end({ 1, 1, 1, dst_volume + offset_elements });
            std::array<uint32_t, GGML_MAX_DIMS> step = { 1, 1, 1, 1 };
            tt::tt_metal::Tensor                tmp  = ttnn::slice(parent, start, end, step);
            res                                      = reshape_tt_tensor_into_ggml(tmp, tensor);
        }
        // The fast path, this is what TTNN is designed for (direct slicing)
        else {
            std::array<uint32_t, GGML_MAX_DIMS> step = { 1, 1, 1, 1 };
            res                                      = ttnn::slice(parent, start, end, step);
        }

        return res;
    }
    if (op == GGML_OP_RESHAPE) {
        const auto & t = realize_ggml_view(src0);
        return reshape_tt_tensor_into_ggml(t, tensor);
    }
    if (op == GGML_OP_PERMUTE) {
        std::array<int32_t, GGML_MAX_DIMS> permute;
        memcpy(permute.data(), tensor->op_params, sizeof(permute));

        int ndiff = 0;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            ndiff += permute[i] != i;
        }
        GGML_ASSERT(ndiff != 1);  // Logically impossible

        const auto & t = realize_ggml_view(src0);

        bool all_zero = true;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (permute[i] != 0) {
                all_zero = false;
                break;
            }
        }
        if (ndiff == 0 || all_zero) {
            return t;
        }

        ttsl::SmallVector<int64_t> permute_tt(GGML_MAX_DIMS);
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            permute_tt[i] = GGML_MAX_DIMS - permute[GGML_MAX_DIMS - i - 1] - 1;
        }
        ttsl::SmallVector<int64_t> permute_tt_real(GGML_MAX_DIMS);
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            permute_tt_real[permute_tt[i]] = i;
        }

        return ttnn::permute(t, permute_tt_real);
    }

    const auto &tt_tensor = get_tt_tensor(tensor);
    if (tt_tensor.tensor_attributes) {
        return tt_tensor;
    }

    if (is_view(tensor) && tensor->view_src != nullptr) {
        // recursivly resolve the source tensor
        return realize_ggml_view(tensor->view_src);
    }

    // HACK: Fallback path: if somehow the framework does not set the real tensor, we can make our own
    // FIXME: GCC says meta->tensor is NULL
    // auto tt_type = ggml2tt_type(tensor->type, meta->tensor->device()->arch());
    // auto shape = ttnn::Shape({uint32_t(tensor->ne[3]), uint32_t(tensor->ne[2]), uint32_t(tensor->ne[1]), uint32_t(tensor->ne[0])});
    // auto res = ttnn::tilize_with_zero_padding(ttnn::zeros(shape, tt::tt_metal::DataType::BFLOAT16).to_device(meta->tensor->device()), std::nullopt, tt_type);
    // meta->tensor = std::make_shared<tt::tt_metal::Tensor>(res);
    // return meta->tensor;
    fmt::println(stderr, "Tensor \"{}\" getting through fallback path. OP = {}, dtype={}", tensor->name,
                 ggml_op_name(tensor->op), ggml_type_name(tensor->type));
    GGML_ASSERT(false && "Fallback path not implemented");
}

tt::tt_metal::Tensor realize_ggml_view(const ggml_tensor * tensor) {
    auto                         res  = realize_ggml_view_impl(tensor);

    if (!ggml_tt_tensors_shape_equal(tensor, res)) {
        std::cout << "FATAL ERROR: Shape mismatch between TTNN and GGML after view op " << ggml_op_name(tensor->op)
                  << "\n"
                  << "  Result: " << res.logical_shape() << "\n"
                  << "  GGML expecting: " << tensor->ne[3] << " " << tensor->ne[2] << " " << tensor->ne[1] << " "
                  << tensor->ne[0] << "\n";
        GGML_ASSERT(ggml_tt_tensors_shape_equal(tensor, res));
    }
    return res;
}

bool is_view(const ggml_tensor * tensor) {
    return tensor->view_src != nullptr || tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE ||
           tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_PERMUTE;
}

tt::tt_metal::Tensor reshape_tt_tensor_into_ggml(const tt::tt_metal::Tensor & tensor, const struct ggml_tensor * node) {
    if (ggml_tt_tensors_shape_equal(node, tensor)) {
        return tensor;
    }

    std::array<uint32_t, GGML_MAX_DIMS> target_shape;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        target_shape[i] = node->ne[GGML_MAX_DIMS - i - 1];
    }

    // std::cerr << "Reshaping tensor " << tensor.logical_shape() << " to " << target_shape << std::endl;
    return ttnn::reshape(tensor, ttnn::Shape(target_shape));
}

const ggml_backend_metalium_debug_flags g_debug_flags = []() {
    auto parse_env = [](const char * env) -> bool {
        const char * val = std::getenv(env);
        if (val != nullptr) {
            std::string str(val);
            std::transform(str.begin(), str.end(), str.begin(), ::tolower);
            if (str != "0" && str != "false" && str != "no" && str != "off") {
                return true;
            }
        }
        return false;
    };

    return ggml_backend_metalium_debug_flags{
        .print_rejected_ops = parse_env("GGML_METALIUM_PRINT_REJECTED_OPS"),
        .print_view         = parse_env("GGML_METALIUM_PRINT_VIEW"),
        .cache_mm_transpose = parse_env(
            "GGML_METALIUM_CACHE_MM_TRANSPOSE"),  // GGML uses pre-transposed weights. Remove this flag when TT implements it
        .disable_program_cache = parse_env("GGML_METALIUM_DISABLE_PROGRAM_CACHE"),
        .experimental_ops      = parse_env("GGML_METALIUM_EXPERIMENTAL_OPS")
    };
}();

tt::tt_metal::Shape tt_shape_of(const ggml_tensor* ggtensor) {
  Shape::Container dims;
  for (int i = 0; i < GGML_MAX_DIMS; i++) {
    dims.push_back(ggtensor->ne[GGML_MAX_DIMS - i - 1]);
  }
  return tt::tt_metal::Shape(dims);
}

}  // namespace ggml_backend_metalium
