#include "ggml-metalium.h"

#include "buffer.hpp"
#include "fmt/base.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "hostdevcommon/common_values.hpp"
#include "tt-metalium/experimental/fabric/fabric.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/ccl/all_reduce/all_reduce.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/moreh/moreh_group_norm/moreh_group_norm.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/types/cluster_descriptor_types.hpp"
#include "utils.hpp"

#include <string.h>
#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <string_view>
#include <ttnn/core.hpp>
#include <ttnn/cpp/ttnn/operations/data_movement/gather/tosa/gather_tosa.hpp>
#include <ttnn/cpp/ttnn/operations/data_movement/scatter/tosa_scatter.hpp>
#include <ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp>
#include <ttnn/device.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/repeat/repeat.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>
#include <ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/nlp_kv_cache_load_slice.hpp>
#include <ttnn/operations/kv_cache/kv_cache.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <vector>

// #include "rope.hpp"
// #include "mul_mat.hpp"
// #include "soft_max.hpp"

using namespace std::literals;

extern void metalium_register_all_kernel();

namespace ggml_backend_metalium {

namespace {

struct ggml_backend_metalium_context {
    std::shared_ptr<ttnn::MeshDevice> device;
    int                               device_id;
    std::string                       name;

    ggml_backend_metalium_context(std::shared_ptr<ttnn::MeshDevice> device, int device_id, std::string name) :
        device(device),
        device_id(device_id),
        name(std::move(name)) {}
};

struct ggml_backend_metalium_device_context {
    std::shared_ptr<ttnn::MeshDevice> device    = nullptr;
    int                               device_id = -1;
    std::string                       name;
    std::string                       description;
};

struct ggml_backend_metalium_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

}  // anonymous namespace

static void dump_ggml_tensor_meta(const ggml_tensor * ggtensor) {
    std::cerr << "GGML tensor: " << ggtensor->name << "\n"
              << "  type: " << ggml_type_name(ggtensor->type) << "\n"
              << "  ne: " << ggtensor->ne[0] << " " << ggtensor->ne[1] << " " << ggtensor->ne[2] << " "
              << ggtensor->ne[3] << "\n"
              << "  nb: " << ggtensor->nb[0] << " " << ggtensor->nb[1] << " " << ggtensor->nb[2] << " "
              << ggtensor->nb[3] << "\n"
              << "  op: " << ggml_op_name(ggtensor->op) << "\n"
              << "  data: " << ggtensor->data << "\n"
              << "  src0: " << ggtensor->src[0] << "\n";
    if (ggtensor->src[0] != nullptr) {
        std::cerr << "    src0->name: " << ggtensor->src[0]->name << "\n"
                  << "    src0->type: " << ggml_type_name(ggtensor->src[0]->type) << "\n"
                  << "    src0->ne:   " << ggtensor->src[0]->ne[0] << " " << ggtensor->src[0]->ne[1] << " "
                  << ggtensor->src[0]->ne[2] << " " << ggtensor->src[0]->ne[3] << "\n"
                  << "    src0->nb:   " << ggtensor->src[0]->nb[0] << " " << ggtensor->src[0]->nb[1] << " "
                  << ggtensor->src[0]->nb[2] << " " << ggtensor->src[0]->nb[3] << "\n"
                  << "    src0->op:   " << ggml_op_name(ggtensor->src[0]->op) << "\n"
                  << "    src0->data: " << ggtensor->src[0]->data << "\n";
    }
    std::cerr << "  src1: " << ggtensor->src[1] << "\n";
    if (ggtensor->src[1] != nullptr) {
        std::cerr << "    src1->name: " << ggtensor->src[1]->name << "\n"
                  << "    src1->type: " << ggml_type_name(ggtensor->src[1]->type) << "\n"
                  << "    src1->ne: " << ggtensor->src[1]->ne[0] << " " << ggtensor->src[1]->ne[1] << " "
                  << ggtensor->src[1]->ne[2] << " " << ggtensor->src[1]->ne[3] << "\n"
                  << "    src1->nb: " << ggtensor->src[1]->nb[0] << " " << ggtensor->src[1]->nb[1] << " "
                  << ggtensor->src[1]->nb[2] << " " << ggtensor->src[1]->nb[3] << "\n"
                  << "    src1->op: " << ggml_op_name(ggtensor->src[1]->op) << "\n"
                  << "    src1->data: " << ggtensor->src[1]->data << "\n";
    }
    std::cerr << "  view_src: " << ggtensor->view_src << "\n";
    if (ggtensor->view_src != nullptr) {
        std::cerr << "    view_src->name: " << ggtensor->view_src->name << "\n"
                  << "    view_src->type: " << ggml_type_name(ggtensor->view_src->type) << "\n"
                  << "    view_src->ne: " << ggtensor->view_src->ne[0] << " " << ggtensor->view_src->ne[1] << " "
                  << ggtensor->view_src->ne[2] << " " << ggtensor->view_src->ne[3] << "\n"
                  << "    view_src->nb: " << ggtensor->view_src->nb[0] << " " << ggtensor->view_src->nb[1] << " "
                  << ggtensor->view_src->nb[2] << " " << ggtensor->view_src->nb[3] << "\n"
                  << "    view_src->op: " << ggml_op_name(ggtensor->view_src->op) << "\n"
                  << "    view_src->data: " << ggtensor->view_src->data << "\n";
    }
}

static ttnn::DeviceComputeKernelConfig make_compute_kernel_config(const ttnn::IDevice & device) {
    ttnn::DeviceComputeKernelConfig cfg;
    if (device.arch() == tt::ARCH::WORMHOLE_B0 || device.arch() == tt::ARCH::BLACKHOLE) {
        cfg = ttnn::WormholeComputeKernelConfig{ .math_fidelity    = MathFidelity::HiFi4,
                                                 .math_approx_mode = false,
                                                 .fp32_dest_acc_en = false,
                                                 .packer_l1_acc    = false };
    } else {
        fmt::println(stderr, "Unsupported device arch {} in make_compute_kernel_config", device.arch());
        abort();
    }
    return cfg;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Backend internal state tracking because GGML API does not allow
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Maintain all base addresses are unique
// TODO: Do we still need this since we already removed the virtual address mapping hack?
static size_t g_metalium_base_offset = 0;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Actual backend code
///////////////////////////////////////////////////////////////////////////////////////////////////////

static bool numpy_broadcast_rule(const ggml_tensor * t, const ggml_tensor * q) {
    int tdim = ggml_n_dims(t);
    int qdim = ggml_n_dims(q);

    int min_dim = tdim < qdim ? tdim : qdim;
    for (int i = 0; i < min_dim; i++) {
        if (t->ne[i] != q->ne[i] && t->ne[i] != 1 && q->ne[i] != 1) {
            return false;
        }
    }
    return true;
}

static bool is_integer_type(ggml_type type) {
    std::array<ggml_type, 4> integer_types = { GGML_TYPE_I32, GGML_TYPE_I16, GGML_TYPE_I8, GGML_TYPE_I64 };
    return std::find(integer_types.begin(), integer_types.end(), type) != integer_types.end();
}

inline static void ggml_metalium_op_src_sanity_check(const struct ggml_tensor * node, int idx) {
    GGML_ASSERT(node->src[idx] != NULL);
    GGML_ASSERT(node->src[idx]->extra != NULL);
    const auto & tt_tensor = get_tt_tensor(node->src[idx]);
    if (tt_tensor.tensor_attributes) {
        GGML_ASSERT(tt_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE);
        GGML_ASSERT(tt_tensor.layout() == tt::tt_metal::Layout::TILE);
    }
}

// Sanity check macros to ensure that the tensors are in the correct format and we won't crash
#define GGML_METALIUM_OP_SANITY_CHECK(_node)           GGML_ASSERT((_node)->extra != NULL);
// Check if the tensor is on the device (so we wont'e be using the CPU) as well as letting us crash early
#define GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, _idx) ggml_metalium_op_src_sanity_check(_node, _idx);
#define GGML_METALIUM_OP_SRC0_SANITY_CHECK(_node)      GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, 0)
#define GGML_METALIUM_OP_SRC1_SANITY_CHECK(_node)      GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, 1)

// Experimental flag to enable or disable custom mul_mat
// #define USE_CUSTOM_MUL_MAT

static bool ggml_backend_metalium_can_mul_mat(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // no permuted.
    if (ggml_is_permuted(src0) || ggml_is_permuted(src1)) {
        return false;
    }

    bool can_mul_mat = false;

    // Same batch dims.
    can_mul_mat |= (src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3]);

    // First mat has batch dims of 1.
    can_mul_mat |= (src0->ne[2] == 1 && src0->ne[3] == 1);

    return can_mul_mat;
}

static void ggml_backend_metalium_mul_mat(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);

    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    // bool can_be_processed_by_ttnn   = src0->ne[0] == src1->ne[0] && src0->ne[2] == 1 && src1->ne[2] == 1 &&
    //                                   (src0->ne[3] == src1->ne[3] || src0->ne[3] == 1);
    // GGML_ASSERT(can_be_processed_by_ttnn);

    GGML_TENSOR_BINARY_OP_LOCALS

    // Sometimes ggml gives a 0-element tensor, for that we just emit an empty
    // tensor of the correct shape.
    if (ggml_nelements(dst) == 0) {
        get_tt_tensor(dst) = ttnn::zeros(tt_shape_of(dst), ggml2tt_type(dst->type, ctx->device->arch()),
                                         tt::tt_metal::Layout::TILE, *ctx->device);
        return;
    }

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const auto & a = realize_ggml_view(src0);
    const auto & b = realize_ggml_view(src1);

    auto & dst_tt = get_tt_tensor(dst);
    dst_tt        = ttnn::operations::matmul::matmul(
        /*input_tensor_a=*/b,
        /*input_tensor_b=*/a,
        /*transpose_a=*/false,
        /*transpose_b=*/!tensor_extra::from(src1)->is_pretransposed,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/make_compute_kernel_config(*a.device()));
    if (tensor_extra::from(src1)->mesh_mapper.has_value()) {
        // the tensor is distributed, so we need a collective here.
        dst_tt = ttnn::all_reduce(dst_tt, ggml_backend_metalium_buffer_context::get(src1->buffer)->tp_axis());
    }
    GGML_ASSERT(dst_tt.storage_type() == tt::tt_metal::StorageType::DEVICE);
}

static bool ggml_backend_metalium_can_cpy(const struct ggml_tensor * dst) {
    if (is_integer_type(dst->type) || is_integer_type(dst->src[0]->type)) {
        return false;
    }
    // Destination must not be a view
    if (dst->op != GGML_OP_CPY) {
        return true;
    }
    ggml_tensor * src1 = dst->src[1];
    return !(ggml_is_permuted(src1) || is_view(src1));
}

static void ggml_backend_metalium_cpy(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    // Don't need sanity check since the copy is lazy
    // GGML_METALIUM_OP_SANITY_CHECK(dst);
    // GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    ggml_tensor * src0 = dst->src[0];

    // TODO: Check we are not writing into a view
    tt::tt_metal::Tensor res = realize_ggml_view(src0);
    if (!ggml_tt_tensors_shape_equal(dst, res)) {
        res = reshape_tt_tensor_into_ggml(res, dst);
    }
    auto result_type = ggml2tt_type(dst->type, res.device()->arch());
    if (res.dtype() != result_type) {
        res = ttnn::typecast(res, result_type);
    }
    if (dst->op == GGML_OP_CPY) {
        auto * src1 = dst->src[1];
        GGML_ASSERT(src1 != NULL);
        GGML_ASSERT(src1->extra != NULL);
        get_tt_tensor(src1) = res;
    }

    // TODO: Type cast to the appropriate type
    get_tt_tensor(dst) = res;
}

static bool ggml_backend_metalium_activations(ggml_backend_metalium_context * ctx,
                                              struct ggml_tensor *            dst,
                                              ggml_unary_op                   op) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];

    const auto & src_tensor = realize_ggml_view(src0);

    tt::tt_metal::Tensor ret;
    switch (op) {
        case GGML_UNARY_OP_ABS:
            ret = ttnn::abs(src_tensor);
            break;
        case GGML_UNARY_OP_SGN:
            ret = ttnn::sign(src_tensor);
            break;
        case GGML_UNARY_OP_NEG:
            ret = ttnn::neg(src_tensor);
            break;
        // Not accurate enough to pass unit tests
        case GGML_UNARY_OP_TANH:
            ret = ttnn::tanh(src_tensor);
            break;
        case GGML_UNARY_OP_ELU:
            ret = ttnn::elu(src_tensor, 1.0f);
            break;
        case GGML_UNARY_OP_RELU:
            ret = ttnn::relu(src_tensor);
            break;
        // Not accurate enough to pass unit tests
        case GGML_UNARY_OP_SIGMOID:
            ret = ttnn::sigmoid(src_tensor);
            break;
        case GGML_UNARY_OP_GELU:
            ret = ttnn::gelu(src_tensor, false);
            break;
        case GGML_UNARY_OP_GELU_QUICK:
            ret = ttnn::gelu(src_tensor);
            break;
        case GGML_UNARY_OP_SILU:
            ret = ttnn::silu(src_tensor);
            break;
        case GGML_UNARY_OP_HARDSWISH:
            ret = ttnn::hardswish(src_tensor);  // , 1.f/6.f, 0.5
            break;
        case GGML_UNARY_OP_HARDSIGMOID:
            ret = ttnn::hardsigmoid(src_tensor);  // , 1.f/6.f, 0.5
            break;
        case GGML_UNARY_OP_STEP:
            // TODO: Make sure the resulting data type matches the input
            ret = ttnn::typecast(ttnn::gtz(src_tensor), ggml2tt_type(dst->type, src_tensor.device()->arch()));
            break;
        case GGML_UNARY_OP_EXP:
            ret = ttnn::exp(src_tensor);
            break;
        case GGML_UNARY_OP_GELU_ERF:
            ret = ttnn::gelu(ttnn::erf(src_tensor), false);
            break;
        default:
            return false;
    }

    get_tt_tensor(dst) = std::move(ret);
    return true;
}

static void ggml_backend_metalium_leaky_relu(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0       = dst->src[0];
    const auto &               src_tensor = realize_ggml_view(src0);

    float negative_slope;
    GGML_ASSERT(dst->op_params != NULL);
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    get_tt_tensor(dst) = ttnn::leaky_relu(src_tensor, negative_slope);
}

static void ggml_backend_metalium_bin_op(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst, ggml_op op) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const auto & src_tensor0 = realize_ggml_view(src0);
    const auto & src_tensor1 = realize_ggml_view(src1);

    tt::tt_metal::Tensor ret;
    switch (op) {
        case GGML_OP_ADD:
            ret = ttnn::add(src_tensor0, src_tensor1);
            break;
        case GGML_OP_MUL:
            ret = ttnn::multiply(src_tensor0, src_tensor1);
            break;
        case GGML_OP_SUB:
            ret = ttnn::subtract(src_tensor0, src_tensor1);
            break;
        case GGML_OP_DIV:
            ret = ttnn::divide(src_tensor0, src_tensor1);
            break;
        default:
            GGML_ASSERT(false && "Unsupported binary operation");
    }

    get_tt_tensor(dst) = std::move(ret);
}

static bool ggml_backend_metalium_can_set(const struct ggml_tensor * dst) {
    int32_t params[5];
    memcpy(params, dst->op_params, sizeof(params));
    auto [nb1, nb2, nb3, offset, inplace] = std::to_array(params);

    if (offset >= nb3 || offset % nb1 != 0 || ggml_n_dims(dst->src[0]) < ggml_n_dims(dst->src[1]) ||
        ggml_n_dims(dst->src[1]) != 1) {
        return false;
    }

    return true;
}

static void ggml_backend_metalium_set(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);

    int32_t params[5];
    memcpy(params, dst->op_params, sizeof(params));
    auto [nb1, nb2, nb3, offset, inplace] = std::to_array(params);

    int idx       = offset / nb1;
    int batch_idx = offset / nb2;
    GGML_ASSERT(offset < nb3);
    GGML_ASSERT(offset % nb1 == 0);
    auto res = ttnn::update_cache(get_tt_tensor(dst->src[0]), get_tt_tensor(dst->src[1]), idx, batch_idx);
    if (!inplace) {
        get_tt_tensor(dst) = res;
    } else {
        get_tt_tensor(dst->src[0]) = res;
    }
}

static void ggml_backend_metalium_clamp(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    float data[2];
    memcpy(data, dst->op_params, sizeof(data));
    auto [min, max] = std::to_array(data);

    const auto & t = realize_ggml_view(dst->src[0]);

    get_tt_tensor(dst) = ttnn::clamp(t, min, max);
}

static void ggml_backend_metalium_scale(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    std::array<float, 2> params;
    memcpy(params.data(), dst->op_params, sizeof(params));
    auto [scale, bias] = params;

    const auto & t = realize_ggml_view(dst->src[0]);
    ttnn::Tensor res;
    if (bias == 0.f) {
        res = ttnn::multiply(t, scale);
    } else {
        res = ttnn::add(ttnn::multiply(t, scale, std::nullopt, ttnn::L1_MEMORY_CONFIG), bias);
    }
    // TODO: Support in-place scaling
    GGML_ASSERT(!is_view(dst->src[0]));

    get_tt_tensor(dst) = std::move(res);
}

static bool ggml_backend_metalium_can_get_rows(const struct ggml_tensor * dst) {
    const ggml_tensor * idxs = dst->src[1];
    // effectivly no-op
    if (idxs->ne[0] == 1 && idxs->ne[1] == 1 && idxs->ne[2] == 1 && idxs->ne[3] == 1 && ggml_n_dims(dst->src[0]) == 1) {
        return true;
    }

    const ggml_tensor * src = dst->src[0];
    if (is_integer_type(src->type)) {
        return false;
    }

    // FIXME: TTNN running into issues with large tensor....?
    if (idxs->ne[0] > 256) {
        return false;
    }

    // FIXME: Doesn't seem to be working correctly when batched
    if (src->ne[2] != 1 || src->ne[3] != 1) {
        return false;
    }

    if (idxs->ne[2] == 1 && src->ne[3] == 1 && !is_view(idxs)) {
        return true;
    }

    return false;
}

static void ggml_backend_metalium_get_rows(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    const auto &        t    = realize_ggml_view(dst->src[0]);
    const ggml_tensor * idxs = dst->src[1];
    if (idxs->ne[0] == 1 && idxs->ne[1] == 1 && idxs->ne[2] == 1 && idxs->ne[3] == 1 && ggml_n_dims(dst->src[0]) == 1) {
        get_tt_tensor(dst) = t;
    } else {
        const auto & idxs_tt  = get_tt_tensor(idxs);
        // The operation wants 3D tensor but we have 4D, op also wants index be 2d
        const auto & src3d    = t.reshape(t.logical_shape().to_rank(3));
        const auto & idx2d    = idxs_tt.reshape(idxs_tt.logical_shape().to_rank(2));
        auto         gathered = ttnn::tosa::gather(src3d, ttnn::tilize_with_zero_padding(idx2d), std::nullopt);
        gathered              = gathered.reshape(gathered.logical_shape().to_rank(4));
        get_tt_tensor(dst)    = std::move(gathered);
    }
}

static bool ggml_backend_metalium_can_set_rows(const struct ggml_tensor * dst) {
    // GGML has a weird order
    // result->src[0] = src
    // result->src[1] = idx
    // result->src[2] = dst
    const ggml_tensor * idxs = dst->src[1];
    // effectivly no-op
    if (idxs->ne[0] == 1 && idxs->ne[1] == 1 && idxs->ne[2] == 1 && idxs->ne[3] == 1 && ggml_n_dims(dst->src[0]) == 1) {
        return true;
    }

    const ggml_tensor * src = dst->src[0];
    if (is_integer_type(src->type)) {
        return false;
    }

    if (idxs->ne[2] == 1 && src->ne[3] == 1 && !is_view(idxs)) {
        return true;
    }

    return false;
}

static void ggml_backend_metalium_set_rows(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC_SANITY_CHECK(dst, 2);

    auto & dst_tt      = get_tt_tensor(dst);
    auto & real_dst_tt = get_tt_tensor(dst->src[2]);
    auto & idx_tt      = get_tt_tensor(dst->src[1]);

    auto                real_dst = realize_ggml_view(dst->src[2]);
    auto                src      = realize_ggml_view(dst->src[0]);
    const ggml_tensor * idxs     = dst->src[1];

    // Setting on a 1 row tensor is guarenteed to just be a replacment
    if (idxs->ne[0] == 1 && idxs->ne[1] == 1 && idxs->ne[2] == 1 && idxs->ne[3] == 1 && ggml_n_dims(dst->src[2]) == 1) {
        dst_tt      = src;
        real_dst_tt = src;
    } else {
        const auto & idx_tt     = get_tt_tensor(idxs);
        // The operation wants 3D tensor but we have 4D, op also wants index be 2d
        const auto & src3d      = src.reshape(src.logical_shape().to_rank(3));
        const auto & idx2d      = idx_tt.reshape(idx_tt.logical_shape().to_rank(2));
        const auto & real_dst3d = real_dst.reshape(real_dst.logical_shape().to_rank(3));
        ttnn::Tensor res = ttnn::tosa_scatter(real_dst3d, ttnn::tilize_with_zero_padding(idx2d), src3d, std::nullopt);
        fmt::println("res: {}", res.logical_shape());
        res         = res.reshape(res.logical_shape().to_rank(4));
        dst_tt      = res;
        real_dst_tt = res;
    }
}

static bool ggml_backend_metalium_can_norm(const struct ggml_tensor * dst, bool rms) {
    GGML_UNUSED(rms);
    // no hard checks but this seems to work well enough, else we run out of SRAM
    if (dst->ne[0] > 4096) {
        return false;
    }
    return true;
}

static void ggml_backend_metalium_norm(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst, bool rms) {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    float esp = 0;
    memcpy(&esp, dst->op_params, sizeof(esp));

    // HACK: the norm implementations in TTNN does not like size 1 tensors - we know the result is going to be sign(x)
    // so let's just make that
    const auto & t = realize_ggml_view(dst->src[0]);
    if (t.logical_shape()[-1] == 1) {
        get_tt_tensor(dst) = ttnn::typecast(ttnn::sign(t), t.dtype());
        return;
    }

    tt::tt_metal::Tensor res;
    if (rms) {
        res = ttnn::rms_norm(t, esp);
    } else {
        res = ttnn::layer_norm(t, esp);
    }
    get_tt_tensor(dst) = std::move(res);
}

static void ggml_backend_metalium_add1(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);

    const auto & t     = realize_ggml_view(dst->src[0]);
    const auto & q     = realize_ggml_view(dst->src[1]);
    get_tt_tensor(dst) = ttnn::add(t, q);
}

static void ggml_backend_metalium_sqrt(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    float esp = 0;
    memcpy(&esp, dst->op_params, sizeof(esp));

    const auto & t     = realize_ggml_view(dst->src[0]);
    get_tt_tensor(dst) = ttnn::sqrt(t);
}

static void ggml_backend_metalium_sqr(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    float esp = 0;
    memcpy(&esp, dst->op_params, sizeof(esp));

    const auto & t     = realize_ggml_view(dst->src[0]);
    get_tt_tensor(dst) = ttnn::square(t);
}

static bool ggml_backend_metalium_can_concat(const struct ggml_tensor * dst) {
    if (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_BF16 || dst->type == GGML_TYPE_F16) {
        return true;
    }
    return false;
}

static void ggml_backend_metalium_concat(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const auto & src_tensor0 = realize_ggml_view(src0);
    const auto & src_tensor1 = realize_ggml_view(src1);

    int32_t axis = 0;
    memcpy(&axis, dst->op_params, sizeof(axis));
    axis = GGML_MAX_DIMS - axis - 1;

    std::vector<tt::tt_metal::Tensor> targets{ src_tensor0, src_tensor1 };
    get_tt_tensor(dst) = ttnn::concat(targets, axis);
}

static bool ggml_backend_metalium_can_softmax(const struct ggml_tensor * dst) {
    GGML_UNUSED(dst);
    return true;
    // std::array<float, 2> params;
    // memcpy(&params, dst->op_params, sizeof(params));
    // auto [scale, max_bias] = params;
    // return max_bias == 0.f;
}

static void ggml_backend_metalium_softmax(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    std::array<float, 2> params;
    memcpy(&params, dst->op_params, sizeof(params));
    auto [scale, max_bias] = params;
#if 0
    auto x = *realize_ggml_view(dst->src[0]);
    if(dst->src[1] == NULL) {
        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(ttggml::soft_max(x, scale))
        };
    }
    else {
        auto mask = *realize_ggml_view(dst->src[1]);
        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(ttggml::soft_max(x, mask, scale))
        };
    }
#else

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    tt::tt_metal::Tensor x = realize_ggml_view(src0);
    // XXX: TTNN's own implementation does not handle broadcasting as GGML wants
    // if(src1 != nullptr) {
    //     auto mask = realize_ggml_view(src1);
    //     x = ttnn::operations::normalization::scale_mask_softmax(*t, scale, *mask);
    // }
    if (scale != 1.f) {
        x = ttnn::multiply(x, scale);
    }

    if (src1 != nullptr) {
        auto mask = realize_ggml_view(src1);
        if (max_bias == 0.f) {
            x = ttnn::add(x, mask);
        } else {
            // TODO: Replace this with a single operator that works on the mask
            // Also TODO: Make a new softmax that just does everything GGML wants
            const int n_head      = src0->ne[2];
            const int n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

            const float  m0     = powf(2.0f, -(max_bias) / n_head_log2);
            const float  m1     = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
            auto         slopes = ttnn::arange(0, n_head, 1, tt::tt_metal::DataType::FLOAT32, *x.device(),
                                               ttnn::DRAM_MEMORY_CONFIG, ttnn::TILE_LAYOUT);
            const auto & base   = ttnn::where(ttnn::lt(slopes, n_head_log2), m0, m1);
            const auto & exp    = ttnn::where(ttnn::lt(slopes, n_head_log2), ttnn::add(slopes, 1),
                                              ttnn::add(ttnn::multiply(ttnn::subtract(slopes, n_head_log2), 2.f), 1));
            slopes              = ttnn::pow(base, exp, tt::tt_metal::DataType::BFLOAT16);
            slopes              = ttnn::transpose(slopes.reshape(slopes.logical_shape().to_rank(4)), 1, 3);
            mask                = ttnn::multiply(mask, slopes);
            x                   = ttnn::add(x, mask);
        }
    }
    x                  = ttnn::softmax(x, 3);
    get_tt_tensor(dst) = std::move(x);
#endif
}

static void ggml_backend_metalium_cos(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];

    const auto & src   = realize_ggml_view(src0);
    get_tt_tensor(dst) = ttnn::cos(src);
}

static void ggml_backend_metalium_sin(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];

    const auto & src   = realize_ggml_view(src0);
    get_tt_tensor(dst) = ttnn::sin(src);
}

static void ggml_backend_metalium_log(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];

    const auto & src   = realize_ggml_view(src0);
    get_tt_tensor(dst) = ttnn::log(src);
}

static void ggml_backend_metalium_arange(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    auto *               device = get_tt_tensor(dst).device();
    std::array<float, 3> params;
    memcpy(&params, dst->op_params, sizeof(params));
    auto [start, end, step] = params;
    auto dtype              = ggml2tt_type(dst->type, device->arch());
    if (dtype == tt::tt_metal::DataType::INVALID) {
        fmt::println(stderr, "Unsupported GGML type {}", ggml_type_name(dst->type));
        GGML_ASSERT(false && "Unsupported GGML type");
    }

    auto tensor        = ttnn::arange(start, end, step, dtype, *device, ttnn::DRAM_MEMORY_CONFIG, ttnn::TILE_LAYOUT);
    get_tt_tensor(dst) = tensor.reshape(tensor.logical_shape().to_rank(4));
}

static void ggml_backend_metalium_group_norm(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    int   n_groups;
    float eps;
    memcpy(&n_groups, dst->op_params, sizeof(n_groups));
    memcpy(&eps, dst->op_params + 1, sizeof(eps));

    // XXX: Moreh's operators needs some cleanup
    const auto & tensor = realize_ggml_view(dst->src[0]);
    auto res = ttnn::moreh_group_norm(tensor, n_groups, eps, std::nullopt, std::nullopt,
                                      std::vector<bool>{ true, false, false }, std::nullopt, std::nullopt, std::nullopt,
                                      std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    GGML_ASSERT(res[0].has_value());
    get_tt_tensor(dst) = res[0].value();
}

static bool ggml_backend_metalium_can_repeat(const struct ggml_tensor * dst) {
    // TODO: File bug report that repear op should support UINT32
    if (dst->type == GGML_TYPE_I32) {
        return false;
    }
    ggml_tensor * src0 = dst->src[0];
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (dst->ne[i] % src0->ne[i] != 0) {
            return false;
        }
    }
    return true;
}

static void ggml_backend_metalium_repeat(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    ggml_tensor * src0 = dst->src[0];

    const auto &                tensor = realize_ggml_view(dst->src[0]);
    ttsl::SmallVector<uint32_t> repeats;
    repeats.resize(GGML_MAX_DIMS);
    int ndiff = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        auto repeat                    = dst->ne[i] / src0->ne[i];
        repeats[GGML_MAX_DIMS - i - 1] = repeat;
        ndiff += (repeat != 1);
    }
    if (ndiff == 0) {
        get_tt_tensor(dst) = tensor;
        return;
    } else {
        get_tt_tensor(dst) = ttnn::repeat(tensor, ttnn::Shape(repeats));
    }
}

static bool ggml_backend_metalium_can_outer_product(const struct ggml_tensor * dst) {
    auto num_ones_in_shape = [](const ggml_tensor * t) {
        int num_ones = 0;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (t->ne[i] == 1) {
                num_ones++;
            }
        }
        return num_ones;
    };
    return num_ones_in_shape(dst->src[0]) == 3 && num_ones_in_shape(dst->src[1]) == 3;
}

static void ggml_backend_metalium_outer_product(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const auto & src0 = realize_ggml_view(dst->src[0]);
    const auto & src1 = realize_ggml_view(dst->src[1]);

    auto res = ttnn::outer(src1, src0);
    // HACK: GGML and TT has different ideas about the shape of the result, sometimes
    if (!ggml_tt_tensors_shape_equal(dst, res)) {
        // Magic herustics
        if (dst->ne[3] == res.logical_shape()[2]) {
            res = ttnn::transpose(res, 0, 2);
        } else if (dst->ne[2] == res.logical_shape()[2]) {
            res = ttnn::transpose(res, 1, 2);
        } else {
            std::cerr << "GGML shape: " << dst->ne[0] << ", " << dst->ne[1] << ", " << dst->ne[2] << ", " << dst->ne[3]
                      << "\n";
            std::cerr << "TT shape: " << res.logical_shape() << "\n";
            GGML_ASSERT(false && "Unsupported outer product shape mismatch");
        }
    }
    get_tt_tensor(dst) = res;
}

static void ggml_backend_metalium_sum(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    auto                              t = realize_ggml_view(dst->src[0]);
    ttnn::WormholeComputeKernelConfig cfg{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = true, .packer_l1_acc = true
    };
    get_tt_tensor(dst) = ttnn::sum(t, std::nullopt, false, std::nullopt, cfg);
}

static bool ggml_backend_metalium_can_sum_rows(const struct ggml_tensor * dst) {
    // FIXME: Don't know why but it's broken for these cases
    ggml_tensor * src0 = dst->src[0];
    return src0->ne[2] == 1 && src0->ne[3] == 1;
}

static void ggml_backend_metalium_sum_rows(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    auto t             = realize_ggml_view(dst->src[0]);
    get_tt_tensor(dst) = ttnn::sum(t, 3);
}

static bool ggml_backend_metalium_can_glu(const struct ggml_tensor * dst) {
    constexpr std::array<ggml_glu_op, 5> supported = { GGML_GLU_OP_REGLU, GGML_GLU_OP_GEGLU_ERF,
                                                       GGML_GLU_OP_GEGLU_QUICK, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU };
    if (std::find_if(supported.begin(), supported.end(), [&](ggml_glu_op op) { return ggml_get_glu_op(dst) == op; }) ==
        supported.end()) {
        return false;
    }

    bool split = dst->src[1] != NULL;
    if (split) {
        return true;
    }

    return dst->src[0]->ne[0] % 2 == 0;
}

static void ggml_backend_metalium_glu(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    ttnn::Tensor a;
    ttnn::Tensor b;
    int          swap  = ggml_get_op_params_i32(dst, 1);
    bool         split = dst->src[1] != NULL;

    if (split) {
        a = realize_ggml_view(dst->src[1]);
        b = realize_ggml_view(dst->src[0]);
    } else {
        auto    t = realize_ggml_view(dst->src[0]);
        // split along the last dimension
        int64_t w = dst->ne[0];

        using Slice = std::array<uint32_t, GGML_MAX_DIMS>;
        Slice mid   = { uint32_t(w), uint32_t(dst->ne[1]), uint32_t(dst->ne[2]), uint32_t(dst->ne[3]) };
        std::reverse(mid.begin(), mid.end());
        Slice mid_start = { uint32_t(w), 0, 0, 0 };
        std::reverse(mid_start.begin(), mid_start.end());
        Slice end = { uint32_t(w * 2), uint32_t(dst->ne[1]), uint32_t(dst->ne[2]), uint32_t(dst->ne[3]) };
        std::reverse(end.begin(), end.end());
        Slice begin  = { 0, 0, 0, 0 };
        Slice stride = { 1, 1, 1, 1 };

        a = ttnn::slice(t, mid_start, end, stride);
        b = ttnn::slice(t, begin, mid, stride);
    }

    if (swap) {
        std::swap(a, b);
    }

    ttnn::Tensor res;
    switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_REGLU:
            res = ttnn::multiply(a, ttnn::relu(b, ttnn::L1_MEMORY_CONFIG));
            break;
        case GGML_GLU_OP_GEGLU_ERF:  // ?
        case GGML_GLU_OP_GEGLU_QUICK:
        case GGML_GLU_OP_GEGLU:
            res = ttnn::multiply(a, ttnn::gelu(b, false, ttnn::L1_MEMORY_CONFIG));
            break;
        case GGML_GLU_OP_SWIGLU:
            res = ttnn::multiply(a, ttnn::swish(b, ttnn::L1_MEMORY_CONFIG));
            break;
        default:
            GGML_ASSERT(false && "Unsupported GLU operation");
    }

    get_tt_tensor(dst) = std::move(res);
}

static bool ggml_backend_metalium_can_rope(const struct ggml_tensor * dst) {
    std::array<int32_t, 5> int_params;
    memcpy(int_params.data(), dst->op_params, sizeof(int_params));
    auto [n_past, n_dims, mode, n_ctx, n_ctx_orig] = int_params;

    if (mode == GGML_ROPE_TYPE_NEOX) {
        return n_dims % 64 == 0;
    }
    if (mode == GGML_ROPE_TYPE_NORMAL) {
        ggml_tensor * ff = dst->src[2];
        if (dst->src[2]) {
            return !ggml_is_quantized(ff->type)                 // we do sub-tile hacking
                   && ff->ne[0] % 32 == 0 && n_dims % 32 == 0;  // XXX: This case fails
        }
        return n_dims % 32 == 0;
    }

    return false;
}

static void ggml_backend_metalium_rope(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

#if 0
    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;
    std::array<int32_t, 5> int_params;
    memcpy(int_params.data(), dst->op_params, sizeof(int_params));
    auto [
        n_past,
        n_dims,
        mode,
        n_ctx,
        n_ctx_orig ] = int_params;

    std::array<float, 6> float_params;
    memcpy(float_params.data(), dst->op_params + int_params.size() , sizeof(float_params));
    auto [
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast,
        beta_slow
    ] = float_params;

    auto res = [&](){
        if(dst->src[2]) {
            return ttggml::rope(
                *realize_ggml_view(dst->src[0]),
                *realize_ggml_view(dst->src[1]),
                *realize_ggml_view(dst->src[2]),
                n_dims,
                mode == GGML_ROPE_TYPE_NEOX ? ttggml::RoPEType::NeoX : ttggml::RoPEType::Normal,
                n_ctx_orig,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow);
        }
        return ttggml::rope(
            *realize_ggml_view(dst->src[0]),
            *realize_ggml_view(dst->src[1]),
            n_dims,
            mode == GGML_ROPE_TYPE_NEOX ? ttggml::RoPEType::NeoX : ttggml::RoPEType::Normal,
            n_ctx_orig,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow);
    }();
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(res)),
    };
#else
    GGML_ABORT("not implemented");
#endif
}

static bool ggml_backend_metalium_can_flash_attn(const struct ggml_tensor * dst) {
    if (!g_debug_flags.experimental_ops) {
        return false;
    }
    auto follow_tensor_upstream = [](const ggml_tensor * tensor) -> const ggml_tensor * {
        if (!tensor) {
            return NULL;
        }
        while (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_PERMUTE) {
            tensor = tensor->src[0];
        }
        return tensor;
    };
    const ggml_tensor * q    = follow_tensor_upstream(dst->src[0]);
    const ggml_tensor * k    = follow_tensor_upstream(dst->src[1]);
    const ggml_tensor * v    = follow_tensor_upstream(dst->src[2]);
    const ggml_tensor * mask = follow_tensor_upstream(dst->src[3]);

    std::array<float, 3> params;
    memcpy(params.data(), dst->op_params, sizeof(float) * 3);
    auto [scale, max_bias, logit_softcap] = params;

    if (max_bias != 0.f) {
        return false;
    }

    // Examoke input
    // REJECT op FLASH_ATTN_EXT (__fattn__-6)
    //   src0 shape [64 32 32 1], dtype = f32, name = 'Qcur-6 (view) (permuted)'
    //   src1 shape [64 1536 8 1], dtype = f16, name = 'cache_k_l6 (view) (permuted)'
    //   src2 shape [64 1536 8 1], dtype = f16, name = 'cache_v_l6 (view) (permuted)'
    //   src3 shape [1536 64 1 1], dtype = f16, name = ' (copy)'
    //   FlashAttention debug details:
    //     src0 follow - query shape [64 32 32 1], dtype = f32, name = 'Qcur-6 (view)'
    //     src1 follow - key shape [64 8 1536 1], dtype = f16, name = 'cache_k_l6 (view)'
    //     src2 follow - value shape [64 8 1536 1], dtype = f16, name = 'cache_v_l6 (view)'
    //     src3 follow - mask shape [1536 64 1 1], dtype = f16, name = ' (copy)'
    //
    // GGML:
    // q:    [n_embd_k, n_batch,     n_head,    ne3 ]
    // k:    [n_embd_k, n_kv,        n_head_kv, ne3 ]
    // v:    [n_embd_v, n_kv,        n_head_kv, ne3 ] !! not transposed !!
    // mask: [n_kv,     n_batch_pad, ne32,      ne33] !! n_batch_pad = GGML_PAD(n_batch, GGML_KQ_MASK_PAD) !!
    // res:  [n_embd_v, n_head,      n_batch,   ne3 ] !! permuted !!
    //
    // TT
    // input_tensor_q (ttnn.Tensor): the input tensor [1 x b x nh x dh]
    // input_tensor_k (ttnn.Tensor): the input tensor [b x nkv x   s x dh]
    // input_tensor_v (ttnn.Tensor): the input tensor [b x nkv x   s x dh]

    int64_t ne3 = q->ne[3];
    if (ne3 != 1) {
        return false;
    }
    if (k->ne[3] != 1 || v->ne[3] != 1) {
        return false;
    }
    if (k->ne[1] < 32 || v->ne[1] < 32 || k->ne[1] % 32 != 0 || v->ne[1] % 32 != 0) {
        return false;
    }
    int64_t b = q->ne[1];
    // Either we don't need to broadcast or we broadcast for them
    if (mask && mask->ne[2] != 1 && !(mask->ne[3] == 1 || mask->ne[3] == b)) {
        return false;
    }
    // The op does not support broadcasting mask
    if (mask && (mask->ne[0] != k->ne[1] || mask->ne[1] != q->ne[1])) {
        return false;
    }

    return true;
}

static void ggml_backend_metalium_flash_attn(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    auto follow_tensor_upstream = [](const ggml_tensor * tensor) -> const ggml_tensor * {
        if (!tensor) {
            return NULL;
        }
        while (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_PERMUTE) {
            tensor = tensor->src[0];
        }
        return tensor;
    };
    const ggml_tensor * q    = follow_tensor_upstream(dst->src[0]);
    const ggml_tensor * k    = follow_tensor_upstream(dst->src[1]);
    const ggml_tensor * v    = follow_tensor_upstream(dst->src[2]);
    const ggml_tensor * mask = follow_tensor_upstream(dst->src[3]);

    std::array<float, 3> params;
    memcpy(params.data(), dst->op_params, sizeof(float) * 3);
    auto [scale, max_bias, logit_softcap] = params;

    auto qt = realize_ggml_view(q);
    auto kt = realize_ggml_view(k);
    auto vt = realize_ggml_view(v);

    uint32_t b = qt.logical_shape()[1];
    if (kt.logical_shape()[0] != b) {
        ttnn::Shape repeat_factor({ b, 1, 1, 1 });
        kt = ttnn::repeat(kt, repeat_factor);
    }

    if (vt.logical_shape()[0] != b) {
        ttnn::Shape repeat_factor({ b, 1, 1, 1 });
        vt = ttnn::repeat(vt, repeat_factor);
    }

    std::optional<ttnn::Tensor> mask_tensor;
    if (mask) {
        mask_tensor = realize_ggml_view(mask);
        if (mask_tensor->logical_shape()[0] != b) {
            ttnn::Shape repeat_factor({ b, 1, 1, 1 });
            *mask_tensor = ttnn::repeat(*mask_tensor, repeat_factor);
        }
    }

    auto res = ttnn::transformer::scaled_dot_product_attention_decode(
        qt, kt, vt, false, mask_tensor, std::vector<uint32_t>{}, std::nullopt, std::nullopt, scale);

    // HACK: I have no idea why
    if (!ggml_tt_tensors_shape_equal(dst, res)) {
        res = ttnn::transpose(res, 1, 2);
    }
    get_tt_tensor(dst) = res;
}

/// TTNN's implementation can handle weight and bias in one operation.
static void ggml_metalium_fused_norm(ggml_backend_metalium_context * ctx,
                                     ggml_tensor *                   norm,
                                     ggml_tensor *                   scale,
                                     ggml_tensor *                   bias) {
    GGML_UNUSED(ctx);

    const ttnn::Tensor &              input = realize_ggml_view(norm->src[0]);
    std::optional<const ttnn::Tensor> scale_tt;
    std::optional<const ttnn::Tensor> bias_tt;
    ttnn::Tensor *                    last = &get_tt_tensor(norm);

    if (scale != nullptr) {
        GGML_ASSERT(scale->src[0] == norm);
        scale_tt.emplace(realize_ggml_view(scale->src[1]));
        last = &get_tt_tensor(scale);
    }
    if (bias != nullptr) {
        GGML_ASSERT(bias->src[0] == (scale != nullptr ? scale : norm));
        bias_tt.emplace(realize_ggml_view(bias->src[1]));
        last = &get_tt_tensor(bias);
    }

    float esp = 0;
    memcpy(&esp, norm->op_params, sizeof(esp));

    // Same rationale as the single-element case in the standalone norm op, but
    // here we have to apply the scale and bias if needed.
    if (input.logical_shape()[-1] == 1) {
        *last = ttnn::typecast(ttnn::sign(input), input.dtype());
        if (scale_tt) {
            *last = ttnn::multiply_(*last, scale_tt.value());
        }
        if (bias_tt) {
            *last = ttnn::add_(*last, bias_tt.value());
        }
        return;
    }

    tt::tt_metal::Tensor res;

    switch (norm->op) {
        case GGML_OP_RMS_NORM:
            *last = ttnn::rms_norm(input, esp, scale_tt, bias_tt);
            break;
        case GGML_OP_NORM:
            *last = ttnn::layer_norm(input, esp, scale_tt, bias_tt);
            break;
        default:
            GGML_ABORT("unexpected op type for norm node %s: %d", norm->name, norm->type);
    }
}

static void ggml_metalium_fused_ffn(ggml_backend_metalium_context * ctx,
                                    const ggml_tensor *             ffn_gate,
                                    const ggml_tensor *             ffn_up,
                                    const ggml_tensor *             ffn_glu,
                                    const ggml_tensor *             ffn_down,
                                    const ggml_tensor *             ffn_add) {
    // Check this subgraph is actually what we expect
    GGML_ASSERT(ffn_gate->src[1] == ffn_up->src[1]);
    GGML_ASSERT(ffn_glu->src[0] == ffn_gate);
    GGML_ASSERT(ffn_glu->src[1] == ffn_up);
    GGML_ASSERT(ffn_down->src[1] == ffn_glu);
    GGML_ASSERT(ffn_add->src[0] == ffn_down);

    // More useful names.
    ggml_tensor * hidden_input  = ffn_gate->src[1];
    ggml_tensor * gate_weight   = ffn_gate->src[0];
    ggml_tensor * up_weight     = ffn_up->src[0];
    ggml_tensor * down_weight   = ffn_down->src[0];
    ggml_tensor * attn_residual = ffn_add->src[1];

    // Empty batches happens during warmup. We can early return at this point
    // after all the setup is done.
    if (ggml_nelements(ffn_add) == 0) {
        tensor_extra::from(ffn_add)->tensor =
            ttnn::zeros(tt_shape_of(ffn_add), ggml2tt_type(ffn_add->type, ctx->device->arch()),
                        tt::tt_metal::Layout::TILE, *ctx->device);
        return;
    }

    ttnn::operations::unary::UnaryOpType activation;
    switch (ggml_get_glu_op(ffn_glu)) {
        case GGML_GLU_OP_SWIGLU:
            activation = ttnn::operations::unary::UnaryOpType::SILU;
            break;
        case GGML_GLU_OP_REGLU:
            activation = ttnn::operations::unary::UnaryOpType::RELU;
            break;
        case GGML_GLU_OP_GEGLU:
            activation = ttnn::operations::unary::UnaryOpType::GELU;
            break;
        default:
            GGML_ABORT("unsupported GLU op");
    }

    auto hidden_tt = realize_ggml_view(hidden_input);

    auto batch_size                 = hidden_input->ne[3] * hidden_input->ne[2] * hidden_input->ne[1];
    auto intermediate_memory_config = batch_size > 512 ? ttnn::DRAM_MEMORY_CONFIG : ttnn::L1_MEMORY_CONFIG;

    auto gate_out = ttnn::matmul(
        /*input_tensor_a=*/hidden_tt,
        /*input_tensor_b=*/realize_ggml_view(gate_weight),
        /*transpose_a=*/false,
        /*transpose_b=*/!tensor_extra::from(gate_weight)->is_pretransposed,
        /*memory_config=*/ttnn::L1_MEMORY_CONFIG,
        /*dtype=*/ggml2tt_type(ffn_gate->type, ctx->device->arch()),
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/make_compute_kernel_config(*ctx->device));
    auto up_out = ttnn::matmul(
        /*input_tensor_a=*/hidden_tt,
        /*input_tensor_b=*/realize_ggml_view(up_weight),
        /*transpose_a=*/false,
        /*transpose_b=*/!tensor_extra::from(up_weight)->is_pretransposed,
        /*memory_config=*/ttnn::L1_MEMORY_CONFIG,
        /*dtype=*/ggml2tt_type(ffn_up->type, ctx->device->arch()),
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/make_compute_kernel_config(*ctx->device));

    ttnn::multiply_(up_out, gate_out,
                    /*post_activations=*/{},
                    /*lhs_activations=*/{},
                    /*rhs_activations=*/{ { activation } });
    gate_out.deallocate();

    GGML_ASSERT(ffn_down->src[1] == ffn_glu);
    auto down_out = ttnn::matmul(
        /*input_tensor_a=*/up_out,
        /*input_tensor_b=*/realize_ggml_view(down_weight),
        /*transpose_a=*/false,
        /*transpose_b=*/!tensor_extra::from(down_weight)->is_pretransposed,
        /*memory_config=*/ttnn::L1_MEMORY_CONFIG,
        /*dtype=*/ggml2tt_type(ffn_down->type, ctx->device->arch()),
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/make_compute_kernel_config(*ctx->device));
    up_out.deallocate();

    // If tensor parallelism is enabled, all-reduce the down output across devices.
    auto * bufctx = ggml_backend_metalium_buffer_context::get(ffn_add->buffer);
    if (bufctx->tp_ne().has_value()) {
        down_out = ttnn::all_reduce(down_out, bufctx->tp_axis());
    }

    tensor_extra::from(ffn_add)->tensor = ttnn::add(down_out, realize_ggml_view(attn_residual),
                                                    /*output_dtype=*/std::nullopt,
                                                    /*memory_config=*/ttnn::DRAM_MEMORY_CONFIG);
    down_out.deallocate();
}

// backend interface

static const char * ggml_backend_metalium_name(ggml_backend_t backend) {
    return "Metalium";

    GGML_UNUSED(backend);
}

static void ggml_backend_metalium_free(ggml_backend_t backend) {
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *) backend->context;
    delete ctx;
    delete backend;
}

struct ggml_backend_metalium_buffer_type_context {
    std::shared_ptr<ttnn::MeshDevice> device = nullptr;
    std::string                       name;
};

static const char * ggml_backend_metalium_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metalium_buffer_type_context * ctx = (ggml_backend_metalium_buffer_type_context *) buft->context;

    return ctx->name.c_str();
}

static size_t ggml_backend_metalium_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // Not using this. Metalium's allication model is not compatible with GGML's allocator
    return 128;
    GGML_UNUSED(buft);
}

// NOTE: I might need to add a metalium tensor wrapper to work around TT tensors have hardware-tagged data types
//       and GGML tensors does not specify the data type during tensor creation.
static size_t ggml_backend_metalium_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_metalium_buffer_type_context * ctx = (ggml_backend_metalium_buffer_type_context *) buft->context;
    return ctx->device->num_dram_channels() * (size_t) ctx->device->dram_size_per_channel();
}

static size_t ggml_backend_metalium_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                               const ggml_tensor *        tensor) {
    // Not using this. Metalium's allication model is not compatible with GGML's allocator
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_metalium_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                            size_t                     size) {
    ggml_backend_metalium_buffer_type_context * buft_ctx = (ggml_backend_metalium_buffer_type_context *) buft->context;

    // FIXME: GGML unit tests fails if I don't add some additional memory to the buffer beyond the requested size
    size_t                                 alloc_size = size + 4096 * 1024;
    // real allocation is deferred until the first tensor is set because we don't know the underlying tensor type yet
    ggml_backend_metalium_buffer_context * ctx =
        new ggml_backend_metalium_buffer_context{ .ggml_buffer_size_bytes = size,
                                                  .name                   = buft_ctx->name,
                                                  .device                 = buft_ctx->device,
                                                  .base_offset            = g_metalium_base_offset,

                                                  .metadata_to_free = {} };
    g_metalium_base_offset += alloc_size;
    // std::cout << "Allocating buffer of size " << size << " bytes\n";
    return ggml_backend_buffer_init(buft, ggml_backend_metalium_buffer_interface, ctx, alloc_size);
}

static bool ggml_backend_metalium_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static ggml_backend_buffer_type_i ggml_backend_metalium_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_metalium_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_metalium_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_metalium_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_metalium_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_metalium_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_metalium_buffer_type_is_host,
};

static ggml_backend_buffer_type_t ggml_backend_metalium_buffer_type(ggml_backend_dev_t                     dev,
                                                                    ggml_backend_metalium_device_context * dev_ctx) {
    auto                                device_id = dev_ctx->device_id;
    ggml_backend_metalium_reg_context * regctx    = (ggml_backend_metalium_reg_context *) (dev->reg->context);

    static std::map<int, ggml_backend_buffer_type>                              buffer_type_map;
    static std::set<std::unique_ptr<ggml_backend_metalium_buffer_type_context>> buffer_type_context_deleter;
    auto                                                                        it = buffer_type_map.find(device_id);
    if (it != buffer_type_map.end()) {
        return &it->second;
    }

    auto bufctx = std::make_unique<ggml_backend_metalium_buffer_type_context>(ggml_backend_metalium_buffer_type_context{
        .device = dev_ctx->device,
        .name   = "METALIUM" + std::to_string(device_id),
    });
    auto * bufctx_ptr = bufctx.get();
    buffer_type_context_deleter.insert(std::move(bufctx));

    // TODO: Make sure the device_id we got is valid
    buffer_type_map[device_id] = {
        /* .iface    = */ ggml_backend_metalium_buffer_type_interface,
        /* .device   = */ regctx->devices[0],
        /* .context  = */ bufctx_ptr,
    };
    return &buffer_type_map[device_id];
}

static enum ggml_status ggml_backend_metalium_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *) backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        // Bypass post conition checks for these ops because they are evaluated lazily
        if (node->op == GGML_OP_VIEW || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_PERMUTE) {
            continue;
        }

        // Check for fused ops
        auto mul_mat_glu = { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_MUL_MAT, GGML_OP_ADD };
        if (ggml_can_fuse_subgraph(cgraph, i, mul_mat_glu, { i + 4 })) {
            const ggml_tensor * ffn_gate = cgraph->nodes[i];
            const ggml_tensor * ffn_up   = cgraph->nodes[i + 1];
            const ggml_tensor * ffn_glu  = cgraph->nodes[i + 2];
            const ggml_tensor * ffn_down = cgraph->nodes[i + 3];
            ggml_tensor *       ffn_add  = cgraph->nodes[i + 4];

            ggml_metalium_fused_ffn(ctx, ffn_gate, ffn_up, ffn_glu, ffn_down, ffn_add);
            i += mul_mat_glu.size() - 1;
            continue;
        }

        // Add the other cases for norm as needed.
        auto norm_scale = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (ggml_can_fuse_subgraph(cgraph, i, norm_scale, { i + 1 })) {
            ggml_tensor * norm  = cgraph->nodes[i];
            ggml_tensor * scale = cgraph->nodes[i + 1];
            ggml_metalium_fused_norm(ctx, norm, scale, nullptr);
            i += norm_scale.size() - 1;
            continue;
        }

        // std::cout << ggml_op_name(node->op) << " node " << node->name << " with address " << node->data << std::endl;
        switch (node->op) {
            case GGML_OP_UNARY:
                {
                    ggml_unary_op unary_op = ggml_get_unary_op(node);
                    bool          ok       = false;
                    switch (unary_op) {
                        case GGML_UNARY_OP_ABS:
                        case GGML_UNARY_OP_SGN:
                        case GGML_UNARY_OP_NEG:
                        case GGML_UNARY_OP_TANH:
                        case GGML_UNARY_OP_ELU:
                        case GGML_UNARY_OP_RELU:
                        case GGML_UNARY_OP_SIGMOID:
                        case GGML_UNARY_OP_GELU:
                        case GGML_UNARY_OP_GELU_QUICK:
                        case GGML_UNARY_OP_SILU:
                        case GGML_UNARY_OP_HARDSWISH:
                        case GGML_UNARY_OP_HARDSIGMOID:
                        case GGML_UNARY_OP_STEP:
                        case GGML_UNARY_OP_EXP:
                        case GGML_UNARY_OP_GELU_ERF:
                            ok = ggml_backend_metalium_activations(ctx, node, unary_op);
                            break;
                        default:
                            fprintf(stderr, "%s: unsupported unary op %s\n", __func__, ggml_unary_op_name(unary_op));
                    }
                    GGML_ASSERT(ok && "Failed to execute unary op");
                    break;
                }
            case GGML_OP_LEAKY_RELU:
                ggml_backend_metalium_leaky_relu(ctx, node);
                break;
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
            case GGML_OP_MUL:
                ggml_backend_metalium_bin_op(ctx, node, node->op);
                break;
            case GGML_OP_MUL_MAT:
                ggml_backend_metalium_mul_mat(ctx, node);
                break;
            case GGML_OP_OUT_PROD:
                ggml_backend_metalium_outer_product(ctx, node);
                break;

            case GGML_OP_CONT:
            case GGML_OP_CPY:
            case GGML_OP_DUP:
                ggml_backend_metalium_cpy(ctx, node);
                break;
            case GGML_OP_SET:
                ggml_backend_metalium_set(ctx, node);
                break;

            case GGML_OP_CLAMP:
                ggml_backend_metalium_clamp(ctx, node);
                break;

            case GGML_OP_SCALE:
                ggml_backend_metalium_scale(ctx, node);
                break;

            case GGML_OP_GET_ROWS:
                ggml_backend_metalium_get_rows(ctx, node);
                break;

            case GGML_OP_NORM:
                ggml_backend_metalium_norm(ctx, node, false);
                break;

            case GGML_OP_RMS_NORM:
                ggml_backend_metalium_norm(ctx, node, true);
                break;

            case GGML_OP_ADD1:
                ggml_backend_metalium_add1(ctx, node);
                break;

            case GGML_OP_SQRT:
                ggml_backend_metalium_sqrt(ctx, node);
                break;

            case GGML_OP_SQR:
                ggml_backend_metalium_sqr(ctx, node);
                break;

            case GGML_OP_CONCAT:
                ggml_backend_metalium_concat(ctx, node);
                break;

            case GGML_OP_SOFT_MAX:
                ggml_backend_metalium_softmax(ctx, node);
                break;

            case GGML_OP_COS:
                ggml_backend_metalium_cos(ctx, node);
                break;

            case GGML_OP_SIN:
                ggml_backend_metalium_sin(ctx, node);
                break;

            case GGML_OP_LOG:
                ggml_backend_metalium_log(ctx, node);
                break;

            case GGML_OP_ARANGE:
                ggml_backend_metalium_arange(ctx, node);
                break;

            case GGML_OP_GROUP_NORM:
                ggml_backend_metalium_group_norm(ctx, node);
                break;

            case GGML_OP_REPEAT:
                ggml_backend_metalium_repeat(ctx, node);
                break;

            case GGML_OP_SUM:
                ggml_backend_metalium_sum(ctx, node);
                break;

            case GGML_OP_SUM_ROWS:
                ggml_backend_metalium_sum_rows(ctx, node);
                break;

            case GGML_OP_GLU:
                ggml_backend_metalium_glu(ctx, node);
                break;

            case GGML_OP_ROPE:
                ggml_backend_metalium_rope(ctx, node);
                break;

            case GGML_OP_FLASH_ATTN_EXT:
                ggml_backend_metalium_flash_attn(ctx, node);
                break;

            case GGML_OP_SET_ROWS:
                ggml_backend_metalium_set_rows(ctx, node);
                break;

            case GGML_OP_NONE:
                break;

            default:
                fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                GGML_ASSERT(false);
        }
        const auto & node_tt = get_tt_tensor(node);
        // std::cout << "Executed " << ggml_op_desc(node) << " with address " << node->data << " and shape " << meta->tensor->logical_shape() << ", GGML wants " << node->ne[0] << " " << node->ne[1] << " " << node->ne[2] << " " << node->ne[3] << std::endl;
        GGML_ASSERT(node_tt.tensor_attributes);
        GGML_ASSERT(node_tt.storage_type() == tt::tt_metal::StorageType::DEVICE);
        if (!ggml_tt_tensors_shape_equal(node, node_tt)) {
            fmt::println(
                stderr, "Mismatched tensor shapes for node '{}' ({}): GGML wants [{}, {}, {}, {}], TTNN generates {}\n",
                node->name, ggml_op_name(node->op), node->ne[0], node->ne[1], node->ne[2], node->ne[3],
                node_tt.logical_shape());
            abort();
        }
    }

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

static bool ggml_backend_metalium_device_supports_op_internal(ggml_backend_dev_t device, const struct ggml_tensor * op);

static bool ggml_backend_metalium_device_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    bool ok = ggml_backend_metalium_device_supports_op_internal(device, op);
    // debug print to log rejected ops
    if (!ok && g_debug_flags.print_rejected_ops) {
        fprintf(stderr, "REJECT op %s (%s)\n", ggml_op_name(op->op), op->name);
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (!op->src[i]) {
                break;
            }
            fprintf(stderr, "  src%d shape [%ld %ld %ld %ld], dtype = %s, name = '%s'\n", i, op->src[i]->ne[0],
                    op->src[i]->ne[1], op->src[i]->ne[2], op->src[i]->ne[3], ggml_type_name(op->src[i]->type),
                    op->src[i]->name);
        }

        // Follow op details
        if (op->op == GGML_OP_FLASH_ATTN_EXT) {
            fprintf(stderr, "  FlashAttention debug details:\n");
            const char * names[] = { "query", "key", "value", "mask" };
            for (int i = 0; i < 4; i++) {
                if (!op->src[i]) {
                    break;
                }
                ggml_tensor * t = op->src[i];
                while (t->op == GGML_OP_PERMUTE) {
                    t = t->src[0];
                }
                fprintf(stderr, "    src%d follow - %s shape [%ld %ld %ld %ld], dtype = %s, name = '%s'\n", i, names[i],
                        t->ne[0], t->ne[1], t->ne[2], t->ne[3], ggml_type_name(t->type), t->name);
            }
        }
    }
    return ok;
}

static bool ggml_backend_metalium_device_supports_op_internal(ggml_backend_dev_t         device,
                                                              const struct ggml_tensor * op) {
    GGML_ASSERT(op != NULL);
    const struct ggml_tensor *             src0 = op->src[0];
    const struct ggml_tensor *             src1 = op->src[1];
    ggml_backend_metalium_device_context * ctx  = (ggml_backend_metalium_device_context *) device->context;

    // The metalium backend has seperated internal data types from the GGML data types. We really only care about
    // what we can convert to and from.
    auto tensor_supported = [&](const struct ggml_tensor * tensor) {
        if (tensor == NULL || !is_ggml_type_supported_by_metalium(tensor->type, ctx->device->arch())) {
            return false;
        }

        tt::tt_metal::DataType tt_type = ggml2tt_type(tensor->type, ctx->device->arch());
        switch (tt_type) {
            case tt::tt_metal::DataType::BFLOAT16:
            case tt::tt_metal::DataType::UINT16:
            case tt::tt_metal::DataType::FLOAT32:
            case tt::tt_metal::DataType::UINT32:
            case tt::tt_metal::DataType::INT32:
            case tt::tt_metal::DataType::BFLOAT8_B:
            case tt::tt_metal::DataType::BFLOAT4_B:
                return true;
            case tt::tt_metal::DataType::INVALID:
                GGML_ASSERT(false && "Unsupported data type");
                break;
            default:
                return false;
        }
        GGML_UNREACHABLE();
    };

    if (!tensor_supported(op)) {
        return false;
    }
    // ARANGE and NONE are special case where src0 is not required
    if (op->op == GGML_OP_NONE || op->op == GGML_OP_ARANGE) {
        return true;
    }
    if (!tensor_supported(src0)) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_EXP:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_NORM:
            return ggml_backend_metalium_can_norm(op, false);
        case GGML_OP_RMS_NORM:
            return ggml_backend_metalium_can_norm(op, true);
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CLAMP:
        case GGML_OP_SCALE:
        case GGML_OP_ADD1:
        case GGML_OP_SQRT:
        case GGML_OP_SQR:
        case GGML_OP_PERMUTE:
        case GGML_OP_LOG:
        case GGML_OP_VIEW:
        case GGML_OP_SUM:
            return true;
        case GGML_OP_GROUP_NORM:
            return false;  // Disabled because the operator seems to be broken
        case GGML_OP_SUM_ROWS:
            return ggml_backend_metalium_can_sum_rows(op);

        case GGML_OP_CONT:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
            return ggml_backend_metalium_can_cpy(op);

        case GGML_OP_SIN:
        case GGML_OP_COS:
            return true;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return tensor_supported(src1) && numpy_broadcast_rule(src0, src1);

        case GGML_OP_MUL_MAT:
            return tensor_supported(src1) && ggml_backend_metalium_can_mul_mat(op);
        case GGML_OP_SET:
            return tensor_supported(src1) && ggml_backend_metalium_can_set(op);
        case GGML_OP_SOFT_MAX:
            return ggml_backend_metalium_can_softmax(op);
        case GGML_OP_GET_ROWS:
            return tensor_supported(src1) && ggml_backend_metalium_can_get_rows(op);
        case GGML_OP_CONCAT:
            return tensor_supported(src1) && ggml_backend_metalium_can_concat(op);
        case GGML_OP_REPEAT:
            return ggml_backend_metalium_can_repeat(op);
        case GGML_OP_OUT_PROD:
            return tensor_supported(src1) && ggml_backend_metalium_can_outer_product(op);
        case GGML_OP_GLU:
            return ((src1 && tensor_supported(src1)) || !src1) && ggml_backend_metalium_can_glu(op);
        case GGML_OP_ROPE:
            return tensor_supported(src1) && ggml_backend_metalium_can_rope(op);
        case GGML_OP_FLASH_ATTN_EXT:
            return tensor_supported(src1) && tensor_supported(op->src[2]) && ggml_backend_metalium_can_flash_attn(op);
        case GGML_OP_SET_ROWS:
            return tensor_supported(src1) && ggml_backend_metalium_can_set_rows(op);
        default:
            return false;
    }
}

static bool ggml_backend_metalium_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_metalium_buffer_type_name) {
        return false;
    }
    ggml_backend_metalium_buffer_type_context * buft_ctx = (ggml_backend_metalium_buffer_type_context *) buft->context;
    ggml_backend_metalium_device_context *      ctx      = (ggml_backend_metalium_device_context *) dev->context;
    return buft_ctx->device == ctx->device;
}

static void ggml_backend_metalium_synchronize(ggml_backend_t backend) {
    return;
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *) backend->context;
    tt::tt_metal::distributed::Finish(ctx->device->get_mesh_device()->mesh_command_queue());
}

static struct ggml_backend_i metalium_backend_i = {
    /* .get_name                = */ ggml_backend_metalium_name,
    /* .free                    = */ ggml_backend_metalium_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_metalium_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_metalium_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_metalium_guid(void) {
    static ggml_guid guid = { 0x91, 0x69, 0xd5, 0x5f, 0x24, 0xe7, 0x44, 0x00,
                              0xb4, 0x2a, 0x73, 0x23, 0x48, 0xb0, 0x4e, 0xe7 };
    return &guid;
}

static ggml_backend_t ggml_backend_metalium_init(ggml_backend_metalium_device_context * dev_ctx) {
    int                               device_id = dev_ctx->device_id;
    std::shared_ptr<ttnn::MeshDevice> device    = dev_ctx->device;
    GGML_ASSERT(device_id >= 0 && (size_t) device_id < tt::tt_metal::GetNumAvailableDevices());
    GGML_ASSERT(device != nullptr);

    ggml_backend_metalium_context * ctx = new ggml_backend_metalium_context{
        /* device            = */ device,
        /* device_id         = */ device_id,
        /* name              = */ dev_ctx->name,
    };

    ggml_backend_t backend =
        new ggml_backend{ /* .guid      = */ ggml_backend_metalium_guid(),
                          /* .interface = */ metalium_backend_i,
                          /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_metalium_reg(), device_id),
                          /* .context   = */ ctx };
    return backend;
}

bool ggml_backend_is_metalium(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_metalium_guid());
}

static const char * ggml_backend_metaliium_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "Metalium";
}

static size_t ggml_backend_metalium_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_metalium_reg_context * ctx = (ggml_backend_metalium_reg_context *) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_metalium_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(index);
    ggml_backend_metalium_reg_context * ctx = (ggml_backend_metalium_reg_context *) reg->context;
    return ctx->devices[0];
}

static const ggml_backend_reg_i ggml_backend_metalium_reg_interface = {
    /* .get_name          = */ ggml_backend_metaliium_reg_get_name,
    /* .get_device_count  = */ ggml_backend_metalium_reg_get_device_count,
    /* .get_device        = */ ggml_backend_metalium_reg_get_device,
    /* .get_proc_address  = */ NULL,
};

static const char * ggml_backend_metalium_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_metalium_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_metalium_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    ggml_backend_metalium_device_context * ctx               = (ggml_backend_metalium_device_context *) dev->context;
    size_t                                 num_dram_channels = ctx->device->num_dram_channels();
    size_t                                 num_devices       = ctx->device->num_devices();
    auto stats = ctx->device->allocator()->get_statistics(tt::tt_metal::BufferType::DRAM);

    *total = stats.total_allocatable_size_bytes * num_dram_channels * num_devices;
    *free  = stats.total_free_bytes * num_dram_channels * num_devices;
}

static enum ggml_backend_dev_type ggml_backend_metalium_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static ggml_backend_t ggml_backend_metalium_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_metalium_device_context * ctx     = (ggml_backend_metalium_device_context *) dev->context;
    ggml_backend_t                         backend = ggml_backend_metalium_init(ctx);
    GGML_ASSERT(backend != NULL);
    return backend;
}

static void ggml_backend_metalium_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_metalium_device_context * ctx   = (ggml_backend_metalium_device_context *) dev->context;
    size_t                                 free  = 0;
    size_t                                 total = 0;
    ggml_backend_metalium_get_memory(dev, &free, &total);
    *props = ggml_backend_dev_props{
        .name         = ctx->name.c_str(),
        .description  = ctx->description.c_str(),
        .memory_free  = free,
        .memory_total = total,
        .type         = ggml_backend_metalium_get_type(dev),
        .device_id    = NULL, // TODO: Set this to a proper ID
        .caps =
            ggml_backend_dev_caps{
                                  .async                = true,
                                  .host_buffer          = false,
                                  .buffer_from_host_ptr = false,
                                  .events               = false,
                                  }
    };
}

static ggml_backend_buffer_type_t ggml_backend_metalium_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *) dev->context;
    return ggml_backend_metalium_buffer_type(dev, ctx);
}

static const ggml_backend_device_i ggml_backend_metalium_device_interface = {
    /* .get_name                = */ ggml_backend_metalium_device_get_name,
    /* .get_description         = */ ggml_backend_metalium_device_get_description,
    /* .get_memory              = */ ggml_backend_metalium_get_memory,
    /* .get_type                = */ ggml_backend_metalium_get_type,
    /* .get_props               = */ ggml_backend_metalium_device_get_props,
    /* .init_backend            = */ ggml_backend_metalium_device_init,
    /* .get_buffer_type         = */ ggml_backend_metalium_get_buffer_type,
    /* .get_host_buffer_type    = */ NULL,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_metalium_device_supports_op,
    /* .supports_buft           = */ ggml_backend_metalium_device_supports_buft,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static std::vector<std::unique_ptr<ggml_backend_device>>                  g_backend_device_holder;
static std::vector<std::unique_ptr<ggml_backend_metalium_device_context>> g_backend_device_context_holder;

}  // namespace ggml_backend_metalium

using namespace ggml_backend_metalium;

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metalium_reg() {
    static ggml_backend_reg reg;
    static std::once_flag   once;
    std::call_once(once, [&]() {
        metalium_register_all_kernel();
        // TODO: TTNN though not have peoper system packaging yet. Does support working in installed for (via Python packages rn)
        // Remove this limitation
        if (getenv("TT_METAL_RUNTIME_ROOT") == NULL) {
            fmt::println(stderr,
                         "The TT_METAL_RUNTIME_ROOT environment variables must be set to use the Metalium backend");
            abort();
        } else {
            fmt::println("Disabling persistent kernel cache. Things will be slower");
        }
        // TODO: Support multiple devices (TT supports mesh configuration so it's going to be tricky)
        // but for now we just work on 1 device at a time
        static std::unique_ptr<ggml_backend_metalium_reg_context> ctx =
            std::make_unique<ggml_backend_metalium_reg_context>();
        const size_t num_devices = 1;
        int          device_id   = 0;

        const char * device_id_env =
            getenv("GGML_METALIUM_DEVICE_ID");  // example GGML_METALIUM_DEVICE_ID=0 - use device 0
        // Mesh shape can be one or two dimensional, delimited by 'x', e.g. "2x4" or "4"
        const char *    mesh_env = getenv("GGML_METALIUM_MESH_SHAPE");
        ttnn::MeshShape mesh_shape(1, 1);
        if (device_id_env != NULL && mesh_env != NULL) {
            GGML_ABORT(
                "Both GGML_METALIUM_DEVICE_ID and GGML_METALIUM_MESH_SHAPE are set. Only one can be used at the same "
                "time");
        }
        if (device_id_env != NULL) {
            try {
                device_id = std::stoi(device_id_env);
            } catch (const std::invalid_argument & e) {
                GGML_ABORT("Invalid device ID in GGML_METALIUM_DEVICE_ID");
            }
        }
        if (mesh_env != NULL) {
            std::string mesh_env_str(mesh_env);
            std::regex  pattern(R"(^(\d+)x(\d+)$)");
            std::smatch matches;

            if (std::regex_match(mesh_env_str, matches, pattern)) {
                int x = std::stoi(matches[1].str());
                GGML_ASSERT(x > 0 && "Mesh shape dimension x must be positive");
                int y = std::stoi(matches[2].str());
                GGML_ASSERT(y > 0 && "Mesh shape dimension y must be positive");
                mesh_shape = ttnn::MeshShape(x, y);
            } else {
                GGML_ABORT("Invalid mesh shape in GGML_METALIUM_MESH_SHAPE. Expected format WxH. ex: 1x4");
            }
        }

        // Fabric config needs to be set before creating devices.
        const char * fabric_config = getenv("GGML_METALIUM_FABRIC_CONFIG");
        if (fabric_config != NULL && mesh_shape.mesh_size() > 1) {
            std::string_view            fabric_config_view(fabric_config);
            tt::tt_fabric::FabricConfig config;
            if (fabric_config_view == "FABRIC_1D_NEIGHBOR_EXCHANGE"sv) {
                config = tt::tt_fabric::FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE;
            } else if (fabric_config_view == "FABRIC_1D"sv) {
                config = tt::tt_fabric::FabricConfig::FABRIC_1D;
            } else if (fabric_config_view == "FABRIC_1D_RING"sv) {
                config = tt::tt_fabric::FabricConfig::FABRIC_1D_RING;
            } else if (fabric_config_view == "FABRIC_2D"sv) {
                config = tt::tt_fabric::FabricConfig::FABRIC_2D;
            } else if (fabric_config_view == "FABRIC_2D_TORUS_X"sv) {
                config = tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X;
            } else if (fabric_config_view == "FABRIC_2D_TORUS_Y"sv) {
                config = tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y;
            } else if (fabric_config_view == "FABRIC_2D_TORUS_XY"sv) {
                config = tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY;
            } else {
                GGML_ABORT("Invalid fabric config in GGML_METALIUM_FABRIC_CONFIG.");
            }
            tt::tt_fabric::SetFabricConfig(config);
        }

        ctx->devices.reserve(num_devices);
        ggml_backend_metalium_device_context * dev_ctx = new ggml_backend_metalium_device_context;
        std::shared_ptr<ttnn::MeshDevice>      device;
        if (mesh_env == NULL) {
            device = ttnn::open_mesh_device(device_id);
        } else {
            device = ttnn::distributed::open_mesh_device(mesh_shape, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE,
                                                         1, tt::tt_metal::DispatchCoreConfig{});
        }
        if (!g_debug_flags.disable_program_cache) {
            ttnn::enable_program_cache(*device);
        }
        // Limit device support to the ones I own (GS is removed as TTNN dropped support)
        GGML_ASSERT(device->arch() == tt::ARCH::WORMHOLE_B0 || device->arch() == tt::ARCH::BLACKHOLE);
        dev_ctx->device    = device;
        dev_ctx->device_id = device_id;
        dev_ctx->name      = "METALIUM" + std::to_string(device_id);
        // WHY???
        // chip_id_t MeshDevice::build_id() const { return reference_device()->id(); }
        // Reference device should be the same... Dafaq?
        tt::ChipId id      = device->get_devices().size() == 1 ? device->get_devices()[0]->id() : device->id();
        GGML_ASSERT(id == device_id && "WTF? Metalium ID should match with asked device ID");
        std::string arch_str = tt::arch_to_str(device->arch());
        std::transform(arch_str.begin(), arch_str.end(), arch_str.begin(), ::toupper);
        if (device->get_devices().size() == 1) {
            auto * real_device   = device->get_devices()[0];
            auto   grid          = real_device->compute_with_storage_grid_size();
            dev_ctx->description = fmt::format("Tenstorrent {} [grid: {}x{}, id: {}]", arch_str, grid.x, grid.y, id);
        } else {
            auto        devshape = device->get_view().shape();
            std::string devshape_str;
            for (size_t i = 0; i < devshape.dims(); i++) {
                devshape_str += std::to_string(devshape[i]) + "x";
            }
            devshape_str.pop_back();
            dev_ctx->description = fmt::format("Tenstorrent {} {} mesh [id: {}]", arch_str, devshape_str, id);
        }

        ggml_backend_dev_t dev =
            new ggml_backend_device{ .iface = ggml_backend_metalium_device_interface, .reg = &reg, .context = dev_ctx };
        ctx->devices.push_back(dev);
        // GGML does not have free for backend_reg and devices. Will force free on exit (thanks to RAII) but Metalium
        // already de-init at that point
        // g_backend_device_context_holder.push_back(std::unique_ptr<ggml_backend_metalium_device_context>(dev_ctx));
        // g_backend_device_holder.push_back(std::unique_ptr<ggml_backend_device>(dev));

        reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_VERSION,
                                /* .interface   = */ ggml_backend_metalium_reg_interface,
                                /* .context     = */ ctx.get() };
    });
    return &reg;
}
