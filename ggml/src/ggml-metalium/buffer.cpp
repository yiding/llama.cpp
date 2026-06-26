#include "buffer.hpp"

#include "ggml.h"
#include "utils.hpp"

#include <optional>
#include <ttnn/distributed/distributed_tensor.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/types.hpp>

namespace {

template <typename SrcType, typename DstType>
tt::tt_metal::HostBuffer host_data_to_tt_host_buffer(const SrcType * src, size_t size) {
    // Converts GGML floating point (FP32, FP16, BF16) to TT floating point (FP32, BF16)
    using Src = std::remove_cv_t<std::remove_reference_t<SrcType>>;
    using Dst = std::remove_cv_t<std::remove_reference_t<DstType>>;
    // Convert from  GGML types to TT types
    static_assert(std::is_same_v<Src, float> || std::is_same_v<Src, ggml_bf16_t> || std::is_same_v<Src, ggml_fp16_t> ||
                  std::is_same_v<Src, int>);
    static_assert(std::is_same_v<Dst, float> || std::is_same_v<Dst, bfloat16> || std::is_same_v<Dst, uint32_t>);

    auto src_adaptor = [](const SrcType & src) -> float {
        if constexpr (std::is_same_v<Src, ggml_fp16_t>) {
            return ggml_fp16_to_fp32(src);
        } else if constexpr (std::is_same_v<Src, ggml_bf16_t>) {
            return ggml_bf16_to_fp32(src);
        } else if constexpr (std::is_same_v<Src, float>) {
            return src;
        } else if constexpr (std::is_same_v<Src, int>) {
            return static_cast<float>(src);
        }
        GGML_UNREACHABLE();
    };

    auto dst_adaptor = [](DstType & dst, float val) {
        if constexpr (std::is_same_v<Dst, bfloat16>) {
            dst = bfloat16(val);
        } else if constexpr (std::is_same_v<Dst, float>) {
            dst = val;
        } else if constexpr (std::is_same_v<Dst, int>) {
            dst = static_cast<int>(val);
        } else if constexpr (std::is_same_v<Dst, uint32_t>) {
            dst = static_cast<uint32_t>(val);
        } else {
            GGML_UNREACHABLE();
        }
    };

    // Optimization: avoid unnecessary initialization and copying like vec<float>(size) as it tanks performance
    Dst * vec = new Dst[size];
    // special case if both GGML and TT types have the same underlying type (e.g. both FP32 or BF16)
    if constexpr (std::is_same_v<Src, Dst> || (std::is_same_v<Src, ggml_bf16_t> && std::is_same_v<Dst, bfloat16>) ||
                  (std::is_same_v<Src, int> &&
                   std::is_same_v<Dst, uint32_t>) ) {  // we really don't care about signedness here
        static_assert(sizeof(Src) == sizeof(Dst), "Src and Dst must have the same size");
        // Make GCC shut up about writing into a class like it's flat memory
        memcpy((void *) vec, src, size * sizeof(Src));
    }
    // special case for BFP16 (much faster then TTNN's implementation)
    else if constexpr (std::is_same_v<Src, float> && std::is_same_v<Dst, bfloat16>) {
        const auto * trait = ggml_get_type_traits_cpu(GGML_TYPE_BF16);
        assert(trait != nullptr);
        trait->from_float(src, vec, size);
    } else {
        for (size_t i = 0; i < size; i++) {
            dst_adaptor(vec[i], src_adaptor(src[i]));
        }
    }

    int *                   refcount = new int(0);
    tt::tt_metal::MemoryPin pin([refcount]() mutable { (*refcount)++; },
                                [refcount, vec]() mutable {
                                    assert(refcount != nullptr);
                                    (*refcount)--;
                                    if (*refcount == 0) {
                                        delete refcount;
                                        delete[] vec;
                                        refcount = nullptr;
                                    }
                                });
    auto                    storage = tt::tt_metal::HostBuffer(ttsl::Span<DstType>(vec, size), std::move(pin));

    return storage;
}

template <typename DstType>
tt::tt_metal::HostBuffer quantized_ggml_data_to_tt_host_buffer(const void * src, const ggml_tensor * tensor) {
    const ggml_type_traits * trait = ggml_get_type_traits(tensor->type);
    const size_t             size  = ggml_nelements(tensor);
    GGML_ASSERT(trait->to_float != NULL);

    if constexpr (std::is_same_v<DstType, float>) {
        std::shared_ptr<float[]> vec(new float[size]);
        trait->to_float(src, vec.get(), size);
        int *                   refcount = new int(0);
        float *                 vec_ptr  = vec.get();
        tt::tt_metal::MemoryPin pin([refcount]() mutable { (*refcount)++; },
                                    [refcount, vec = std::move(vec)]() mutable {
                                        assert(refcount != nullptr);
                                        (*refcount)--;
                                        if (*refcount == 0) {
                                            delete refcount;
                                            vec.reset();
                                        }
                                    });
        return tt::tt_metal::HostBuffer(ttsl::Span<float>(vec_ptr, size), std::move(pin));
    } else if constexpr (std::is_same_v<DstType, bfloat16>) {
        std::shared_ptr<bfloat16[]> vec(new bfloat16[size]);
        size_t                      block_size_in_bytes    = ggml_type_size(tensor->type);
        size_t                      block_size_in_elements = ggml_blck_size(tensor->type);
        std::vector<float>          tmp(block_size_in_elements);

        const auto * bfp16trait = ggml_get_type_traits(GGML_TYPE_BF16);

        size_t       idx  = 0;
        const auto * data = (const std::byte *) src;
        for (size_t i = 0; i < ggml_nbytes(tensor); i += block_size_in_bytes) {
            trait->to_float(data + i, tmp.data(), block_size_in_elements);
            bfp16trait->from_float_ref(tmp.data(), vec.get() + idx, block_size_in_elements);
            idx += block_size_in_elements;
        }

        int *                   refcount = new int(0);
        bfloat16 *              vec_ptr  = vec.get();
        tt::tt_metal::MemoryPin pin([refcount]() mutable { (*refcount)++; },
                                    [refcount, vec = std::move(vec)]() mutable {
                                        assert(refcount != nullptr);
                                        (*refcount)--;
                                        if (*refcount == 0) {
                                            delete refcount;
                                            vec.reset();
                                        }
                                    });
        return tt::tt_metal::HostBuffer(ttsl::Span<bfloat16>(vec_ptr, size), std::move(pin));
    } else {
        std::shared_ptr<float[]> vec(new float[size]);
        trait->to_float(src, vec.get(), size);
        return host_data_to_tt_host_buffer<float, DstType>(vec.get(), size);
    }
}

// Copies the content of the TT tensor into memory pointed by `dst` with data of type `dst_ggtype`
// This function will do it's best to convert whatever it is in the TT tensor into types accaptable
// by GGML
// This function works by deciding if the tensor is already in the desired format, and if not
// convert to FP32 then convert into the desired format
template <typename SrcType>
void copy_tt_tensor_to_host_pointer(const tt::tt_metal::Tensor & tensor, void * dst, ggml_type dst_ggtype) {
    ttnn::Shape shape        = tensor.logical_shape();
    ttnn::Shape padded_shape = tensor.padded_shape();

    // we only support reading from these types that is held in TT tensor
    static_assert(std::is_same_v<SrcType, float> || std::is_same_v<SrcType, bfloat16> ||
                  std::is_same_v<SrcType, uint32_t>);

    tt::tt_metal::Tensor row_major_tensor = tensor;
    if (tensor.layout() == ttnn::TILE_LAYOUT) {
        // FIXME: untilize is cursed. Causes _MANY_ corruption errors. Replacing it with to_layout
        // Fixes the majority of accuracy and corruption errors in test-backend-ops
        // Ofc this is slower so we really want to enable untilize on device
        // row_major_tensor = ttnn::untilize(tensor).cpu();
        row_major_tensor = tensor.cpu().to_layout(ttnn::ROW_MAJOR_LAYOUT);
    } else {
        row_major_tensor = tensor.cpu();
    }
    GGML_ASSERT(row_major_tensor.storage_type() == tt::tt_metal::StorageType::HOST);
    GGML_ASSERT(row_major_tensor.layout() == ttnn::ROW_MAJOR_LAYOUT);

    // Grab the data held in the TT tensor
    const tt::tt_metal::HostStorage & storage  = row_major_tensor.host_storage();
    const auto                        buffer   = storage.buffer().get_shard({ 0, 0 }).value();
    auto                              view     = buffer.view_as<SrcType>();
    const SrcType *                   buf      = &view[0];
    size_t                            buf_size = view.size();
    GGML_ASSERT(buf != nullptr);

    // Determine our conversion strategy
    void * intermid = nullptr;  // pointer to a buffer that can hold the intermediate data (if needed)
    bool   need_quantized_conversion =
        false;  // flag indicating whether we need to qunatize the value extracted from TT later for GGML use
    bool src_dst_same = false;            // If TT and GGML both have the same type - we can just memcpy

    std::vector<std::byte> intermid_buf;  // In case we need it, some place to put data

    // If both side is FP32
    if (dst_ggtype == GGML_TYPE_F32 && !std::is_same_v<SrcType, float>) {
        intermid                  = dst;
        need_quantized_conversion = false;
        src_dst_same              = false;
    }
    // If both side are the same type fundimentally
    // NOTE: Just putting the integer types here to remind me TT tensors can have integer types
    else if ((std::is_same_v<SrcType, float> && dst_ggtype == GGML_TYPE_F32) ||
             (std::is_same_v<SrcType, bfloat16> && dst_ggtype == GGML_TYPE_BF16) ||
             (std::is_same_v<SrcType, int32_t> && dst_ggtype == GGML_TYPE_I32) ||
             (std::is_same_v<SrcType, uint32_t> && dst_ggtype == GGML_TYPE_I32) ||
             (std::is_same_v<SrcType, int16_t> && dst_ggtype == GGML_TYPE_I16) ||
             (std::is_same_v<SrcType, int8_t> && dst_ggtype == GGML_TYPE_I8)) {
        intermid                  = dst;
        need_quantized_conversion = false;
        src_dst_same              = true;
    }
    // If both side are different - allocate the intermediate buffer and we need to convert
    else {
        intermid_buf.resize(shape.volume() * sizeof(float));
        intermid                  = intermid_buf.data();
        need_quantized_conversion = true;
        src_dst_same              = false;
    }

    auto src_adaptor = [](const SrcType & src) -> float {
        if constexpr (std::is_same_v<SrcType, bfloat16>) {
            return static_cast<float>(src);
        }
        if constexpr (std::is_same_v<SrcType, float>) {
            return src;
        }
        if constexpr (std::is_same_v<SrcType, uint32_t>) {
            return src;
        }
        GGML_UNREACHABLE();
    };

    // Tilize to ROW_MAJOR doesn't mean the tensor is contiguous. It produces tensors that has 0 padded up to the nearest
    // 32 elements on last two (for GGML first two) dimentions.
    // Compute the stride for each dimension
    std::array<size_t, 4> stride            = { 1, 1, 1, 1 };
    size_t                cumulative_stride = 1;
    for (int i = padded_shape.size() - 1; i >= 0; i--) {
        stride[i] = cumulative_stride;
        cumulative_stride *= padded_shape[i];
    }

    // Convert TT shape to GGML shape
    std::array<size_t, 4> nshape{ 1, 1, 1, 1 };
    for (size_t i = 0; i < shape.size(); i++) {
        nshape[4 - shape.size() + i] = shape[i];
    }

    static_assert(GGML_MAX_DIMS == 4, "Looping depth is hardcoded to 4");
    // Sanity check: src_dst_same shuld indicate there is no need for quantized conversion
    GGML_ASSERT(((src_dst_same && !need_quantized_conversion) || !src_dst_same) &&
                "src and dst should be the same type if src_dst_same is true");
    // NOTE: The following optimizations are not full and has some slow paths taken unoptimally. But good enough for now

    // Optimization: large block copy
    // If  row major in TT is continous - memcpy it directly or (since we are converting from float) abuse the pointer
    if (nshape[3] % 32 == 0 && ((nshape[0] == 1 && nshape[1] == 1) || nshape[2] % 32 == 0)) {
        const size_t buf_size = std::accumulate(nshape.begin(), nshape.end(), 1, std::multiplies<size_t>());
        // Both sides are same type - memcpy and call it a day
        if (src_dst_same && !need_quantized_conversion) {
            memcpy(dst, buf, sizeof(SrcType) * buf_size);
            return;
        }
        // need conversion but TT side is already FP32 - pointer abuse
        if (std::is_same_v<SrcType, float> && need_quantized_conversion) {
            intermid = const_cast<void *>(static_cast<const void *>(buf));
        }
        // else we manually convert
        else {
            for (size_t i = 0; i < buf_size; i++) {
                ((float *) intermid)[i] = src_adaptor(buf[i]);
            }
        }
    }
    // If the 2nd dimension is not divisible by 32, we can still copy block by block
    else if (nshape[0] % 32 == 0 && nshape[1] % 32 != 0) {
        const size_t src_block_size   = nshape[2] * nshape[3];
        const size_t src_block_stride = stride[1];
        if (src_dst_same) {
            for (size_t i = 0; i < nshape[0] * nshape[1]; i++) {
                memcpy((SrcType *) intermid + i * src_block_size, buf + i * src_block_stride,
                       sizeof(SrcType) * src_block_size);
            }
        } else {
            for (size_t i = 0; i < nshape[0] * nshape[1]; i++) {
                for (size_t j = 0; j < src_block_size; j++) {
                    ((SrcType *) intermid)[i * src_block_size + j] = src_adaptor(buf[i * src_block_stride + j]);
                }
            }
        }
    }
    // row-by-row copy
    // Only avoid small copies via memcpy if not copying into FP32 - we rely on raw copies for other types as the
    // fallback loop asserts FP32
    else if (src_dst_same && !need_quantized_conversion && (shape[3] >= 4 || !std::is_same_v<SrcType, float>) ) {
        const size_t dst_stride = nshape[3];
        for (size_t i = 0; i < nshape[0] * nshape[1]; i++) {
            for (size_t j = 0; j < nshape[2]; j++) {
                // optimization: copy a row of memory at a time
                const size_t src_idx = i * stride[1] + j * stride[2];
                memcpy((SrcType *) intermid + j * dst_stride + i * nshape[2] * dst_stride, buf + src_idx,
                       sizeof(SrcType) * nshape[3]);
            }
        }
    }
    // Slow path: src and dst are different types or the data is not contiguous in memory
    else {
        size_t idx = 0;
        for (size_t w = 0; w < nshape[0]; w++) {
            for (size_t z = 0; z < nshape[1]; z++) {
                for (size_t y = 0; y < nshape[2]; y++) {
                    for (size_t x = 0; x < nshape[3]; x++) {
                        const size_t src_idx = w * stride[0] + z * stride[1] + y * stride[2] + x * stride[3];
                        GGML_ASSERT(src_idx < buf_size);
                        if (!src_dst_same) {
                            ((float *) intermid)[idx] = src_adaptor(buf[src_idx]);
                        } else {
                            // memcpy((SrcType*)intermid + idx, buf + src_idx, sizeof(SrcType));
                            const SrcType * src_ptr = buf + src_idx;
                            SrcType *       dst_ptr = (SrcType *) intermid + idx;
                            *dst_ptr                = *src_ptr;
                        }
                        idx++;
                    }
                }
            }
        }
    }

    if (need_quantized_conversion) {
        GGML_ASSERT((ggml_is_quantized(dst_ggtype) || dst_ggtype == GGML_TYPE_F16 || dst_ggtype == GGML_TYPE_I32) &&
                    "This block should only reach for quantized data types or FP16");
        GGML_ASSERT(intermid_buf.size() != 0);
        const ggml_type_traits_cpu * trait = ggml_get_type_traits_cpu(dst_ggtype);
        GGML_ASSERT(trait->from_float != NULL);
        trait->from_float((float *) intermid, dst, shape.volume());
    }
}

}  // anonymous namespace

namespace ggml_backend_metalium {

static void ggml_backend_metalium_buffer_free_buffer(ggml_backend_buffer_t buffer);

static bool ggml_backend_buffer_is_metalium(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_metalium_buffer_free_buffer;
}

static void ggml_backend_metalium_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = ggml_backend_metalium_buffer_context::get(buffer);
    delete ctx;
}

static void * ggml_backend_metalium_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = ggml_backend_metalium_buffer_context::get(buffer);
    return (uint8_t *) 0xdeadbeef + ctx->base_offset;
}

static void ggml_backend_metalium_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                    ggml_tensor *         tensor,
                                                    const void *          data,
                                                    size_t                offset,
                                                    size_t                size) {
    // Here's the general logic of set_tensor
    // 1. Make a flat buffer and copy the data into it
    //    - If the data is quantized, convert it to BFLOAT16
    //    - Try to directly copy the data if it is already in the correct format
    // 2. Create a TT tensor from the flat buffer as ROW_MAJOR. Send it to the device and tile it
    // 3. If the data is quantized, cast down to BFLOAT8_B or BFLOAT4_B
    // There's a lot of things to do here.
    // TODO: Make a scalable way to decide which GGML type casts to TT quantized types
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(tensor->extra != NULL);

    auto * bufctx = ggml_backend_metalium_buffer_context::get(buffer);
    GGML_ASSERT(bufctx != NULL);
    ggml_type              ggtype    = tensor->type;
    tt::tt_metal::Tensor & tt_tensor = get_tt_tensor(tensor);

    // Make sure we are not writing to a view tensor
    if (size != ggml_nbytes(tensor) ||
        (tt_tensor.tensor_attributes && ggml_tt_tensors_shape_equal(tensor, tt_tensor) == false) ||
        tensor->view_src != NULL) {
        // FIXME: Reenable this when got time
        // fprintf(stderr, "Warning: Metalium set_tensor() does not work with tensor views\n");
        return;
    }

    std::optional<tt::tt_metal::HostBuffer> storage;
    tt::tt_metal::DataType                  intermidiate_type = tt::tt_metal::DataType::BFLOAT16;
    bool                                    tilize            = true;
    if (ggtype == GGML_TYPE_F32) {
        // For now we cast F32 to BF16. Need a scalable way to handle this as WORMHOLD_B0 have native support for F32
        // TODO: Enable proper FP32 when all related bugs gets fixed for devices that support it
        storage = host_data_to_tt_host_buffer<float, bfloat16>((const float *) data, size / sizeof(float));
    } else if (ggtype == GGML_TYPE_F16) {
        // TT hardware claims to support FP16 but the API does not expose it. For now we use BF16 as it is close enough
        storage =
            host_data_to_tt_host_buffer<ggml_fp16_t, bfloat16>((const ggml_fp16_t *) data, size / sizeof(ggml_fp16_t));
    } else if (ggtype == GGML_TYPE_BF16) {
        storage =
            host_data_to_tt_host_buffer<ggml_bf16_t, bfloat16>((const ggml_bf16_t *) data, size / sizeof(ggml_bf16_t));
    } else if (ggtype == GGML_TYPE_I32) {
        storage           = host_data_to_tt_host_buffer<int, uint32_t>((const int *) data, size / sizeof(int));
        intermidiate_type = tt::tt_metal::DataType::UINT32;
        tilize            = false;  // Integer tensors are indices - operations will want them untiled
    } else if (ggml_is_quantized(ggtype)) {
        // Even though in theory transfering BFP16 to device uses much less bandwidth then FP32. GGML nativly have support
        // converting quantized types into FP32. Converting to BFP16 would be an extra step making everything slower
        storage           = quantized_ggml_data_to_tt_host_buffer<float>(data, tensor);
        intermidiate_type = tt::tt_metal::DataType::FLOAT32;
    } else {
        fmt::println(stderr, "Unsupported data type while uploading to device: {}, name '{}', op type: {}\n",
                     ggml_type_name(ggtype), tensor->name, ggml_op_name(tensor->op));
        GGML_ASSERT(false && "Unsupported data type while uploading to device");
    }
    GGML_ASSERT(storage.has_value() && "Failed to convert data to TT storage");

    auto shape = tt_shape_of(tensor);

    std::optional<ttsl::SmallVector<int64_t>> permute;
    // In case GGML sent us a non-contiguous tensor, we need to permute it to make it contiguous
    // We don't care about reshape as that doesn't make a difference in row-major layout
    // TODO: This code does not handle yucky cases like stries of [4, 8, 0, 0] but I assume GGML
    // is decent enough to not send us such tensors
    if (!ggml_is_contiguous(tensor)) {
        // Look at ne (aka strides) and figure out the real underlying shape
        std::array<std::pair<uint64_t, int>, GGML_MAX_DIMS> strides;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            strides[i] = { tensor->nb[i], i };
        }
        std::sort(strides.begin(), strides.end(), [](const auto & a, const auto & b) { return a.first < b.first; });

        std::array<std::pair<uint64_t, int>, GGML_MAX_DIMS> s;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            s[i] = { tensor->ne[i], strides[i].second };
        }
        std::sort(s.begin(), s.end(), [](const auto & a, const auto & b) { return a.second < b.second; });
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            shape[GGML_MAX_DIMS - i - 1] = s[i].first;
        }

        // Now we can figure out the permutation that we need to apply
        ttsl::SmallVector<int64_t> perm(GGML_MAX_DIMS, -1);
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            perm[strides[i].second] = i;
        }
        permute = perm;
    }

    auto * ctx = tensor_extra::from(tensor);

    tt::tt_metal::Tensor t(std::move(*storage), ttnn::Shape(shape), intermidiate_type, tt::tt_metal::Layout::ROW_MAJOR);

    tt::tt_metal::DataType final_type = ggml2tt_type(ggtype, bufctx->device->arch());
    if (tilize) {
        t = ctx->mesh_mapper.has_value() ?
                ttnn::distributed::distribute_tensor(t, ctx->mesh_mapper.value(), *bufctx->device) :
                t.to_device(bufctx->device.get());
        t = ttnn::tilize_with_zero_padding(t, std::nullopt, final_type);
        if (permute.has_value()) {
            t = ttnn::permute(t, *permute);
        }
    } else {
        GGML_ASSERT(t.dtype() == final_type && "Tensor dtype mismatch during tensor creation for row major tensors");
        GGML_ASSERT(!permute.has_value() && "Cannot permute tensor without tilizing");

        t = ctx->mesh_mapper.has_value() ?
                ttnn::distributed::distribute_tensor(t, ctx->mesh_mapper.value(), *bufctx->device) :
                t.to_device(bufctx->device.get());
    }

    if (ctx->is_pretransposed) {
        t = ttnn::transpose(std::move(t), -2, -1);
    }

    GGML_ASSERT(t.storage_type() == tt::tt_metal::StorageType::DEVICE);
    GGML_ASSERT(t.dtype() == final_type);
    GGML_ASSERT(ggml_tt_tensors_shape_equal(tensor, t));
    GGML_ASSERT(t.layout() == (tilize ? tt::tt_metal::Layout::TILE : tt::tt_metal::Layout::ROW_MAJOR));

    tt_tensor = std::move(t);
}

static void ggml_backend_metalium_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                    const ggml_tensor *   tensor,
                                                    void *                data,
                                                    size_t                offset,
                                                    size_t                size) {
    GGML_UNUSED(buffer);
    // Here's the general logic of get_tensor
    // 1. Get the TT tensor from the metadata
    // 2. If the TT tensor is quantized, cast it to BFLOAT16
    // 3. Call copy_tt_tensor_to_host_pointer to convert the TT tensor to GGML tensor
    //    - copy_tt_tensor_to_host_pointer internally handles the data type conversion
    GGML_ASSERT(tensor->extra != NULL);

    // ggml_backend_metalium_buffer_context * ctx = (ggml_backend_metalium_buffer_context *)buffer->context;

    ggml_type dst_ggtype = tensor->type;

    // auto *meta = (ggml_tensor_extra_metalium*)tensor->extra;
    // auto shape = meta->tensor->logical_shape();
    // std::cout << "get_tensor():\n";
    // std::cout << "  GGML thinks shape: " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << std::endl;
    // std::cout << "  TTNN thinks shape: " << shape << std::endl;
    tt::tt_metal::Tensor t;
    if (tensor->op == GGML_OP_TRANSPOSE) {
        // std::cout << "Reading out to transpose tensor" << std::endl;
        // HACK: Yeah this one is stupid. GGML as a row-major framework uses lazy evaluation for transpose.
        //      Which means if we try to copy a transposed tensor. We should not transpose it. Else the other
        //      backend would transpose it again.
        ggml_tensor * src          = tensor->src[0];
        bool          do_transpose = false;
        while (src->op == GGML_OP_TRANSPOSE) {
            do_transpose = !do_transpose;
            src          = src->src[0];
            GGML_ASSERT(src != NULL);
        }
        GGML_ASSERT(src != NULL);
        t = realize_ggml_view(src);
        if (do_transpose) {
            t = ttnn::transpose(t, -2, -1);
        }
    } else if (tensor->op == GGML_OP_PERMUTE) {
        // DITTO above.
        // XXX: This only handles the case where the permute is the only view class operation
        // May broke if there are multiple permutes
        ggml_tensor * src = tensor->src[0];
        t                 = realize_ggml_view(src);
    } else if (tensor->op == GGML_OP_RESHAPE) {
        // No reason to do actual reshaping as it doesn't make a difference in row-major layout
        ggml_tensor * src = tensor->src[0];
        while (src->op == GGML_OP_RESHAPE) {
            src = src->src[0];
            GGML_ASSERT(src != NULL);
        }
        GGML_ASSERT(src != NULL);
        t = realize_ggml_view(src);
    } else {
        t = realize_ggml_view(tensor);
        GGML_ASSERT(ggml_tt_tensors_shape_equal(tensor, t));
    }

    if (tensor_extra::from(tensor)->is_pretransposed) {
        t = ttnn::transpose(t, -2, -1);
    }

    // Support some sub-tensor fetches that we can infer from the size / offset.
    if (size == ggml_nbytes(tensor) && offset == 0) {  // Fetch everything.
    } else if (ggml_n_dims(tensor) == 2 && !ggml_is_quantized(tensor->type) &&
               // starting from the start of a row
               offset % tensor->nb[1] == 0 &&
               // fetching entire rows
               size % tensor->nb[1] == 0) {
        // Fetching whole rows of a 2D tensor, e.g. for fetching from embeddings.
        uint32_t start_row = offset / tensor->nb[1];
        uint32_t n_rows    = size / tensor->nb[1];
        t = ttnn::slice<uint32_t>(t, { 0, 0, start_row, 0 }, { 1, 1, start_row + n_rows, t.logical_shape()[3] },
                                  { 1, 1, 1, 1 });
    } else {
        GGML_ABORT("Unsupported sub-tensor get");
    }

    if (t.dtype() != tt::tt_metal::DataType::BFLOAT16 && t.dtype() != tt::tt_metal::DataType::FLOAT32 &&
        t.dtype() != tt::tt_metal::DataType::UINT32) {
        t = ttnn::typecast(t, tt::tt_metal::DataType::BFLOAT16);
    }

    // TODO: Proper handling of data types
    GGML_ASSERT(dst_ggtype != GGML_TYPE_F64 && dst_ggtype != GGML_TYPE_I16 && dst_ggtype != GGML_TYPE_I8);
    switch (t.dtype()) {
        case tt::tt_metal::DataType::BFLOAT16:
            copy_tt_tensor_to_host_pointer<bfloat16>(t, (float *) data, dst_ggtype);
            break;
        case tt::tt_metal::DataType::FLOAT32:
            copy_tt_tensor_to_host_pointer<float>(t, (float *) data, dst_ggtype);
            break;
        case tt::tt_metal::DataType::UINT32:
            copy_tt_tensor_to_host_pointer<uint32_t>(t, (int *) data, dst_ggtype);
            break;
        default:
            GGML_ABORT("Unsupported data type in TT tensor when converting to GGML tensor");
    }
}

static void ggml_backend_metalium_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    // Not using this. Metalium's allication model is not compatible with GGML's allocator
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static bool ggml_backend_metalium_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    const ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    GGML_UNUSED(buffer);

    if (!ggml_backend_buffer_is_metalium(src->buffer)) {
        return false;
    }
    GGML_ASSERT(dst->extra != nullptr);

    const tt::tt_metal::Tensor & src_tensor = get_tt_tensor(src);

    tt::tt_metal::Tensor ret = ttnn::identity(src_tensor);
    GGML_ASSERT(ret.storage_type() == tt::tt_metal::StorageType::DEVICE);
    get_tt_tensor(dst) = std::move(ret);
    return true;
}

static void ggml_backend_metalium_buffer_reset(ggml_backend_buffer_t buffer) {
    auto * bufctx = ggml_backend_metalium_buffer_context::get(buffer);
    bufctx->metadata_to_free.clear();
}

static enum ggml_status ggml_backend_metalium_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    using namespace std;
    using namespace tt::tt_metal;

    auto * bufctx = ggml_backend_metalium_buffer_context::get(buffer);

    bufctx->metadata_to_free.push_back(std::make_unique<tensor_extra>());
    tensor_extra * meta = bufctx->metadata_to_free.back().get();
    tensor->extra       = meta;

    const auto & shape = tt_shape_of(tensor);

    string_view name{ tensor->name };
    if (name.find("ffn_gate.weight") != string_view::npos || name.find("ffn_up.weight") != string_view::npos) {
        meta->is_pretransposed = true;
        // Sharding happens before possible transposing, so we use dim 2 (i.e.
        // the contracting dimension)
        if (bufctx->tp_ne().has_value()) {
            meta->tp_dim = 2;
            meta->mesh_mapper =
                std::move(*ttnn::distributed::shard_tensor_to_mesh_mapper(*bufctx->device, 2, bufctx->tp_axis()));
        }
    } else if (name.find("ffn_down.weight") != string_view::npos) {
        meta->is_pretransposed = true;
        // Sharding happens before possible transposing, so we use dim 2 (i.e.
        // the contracting dimension)
        if (bufctx->tp_ne().has_value()) {
            meta->tp_dim = 3;
            meta->mesh_mapper =
                std::move(*ttnn::distributed::shard_tensor_to_mesh_mapper(*bufctx->device, 3, bufctx->tp_axis()));
        }

    } else {
        meta->is_pretransposed = false;
    }

    bool needs_init = false;

    // As an optimization, ggml does not call tensor_set for 0-sized tensors, so
    // we initialize the tt tensor here for 0 sized tensors.
    // TODO(yiding): we should probably just allocate all the time in init, and
    // ensure operations that write to this tensor use existing allocated tensor.
    needs_init |= ggml_nbytes(tensor) == 0;

    // HACK: Make KV cache work. They don't get set before first use
    // TODO: Most likely we'd want to refer this allocation to first time use of the tensor to support proper KV cache setup
    //       as the "real" shape information (GGML allocates KV cache as a very long 1D tensor) is missing here
    needs_init |= (name.find("cache") != std::string::npos && tensor->op == GGML_OP_NONE);

    if (needs_init) {
        auto t = ttnn::zeros(tt_shape_of(tensor), ggml2tt_type(tensor->type, bufctx->device->arch()),
                             tt::tt_metal::Layout::TILE);
        if (meta->mesh_mapper.has_value()) {
            t = ttnn::distributed::distribute_tensor(t, meta->mesh_mapper.value(), *bufctx->device);
        } else {
            t = t.to_device(bufctx->device.get());
        }
        meta->tensor = std::move(t);
    }
    return GGML_STATUS_SUCCESS;
}

const tt::tt_metal::Tensor & get_tt_tensor(const ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_metalium(tensor->buffer));
    return static_cast<const tensor_extra *>(tensor->extra)->tensor;
}

tt::tt_metal::Tensor & get_tt_tensor(ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_metalium(tensor->buffer));
    return static_cast<tensor_extra *>(tensor->extra)->tensor;
}

struct ggml_backend_buffer_i ggml_backend_metalium_buffer_interface = {
    .free_buffer   = ggml_backend_metalium_buffer_free_buffer,
    .get_base      = ggml_backend_metalium_buffer_get_base,
    .init_tensor   = ggml_backend_metalium_buffer_init_tensor,
    .memset_tensor = nullptr,
    .set_tensor    = ggml_backend_metalium_buffer_set_tensor,
    .get_tensor    = ggml_backend_metalium_buffer_get_tensor,
    .set_tensor_2d = nullptr,
    .get_tensor_2d = nullptr,
    .cpy_tensor    = ggml_backend_metalium_buffer_cpy_tensor,
    .clear         = ggml_backend_metalium_buffer_clear,
    .reset         = ggml_backend_metalium_buffer_reset,
};

}  // namespace ggml_backend_metalium
