 // This file is like test-backend-ops.cpp but we expect _everything_ to be supported by the Metalium backend.
// Also tests for edge cases in Metalium. (ex: Metalium/TTNN nativly uses 32x32 matrices as it's smallest unit)

// some code stolen from test-backend-ops.cpp
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cpp.h>

#include <functional>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>

#include <gtest/gtest.h>

#if !defined (GGML_USE_METALIUM)
    #error "This file should only be compiled with Metalium backend enabled"
#endif

#include <ggml-metalium.h>


static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    const ggml_type_traits* tt = ggml_get_type_traits(t->type);
    size_t bs = ggml_blck_size(t->type);
    std::vector<float> vq(ggml_blck_size(t->type));
    bool quantized = ggml_is_quantized(t->type);

    // access elements by index to avoid gaps in views
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += bs) {
                    size_t i = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0/bs*t->nb[0];
                    if (t->type == GGML_TYPE_F16) {
                        tv.push_back(ggml_fp16_to_fp32(*(ggml_fp16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_BF16) {
                        tv.push_back(ggml_bf16_to_fp32(*(ggml_bf16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_F32) {
                        tv.push_back(*(float *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I32) {
                        tv.push_back((float)*(int32_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I16) {
                        tv.push_back((float)*(int16_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I8) {
                        tv.push_back((float)*(int8_t *) &buf[i]);
                    } else if (quantized) {
                        tt->to_float(&buf[i], vq.data(), bs);
                        tv.insert(tv.end(), vq.begin(), vq.end());
                    } else {
                        GGML_ASSERT(false);
                    }
                }
            }
        }
    }

    return tv;
}

static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static double pcc(const float * a, const float * b, size_t n) {
    // Calculate the mean of x and y values
    double a_mean = 0.0;
    double b_mean = 0.0;

    for (size_t i = 0; i < n; i++) {
        a_mean += a[i];
        b_mean += b[i];
    }

    a_mean /= n;
    b_mean /= n;

    // Calculate the covariance and standard deviation of x and y values
    float covariance = 0.0f;
    float x_stddev = 0.0f;
    float y_stddev = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float x_diff = a[i] - a_mean;
        float y_diff = b[i] - b_mean;

        covariance += x_diff * y_diff;
        x_stddev += x_diff * x_diff;
        y_stddev += y_diff * y_diff;
    }

    covariance /= n;
    x_stddev /= n;
    y_stddev /= n;

    // Calculate the correlation coefficient
    double correlation_coefficient_ = covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));
    return correlation_coefficient_;
}

static bool isinf_or_max(float f) {
    return std::isinf(f) || f == std::numeric_limits<float>::max() || f == -std::numeric_limits<float>::max();
}

static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    static std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(min, max);
    size_t size = ggml_nelements(tensor);
    std::vector<float> data(size);

    for (size_t i = 0; i < size; i++) {
        data[i] = distribution(generator);
    }

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, size * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(size % ggml_blck_size(tensor->type) == 0);
        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, size));
        std::vector<float> imatrix(tensor->ne[0], 1.0f); // dummy importance matrix
        const float * im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            // when the imatrix is optional, we want to test both quantization with and without imatrix
            // use one of the random numbers to decide
            if (data[0] > 0.5f*(min + max)) {
                im = nullptr;
            }
        }
        ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), 0, size/tensor->ne[0], tensor->ne[0], im);
        GGML_ASSERT(ggml_validate_row_data(tensor->type, dataq.data(), dataq.size()));
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else if (tensor->type == GGML_TYPE_I32) {
        std::vector<int32_t> datai32(size);
        std::uniform_int_distribution<int32_t> distribution_int32(0, 2048);
        for (size_t i = 0; i < size; i++) {
            datai32[i] = distribution_int32(generator);
        }
        ggml_backend_tensor_set(tensor, datai32.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ASSERT(false);
    }
}

static void initialize_tensors(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        init_tensor_uniform(t);
    }
}

struct test_case
{
    test_case(
      std::function<ggml_tensor* (ggml_context*)> build_graph,
      const std::function<double(const float*, const float*, size_t n)>& loss = nmse,
      float max_err = 1e-4
    ) : max_err(max_err), loss(loss), build_graph(std::move(build_graph)) {}
    float max_err;
    std::function<double(const float*, const float*, size_t n)> loss;
    std::function<ggml_tensor* (ggml_context*)> build_graph;

    static const int sentinel_size = 1024;
    std::vector<ggml_tensor *> sentinels;
    ggml_cgraph * gf = nullptr;

    void add_sentinel(ggml_context * ctx) {
        ggml_tensor * sentinel = ::ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sentinel_size);
        ggml_format_name(sentinel, "sent_%zu", sentinels.size());
        sentinels.push_back(sentinel);
    }

    void eval(ggml_backend_t backend1, ggml_backend_t backend2) {
        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context_ptr ctx_ptr{ggml_init(params)};
        ggml_context * ctx = ctx_ptr.get();

        gf = ggml_new_graph(ctx);

        // pre-graph sentinel
        add_sentinel(ctx);

        ggml_tensor * out = build_graph(ctx);

        ASSERT_NE(out->op, GGML_OP_NONE) << "operator should not be NONE. Test is buggy";

        // check if the backends support the ops
        for (ggml_backend_t backend : {backend1, backend2}) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
                if (!ggml_backend_supports_op(backend, t)) {
                    GTEST_SKIP() << "not supported by backend: " << ggml_backend_name(backend) << ". Rejected OP: " << ggml_op_desc(t);
                }
            }
        }
        // post-graph sentinel
        add_sentinel(ctx);

        // allocate
        ggml_backend_buffer_ptr buf{ggml_backend_alloc_ctx_tensors(ctx, backend1)};
        ASSERT_TRUE(buf) << "failed to allocate tensors [" << ggml_backend_name(backend1) << "]";

        // build graph
        ggml_build_forward_expand(gf, out);

        // add sentinels as graph nodes so that they are checked in the callback
        for (ggml_tensor * sentinel : sentinels) {
            ggml_graph_add_node(gf, sentinel);
        }

        // randomize tensors
        initialize_tensors(ctx);

        // compare
        struct callback_userdata {
            double max_err;
            ggml_backend_t backend1;
            ggml_backend_t backend2;
            std::function<double(const float*, const float*, size_t n)> loss;
        };

        callback_userdata ud {
            max_err,
            backend1,
            backend2,
            loss
        };

        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            callback_userdata * ud = (callback_userdata *) user_data;
            const char * bn1 = ggml_backend_name(ud->backend1);
            const char * bn2 = ggml_backend_name(ud->backend2);

            if (t1->op == GGML_OP_NONE) {
                // sentinels must be unchanged
                std::vector<uint8_t> t1_data(ggml_nbytes(t1));
                std::vector<uint8_t> t2_data(ggml_nbytes(t2));
                ggml_backend_tensor_get(t1, t1_data.data(), 0, ggml_nbytes(t1));
                ggml_backend_tensor_get(t2, t2_data.data(), 0, ggml_nbytes(t2));

                if (memcmp(t1_data.data(), t2_data.data(), ggml_nbytes(t1)) != 0) {
                    ADD_FAILURE() << "sentinel mismatch: " << t1->name;
                    return true;
                }
            }

            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);

            for (size_t i = 0; i < f1.size(); i++) {
                // check for nans
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                  ADD_FAILURE() << "[" << ggml_op_desc(t1) << "] NaN at index " << i << " (" << bn1 << "=" << f1[i] << " " << bn2 << "=" << f2[i] << ")";
                  return true;
                }
                // check for infs: both must be inf of the same sign, or both must be finite
                if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) {
                    if (isinf_or_max(f1[i]) && isinf_or_max(f2[i])) {
                        if (std::signbit(f1[i]) != std::signbit(f2[i])) {
                            ADD_FAILURE() << "[" << ggml_op_desc(t1) << "] inf sign mismatch: " << bn1 << "=" << f1[i] << " " << bn2 << "=" << f2[i];
                            return true;
                        }
                    } else {
                        ADD_FAILURE() << "[" << ggml_op_desc(t1) << "] inf mismatch: " << bn1 << "=" << f1[i] << " " << bn2 << "=" << f2[i];
                        return true;
                    }
                }
            }

            double err = ud->loss(f1.data(), f2.data(), f1.size());
            EXPECT_LE(err, ud->max_err) << "[" << ggml_op_desc(t1) << "] loss = " << err << " > " << ud->max_err;
            return true;

            GGML_UNUSED(index);
        };

        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud, &out, 1);
        EXPECT_TRUE(cmp_ok) << "compare failed";
    }
};

static std::unique_ptr<test_case> make_test(const std::function<ggml_tensor* (ggml_context*)> & build_graph, std::string, float max_err = 1e-4) {
    std::unique_ptr<test_case> tc = std::make_unique<test_case>(build_graph, nmse, max_err);
    tc->max_err = max_err;
    return tc;
}

static std::string type_name(ggml_type type)
{
    return ggml_get_type_traits(type)->type_name;
}


class MetaliumTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    cpu_backend.reset(ggml_backend_cpu_init());

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("Metalium");
    if(reg == nullptr) {
      GTEST_SKIP() << "Cannot find the Metalium backend. Is the Metalium backend disabled?";
    }
    if(ggml_backend_reg_dev_count(reg) == 0) {
      GTEST_SKIP() << "No devices found for Metalium backend. Is the kernel driver working?";
    }
    metalium_backend.reset(ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr));
  }

  static void TearDownTestSuite() {
    metalium_backend = nullptr;
    cpu_backend = nullptr;
  }

  static ggml_backend_ptr cpu_backend;
  static ggml_backend_ptr metalium_backend;

  template<typename F>
  void compare_graph(F&& f) {
    test_case{std::forward<F>(f)}.eval(cpu_backend.get(), metalium_backend.get());
  }
};

ggml_backend_ptr MetaliumTest::cpu_backend = nullptr;
ggml_backend_ptr MetaliumTest::metalium_backend = nullptr;



class MetaliumUnaryOpTest
  : public MetaliumTest,
    public ::testing::WithParamInterface<std::tuple<ggml_unary_op, ggml_type>> {
};

TEST_P(MetaliumUnaryOpTest, UnaryOp) {
  ggml_unary_op op;
  ggml_type type;

  std::tie(op, type) = GetParam();
  compare_graph([op, type](ggml_context* ctx) {
    ggml_tensor* a = ggml_new_tensor_2d(ctx, type, 64, 64);
    return ggml_unary(ctx, a, op);
  });
}

INSTANTIATE_TEST_SUITE_P(
  UnaryOp,
  MetaliumUnaryOpTest,
  ::testing::Combine(
    ::testing::ValuesIn({
      GGML_UNARY_OP_ABS,
      GGML_UNARY_OP_SGN,
      GGML_UNARY_OP_NEG,
      GGML_UNARY_OP_STEP, // Not supported by Metalium
      GGML_UNARY_OP_TANH,
      GGML_UNARY_OP_ELU,
      GGML_UNARY_OP_RELU,
      GGML_UNARY_OP_SIGMOID,
      GGML_UNARY_OP_GELU,
      GGML_UNARY_OP_GELU_QUICK,
      GGML_UNARY_OP_SILU,
      GGML_UNARY_OP_HARDSWISH,
      GGML_UNARY_OP_HARDSIGMOID,
      GGML_UNARY_OP_EXP
    }),
    ::testing::ValuesIn({
      GGML_TYPE_F32,
      // GGML_TYPE_F16,
      // GGML_TYPE_BF16
      // GGML_TYPE_Q8_0,
      // GGML_TYPE_Q5_0,
      // GGML_TYPE_Q4_0
    })
  ),
  [](const testing::TestParamInfo<std::tuple<ggml_unary_op, ggml_type>>& info) {
    return std::string(ggml_unary_op_name(std::get<0>(info.param))) + "__" + ggml_type_name(std::get<1>(info.param));
  }
);

TEST_F(MetaliumTest, activation_of_view) {
  compare_graph([](ggml_context* ctx) {
      ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 96, 96);
      ggml_tensor* v = ggml_view_2d(ctx, a, 64, 64, a->nb[1], 0);
      return ggml_unary(ctx, v, GGML_UNARY_OP_ABS);
  });
}

TEST_F(MetaliumTest, CONT_on_real_tensor) {
  compare_graph([](ggml_context* ctx) {
    ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    ggml_tensor* b = ggml_cont(ctx, a);
    return b;
  });
}

TEST_F(MetaliumTest, CONT_on_integer_tensor) {
  compare_graph([](ggml_context* ctx) {
      ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 64, 64);
      ggml_tensor* b = ggml_cont(ctx, a);
      return b;
  });
}

TEST_F(MetaliumTest, noop_view) {
  compare_graph([](ggml_context* ctx) {
      ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
      ggml_tensor* view = ggml_view_2d(ctx, a, 64, 64, a->nb[1], 0);
      ggml_tensor* b = ggml_cont(ctx, view);
      return b;
  });
}

TEST_F(MetaliumTest, View_into_2D_matrix) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 32, 32, a->nb[1], 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    });
}

TEST_F(MetaliumTest, View_flat_buffer_into_2D_matrix) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32*32);
        ggml_tensor* view = ggml_view_1d(ctx, a, 32*32, 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    });
}

TEST_F(MetaliumTest, View_flat_buffer_into_2D_matrix_2) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32*32);
        ggml_tensor* view = ggml_view_2d(ctx, a, 32, 32, 32 * sizeof(float), 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    });
}

TEST_F(MetaliumTest, Transposed_flat_buffer_into_2D_matrix) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32*32);
        ggml_tensor* view = ggml_view_2d(ctx, a, 32, 32, 32 * sizeof(float), 0);
        ggml_tensor* transposed = ggml_transpose(ctx, view);
        ggml_tensor* b = ggml_cont(ctx, transposed);
        return b;
    });
}

TEST_F(MetaliumTest, View_flat_buffer_into_3D_tensor) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32 * 32 * 64);
        ggml_tensor* view = ggml_view_3d(ctx, a, 32, 32, 32, 32 * sizeof(float), 32 * 32 * sizeof(float), 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    });
}

TEST_F(MetaliumTest, View_flat_buffer_into_3D_tensor_with_offset) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32 * 32 * 64);
        ggml_tensor* view = ggml_view_3d(ctx, a, 32, 32, 32, 32 * sizeof(float), 32 * 32 * sizeof(float), 32 * sizeof(float));
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    });
}

TEST_F(MetaliumTest, View_flat_buffer_into_3D_tensor_transposed) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32 * 32 * 64);
        ggml_tensor* view = ggml_view_3d(ctx, a, 32, 32, 32, 32 * sizeof(float), 32 * 32 * sizeof(float), 0);
        ggml_tensor* transposed = ggml_transpose(ctx, view);
        ggml_tensor* b = ggml_cont(ctx, transposed);
        return b;
    });
}

TEST_F(MetaliumTest, View_into_2D_matrix_non_tile_aligned) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 30, 30, a->nb[1], 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    });
}

TEST_F(MetaliumTest, View_into_2D_matrix_with_offset) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 32, 32, a->nb[1], ggml_type_size(GGML_TYPE_F32));
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    });
}

TEST_F(MetaliumTest, ID_view_into_2D_matrix) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 48, 1, a->nb[1], 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    });
}

TEST_F(MetaliumTest, Rectangular_view_into_2D_matrix) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 30, 28, a->nb[1], 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    });
}

TEST_F(MetaliumTest, transpose_2D_square_matrix) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        return ggml_transpose(ctx, a);
    });
}

TEST_F(MetaliumTest, transpose_2D_rectangular_matrix) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 28);
        return ggml_transpose(ctx, a);
    });
}

TEST_F(MetaliumTest, transpose_2D_small_matrix) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
        return ggml_transpose(ctx, a);
    });
}

TEST_F(MetaliumTest, transpose_3D_square_matrix) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 64, 64);
        return ggml_transpose(ctx, a);
    });
}

TEST_F(MetaliumTest, transpose_3D_square_matrix_cont) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 64, 64);
        return ggml_cont(ctx, ggml_transpose(ctx, a));
    });
}

#if 0
    // Failing - need to support tensor copy
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 4, 4, 4);
    //     ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 256, 4, 4, 4);
    //     return ggml_cpy(ctx, a, b);
    // }, "4D tensor copy"));

    // Failing - need to support sum
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 14, 2, 3);
    //     return ggml_sum(ctx, a);
    // }, "sum"));

    // Failing - need to support sum_rows
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 14, 2, 3);
    //     return ggml_sum_rows(ctx, a);
    // }, "sum rows"));

    // Failing
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 4, 4, 4);
    //     ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 16, 1, 4);
    //     return ggml_cpy(ctx, a, b);
    // }, "Copy tensor into tensor of different shape"));

    // FIXME: This sould work but is failing
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 4, 4, 4);
    //     return ggml_view_2d(ctx, ggml_transpose(ctx, a), 4, 12, 4 * 4, 0);
    // }, "View of transposed 4D tensor"));

#endif

TEST_F(MetaliumTest, Reshape_to_tile_aligned_tensor) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 64, 4, 1);
        return ggml_reshape_4d(ctx, a, 32, 128, 4, 1);
    });
}

TEST_F(MetaliumTest, Reshape_to_non_tile_aligned_tensor) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 32, 1, 1);
        return ggml_reshape_4d(ctx, a, 16, 32, 2, 1);
    });
}

TEST_F(MetaliumTest, Tensor_duplication) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
        return ggml_dup(ctx, a);
    });
}

TEST_F(MetaliumTest, Tensor_duplication_via_view) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
        return ggml_dup(ctx, ggml_view_tensor(ctx, a));
    });
}

TEST_F(MetaliumTest, Write_via_view) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
        ggml_tensor* view = ggml_view_tensor(ctx, a);
        ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
        return ggml_cpy(ctx, view, b);
    });
}

    // Not working yet. Need write support for views
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
    //     ggml_tensor* view = ggml_view_2d(ctx, a, 8, 12, a->nb[1], 1);
    //     ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 12);
    //     return ggml_cpy(ctx, view, b);
    // }, "partial write via view"));
    // TODO: Expend this to attempt all permutations possible

TEST_F(MetaliumTest, Permute) {
    std::array<int, GGML_MAX_DIMS> permute_order;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        permute_order[i] = i;
    }
    do {
        std::string name = "Permutation, order=[";
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            name += std::to_string(permute_order[i]) + " ";
        }
        name.pop_back();
        name += "]";
        SCOPED_TRACE(name);
        compare_graph([permute_order](ggml_context* ctx) {
            ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 32, 8, 4);
            return ggml_permute(ctx, a, permute_order[0], permute_order[1], permute_order[2], permute_order[3]);
        });
    } while (std::next_permutation(permute_order.begin(), permute_order.end()));
}

    // (Basics of) what we need to get KV cache working
    // TODO: Map GGML operations into TTNN nlp_kv_cache_load_slice and update_cache_multi_core

TEST_F(MetaliumTest, Set_row_of_2D_matrix) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 24);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        return ggml_set_2d(ctx, a, b, b->nb[1], 0);
    });
}

TEST_F(MetaliumTest, Set_row_of_2D_matrix_with_offset) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 24);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        return ggml_set_2d(ctx, a, b, b->nb[1], a->nb[1]);
    });
}

TEST_F(MetaliumTest, Matrix_multiplication_2D) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 64);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 64);
        return ggml_mul_mat(ctx, a, b);
    });
}

TEST_F(MetaliumTest, Matrix_multiplication_2D_non_square_result) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 64);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 128);
        return ggml_mul_mat(ctx, a, b);
    });
}

TEST_F(MetaliumTest, Matrix_multiplication_2D_non_square_non_tile_aligned) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 38, 64);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 38, 72);
        return ggml_mul_mat(ctx, a, b);
    });
}

TEST_F(MetaliumTest, Matrix_multiplication_4D) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 64, 1, 10);
        ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 64, 1, 10);
        return ggml_mul_mat(ctx, a, b);
    });
}

TEST_F(MetaliumTest, Matrix_multiplication_4D_with_broadcast) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 64, 1, 1);
        ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 64, 1, 10);
        return ggml_mul_mat(ctx, a, b);
    });
}

TEST_F(MetaliumTest, matrix_vector_multiplication) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        return ggml_mul_mat(ctx, a, b);
    });
}

TEST_F(MetaliumTest, matrix_vector_multiplication_non_tile_aligned) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 18);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 24);
        return ggml_mul_mat(ctx, a, b);
    });
}

TEST_F(MetaliumTest, Add_broadcasted_vector_to_matrix) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 64);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
        return ggml_add(ctx, a, b);
    });
}

    // TODO: TTNN does not support the style of broadcasting GGML wants
    // Failing
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 64, 20);
    //     ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 64, 10);
    //     return ggml_mul_mat(ctx, a, b);
    // }, "3D matrix multiplication (broadcast)"));

// Misc

TEST_F(MetaliumTest, Clamp) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 38, 64, 3, 26);
        return ggml_clamp(ctx, a, -0.1, 0.25);
    });
}

TEST_F(MetaliumTest, Scale) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 38, 64, 3, 26);
        return ggml_scale(ctx, a, 2.0);
    });
}

TEST_F(MetaliumTest, Scale_in_place) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 38, 64, 3, 26);
        return ggml_scale_inplace(ctx, a, 1.5);
    });
}

    // RoPE
#if 0
    for(auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) { // Really a limitation of GGML's CPU implementation - we support more
        tests.push_back(make_test([type](ggml_context* ctx) {
            ggml_tensor* a = ggml_new_tensor_3d(ctx, type, 2048, 16, 2);
            ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
            return ggml_rope(ctx, a, b, 128, GGML_ROPE_TYPE_NEOX);
        }, "RoPE NEOX " + std::string(ggml_type_name(type))));

        tests.push_back(make_test([type](ggml_context* ctx) {
            float freq_base = 20000.f;
            float freq_scale = 1.4245f;
            float attn_factor = 1.424500f;
            float ext_factor = 0.746500f;
            float beta_fast = 32.f;
            float beta_slow = 1.f;
            ggml_tensor* a = ggml_new_tensor_3d(ctx, type, 2048, 16, 2);
            ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
            return ggml_rope_ext(ctx, a, b, NULL, 128, GGML_ROPE_TYPE_NEOX, 512, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        }, "RoPE NEOX " + std::string(ggml_type_name(type)) + " with YaRN"));

        tests.push_back(make_test([type](ggml_context* ctx) {
            ggml_tensor* a = ggml_new_tensor_3d(ctx, type, 512, 32, 1);
            ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            return ggml_rope(ctx, a, b, 512, GGML_ROPE_TYPE_NEOX);
        }, "RoPE NEOX in Gemma " + std::string(ggml_type_name(type))));

        tests.push_back(make_test([](ggml_context* ctx) {
            ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 32, 1);
            ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            return ggml_rope(ctx, a, b, 32, GGML_ROPE_TYPE_NORMAL);
        }, "RoPE Normal " + std::string(ggml_type_name(GGML_TYPE_F32))));
    }
#endif

    // TODO: Need a way to inform the RNG
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     int n = 300*256;
    //     int m = 60;
    //     int r = 8;
    //     int be1 = 1;
    //     int be2 = 1;
    //     ggml_tensor * in = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n, m, be1, be2);
    //     ggml_tensor * rows = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, r, be1, be2);
    //     ggml_tensor * out = ggml_get_rows(ctx, in, rows);
    //
    // // DITTO
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor * dst = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 32, 200, 1, 1);

    //     ggml_tensor * src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 1, 1, 1);
    //     ggml_tensor * idx = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1);

    //     ggml_tensor * out = ggml_set_rows(ctx, dst, src, idx);

    //     return out;
    // }, "test MM", 1e-5));

    //     return out;
    // }, "Simple GET_ROWS", 1e-5));

// more complex tests

TEST_F(MetaliumTest, Multi_layer_perceptron) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 18);
        ggml_tensor* w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 64);
        ggml_tensor* b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        ggml_tensor* w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 48);
        ggml_tensor* b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 48);

        ggml_tensor* h1 = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w1, x), b1));
        ggml_tensor* h2 = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w2, h1), b2));

        return h2;
    });
}

TEST_F(MetaliumTest, MLP_mixer) {
    compare_graph([](ggml_context* ctx) {
        ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
        std::array<ggml_tensor*, 4> h;
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                ggml_tensor* patch = ggml_view_2d(ctx, in, 32, 32, in->nb[1], 32 * y + x * 32);
                h[y * 2 + x] = ggml_relu(ctx, ggml_mul_mat(ctx, w1, patch));
            }
        }
        ggml_tensor* h1 = ggml_concat(ctx, h[0], h[1], 1);
        ggml_tensor* h2 = ggml_concat(ctx, h[2], h[3], 1);
        ggml_tensor* h_all = ggml_concat(ctx, h1, h2, 1);
        return ggml_transpose(ctx, h_all);
    });
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
