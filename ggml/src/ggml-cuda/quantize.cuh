//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "common.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_row_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_mmq_q8_1_id_cuda(
    const float * x, void * vy, const char * row_mapping, const int64_t kx0, const int64_t kx1, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

// For now only applicable for tensors with ne[1] = 1, ne[3] = 1, and useful if ne[2] > 1
void quantize_tensor_q8_1_cuda(const struct ggml_tensor * src, void * vy, const enum ggml_type type, cudaStream_t stream);

// GGML_Z6_Q8_REUSE: a supported producer also writes Q8_1, including padding.
// Storage belongs to the CUDA context and remains stable for captured graphs.
block_q8_1 * ggml_cuda_z6_q8_producer(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
block_q8_1 * ggml_cuda_z6_q8_flat_producer(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
const char * ggml_cuda_z6_q8_consumer(ggml_backend_cuda_context & ctx, const ggml_tensor * src);
void * ggml_cuda_z6_gemm_scratch(ggml_backend_cuda_context & ctx);
void ggml_cuda_z6_q8_trace(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
    const ggml_tensor * dst, const ggml_cgraph * graph, int node);

struct ggml_cuda_z6_q8_input {
    ggml_cuda_pool_alloc<char> temporary;
    char * ptr;
    bool reused;

    ggml_cuda_z6_q8_input(ggml_backend_cuda_context & ctx, const ggml_tensor * src, size_t bytes, bool compatible) :
        temporary(ctx.pool()), ptr(compatible ? const_cast<char *>(ggml_cuda_z6_q8_consumer(ctx, src)) : nullptr),
        reused(ptr != nullptr) {
        if (!ptr) {
            ptr = temporary.alloc(bytes);
        }
    }

    char * get() const { return ptr; }
};

static __device__ __forceinline__ void ggml_cuda_z6_store_q8(float v, block_q8_1 * q, int i) {
    const float amax = warp_reduce_max(fabsf(v));
    const float sum = warp_reduce_sum(v);
    const float d = amax / 127;
    q[i / QK8_1].qs[i % QK8_1] = amax == 0.0f ? 0 : roundf(v / d);
    if (i % QK8_1 == 0) {
        reinterpret_cast<half &>(q[i / QK8_1].ds.x) = d;
        reinterpret_cast<half &>(q[i / QK8_1].ds.y) = sum;
    }
}
