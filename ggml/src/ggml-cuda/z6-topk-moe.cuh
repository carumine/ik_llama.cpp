#pragma once

// One warp per token, register-only selection. Keep the reference softmax and
// normalization order; avoid the dynamically indexed wt[] store in topk_moe_cuda.
template <bool normalize>
static __global__ void z6_topk_moe_cuda(const float * logits, float * weights, int32_t * ids) {
    const int lane = threadIdx.x;
    float w[16];
    float maxv = -INFINITY;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        w[j] = logits[32*j + lane];
        maxv = fmaxf(maxv, w[j]);
    }
    maxv = warp_reduce_max(maxv);
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        w[j] = expf(w[j] - maxv);
        sum += w[j];
    }
    sum = warp_reduce_sum(sum);
    const float inv = 1.0f / sum;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        w[j] *= inv;
    }
    float selected_sum = 0.0f;
    float selected[10];
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        float best = -INFINITY;
        int expert = lane;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (w[j] > best) {
                best = w[j];
                expert = 32*j + lane;
            }
        }
        // Softmax weights are nonnegative. Ampere's integer warp reduction
        // replaces ten shuffle/compare instructions for each selected expert.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
        const float value = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(best)));
        const int winner = __reduce_min_sync(0xffffffff, best == value ? expert : INT_MAX);
#else
        const float value = warp_reduce_max(best);
        const int winner = __ffs(__ballot_sync(0xffffffff, best == value)) - 1;
        const int winner_expert = __shfl_sync(0xffffffff, expert, winner);
#endif
        selected[k] = value;
        selected_sum += value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
        const int chosen = winner;
#else
        const int chosen = winner_expert;
#endif
        if (lane == k) {
            ids[k] = chosen;
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (32*j + lane == chosen) {
                w[j] = -INFINITY;
            }
        }
    }
    const float norm = normalize ? 1.0f / selected_sum : 1.0f;
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        if (lane == k) {
            weights[k] = selected[k] * norm;
        }
    }
}
