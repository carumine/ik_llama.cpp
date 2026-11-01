#include "z6-hc.cuh"
#include "quantize.cuh"

// Z6: hyper-connection fuse di qwen4exp. Ognuno sostituisce 2-5 kernel da circa 1-2 µs (nella generazione
// la GPU esegue circa 3000 kernel per token); le somme seguono l'ordine delle operazioni separate.

// dst[e,h,t] = res[e,h,t] + b[e,t] * s_out*sigmoid(s_in*inj[h,t])
static __global__ void k_z6_hc_combine(const float * __restrict__ res, const float * __restrict__ b,
        const float * __restrict__ inj, float * __restrict__ dst, const int E, const int H, const int n,
        const float s_in, const float s_out) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int e = i % E;
    const int row = i / E;           // h + H*t
    const int t = row / H;
    const float w = s_out / (1.0f + expf(-s_in*inj[row]));
    dst[i] = res[i] + b[t*E + e]*w;
}

// dst[e,t] = scale * sum_h sigmoid(up[e+h*E,t]) * xn[e+h*E,t]
template <bool q8 = false>
static __global__ void k_z6_hc_gate_sum(const float * __restrict__ up, const float * __restrict__ xn,
        float * __restrict__ dst, const int E, const int H, const int n, const float scale, block_q8_1 * q = nullptr) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int e = i % E;
    const int t = i / E;
    float s = 0.0f;
    for (int h = 0; h < H; ++h) {
        const int j = t*E*H + h*E + e;
        s += xn[j] / (1.0f + expf(-up[j]));
    }
    dst[i] = s*scale;
    if constexpr (q8) {
        ggml_cuda_z6_store_q8(s*scale, q, i);
    }
}

// dst[e+h*E,t] = x[e,h,t] / sqrt(mean_e x^2 + eps) * w[e+h*E]: un blocco per riga (h,t)
template <int block_size, bool q8 = false>
static __global__ void k_z6_hc_norm(const float * __restrict__ x, const float * __restrict__ w,
        float * __restrict__ dst, const int E, const int H, const float eps, block_q8_1 * q = nullptr) {
    const int row = blockIdx.x;
    const int h = row % H;
    const float * xr = x + (int64_t) row*E;
    float sum = 0.0f;
    for (int e = threadIdx.x; e < E; e += block_size) {
        sum += xr[e]*xr[e];
    }
    sum = warp_reduce_sum(sum);
    if constexpr (block_size > WARP_SIZE) {
        __shared__ float s_sum[block_size/WARP_SIZE];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = sum;
        }
        __syncthreads();
        sum = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
    }
    const float scale = rsqrtf(sum/E + eps);
    float * dr = dst + (int64_t) row*E;
    const float * wr = w + (int64_t) h*E;
    for (int e = threadIdx.x; e < E; e += block_size) {
        dr[e] = xr[e]*scale*wr[e];
        if constexpr (q8) {
            ggml_cuda_z6_store_q8(xr[e]*scale*wr[e], q, row*E + e);
        }
    }
}

// One CTA per injection output. Each CTA normalizes all four streams locally,
// so the four dots need no cross-CTA barrier. CTA zero also publishes xn/Q8.
template <bool q8>
static __global__ void k_z6_hc_norm_inject(const float * __restrict__ x, const float * __restrict__ w,
        const float * __restrict__ wi, float * __restrict__ dst, int E, float eps, block_q8_1 * q) {
    constexpr int group = 256;
    const int tid = threadIdx.x, h = tid/group, local = tid%group;
    const int warp = tid/WARP_SIZE, lane = tid%WARP_SIZE;
    const int n = 4*E, r = blockIdx.x;
    __shared__ float sums[32];
    float sum = 0.0f;
    for (int e = local; e < E; e += group) {
        const float v = x[h*E + e];
        sum += v*v;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) sums[warp] = sum;
    __syncthreads();
    sum = lane < group/WARP_SIZE ? sums[h*(group/WARP_SIZE) + lane] : 0.0f;
    sum = warp_reduce_sum(sum);
    const float scale = rsqrtf(sum/E + eps);
    float dot = 0.0f;
    for (int e = local; e < E; e += group) {
        const int i = h*E + e;
        const float v = x[i]*scale*w[i];
        dot += v*wi[r*n + i];
        if (r == 0) {
            dst[i] = v;
            if constexpr (q8) ggml_cuda_z6_store_q8(v, q, i);
        }
    }
    dot = warp_reduce_sum(dot);
    // All warps have consumed their normalization partials before reusing sums.
    __syncthreads();
    if (lane == 0) sums[warp] = dot;
    __syncthreads();
    if (warp == 0) {
        dot = warp_reduce_sum(sums[lane]);
        if (lane == 0) dst[n + r] = dot;
    }
}

// Decode: SiLU over Q/K/V, then L2 over each Q/K head. V stays unnormalized.
static __global__ void k_z6_ssm_silu_norm(const float * __restrict__ x, float * __restrict__ dst,
        int key_dim, int head_dim, float eps) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const int i = row*head_dim + tid;
    float v = tid < head_dim ? x[i] / (1.0f + expf(-x[i])) : 0.0f;
    if (row*head_dim < 2*key_dim) {
        float sum = warp_reduce_sum(v*v);
        __shared__ float sums[8];
        const int lane = tid%WARP_SIZE, warp = tid/WARP_SIZE;
        if (lane == 0) sums[warp] = sum;
        __syncthreads();
        sum = warp_reduce_sum(lane < 8 ? sums[lane] : 0.0f);
        v *= rsqrtf(fmaxf(sum, eps*eps));
    }
    if (tid < head_dim) dst[i] = v;
}

// gather_f16: celle idx di una cache Q8_0 convertite in f16. Sostituisce get_rows dei byte e la conversione
// q8_0 -> f16 che flash attention faceva dopo: un passaggio sulla memoria invece di tre.
static __global__ void k_z6_gather_q8_0_f16(const block_q8_0 * __restrict__ cache, const int * __restrict__ idx,
        half * __restrict__ dst, const int blocchi_cella) {
    const int c = blockIdx.x;
    const block_q8_0 * src = cache + (int64_t) idx[c]*blocchi_cella;
    half * out = dst + (int64_t) c*blocchi_cella*QK8_0;
    for (int i = threadIdx.x; i < blocchi_cella*QK8_0; i += blockDim.x) {
        const block_q8_0 & b = src[i/QK8_0];
        out[i] = __float2half(__half2float(b.d)*b.qs[i % QK8_0]);
    }
}

void ggml_cuda_op_z6_hc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int modo = dst->op_params[0];
    float p0, p1;
    memcpy(&p0, &dst->op_params[1], sizeof(p0));
    memcpy(&p1, &dst->op_params[2], sizeof(p1));
    cudaStream_t stream = ctx.stream();
    float * d = (float *) dst->data;
    constexpr int kBlock = 256;
    if (modo == 6) {
        const int key_dim = dst->op_params[3], head_dim = dst->op_params[4];
        k_z6_ssm_silu_norm<<<dst->ne[0]/head_dim, 256, 0, stream>>>((const float *) dst->src[0]->data,
                d, key_dim, head_dim, p0);
    } else if (modo == 4) {
        const auto * x = dst->src[0];
        auto norm = *dst;
        norm.ne[0] = x->ne[0]*x->ne[1];
        norm.nb[1] = norm.nb[2] = norm.nb[3] = norm.ne[0]*sizeof(float);
        auto q = x->ne[0] % MATRIX_ROW_PADDING == 0 ? ggml_cuda_z6_q8_producer(ctx, &norm) : nullptr;
        if (q) {
            k_z6_hc_norm_inject<true><<<4, 1024, 0, stream>>>((const float *) x->data,
                    (const float *) dst->src[1]->data, (const float *) dst->src[2]->data, d, x->ne[0], p0, q);
        } else {
            k_z6_hc_norm_inject<false><<<4, 1024, 0, stream>>>((const float *) x->data,
                    (const float *) dst->src[1]->data, (const float *) dst->src[2]->data, d, x->ne[0], p0, nullptr);
        }
    } else if (modo == 0) {
        const ggml_tensor * res = dst->src[0];
        const int E = res->ne[0], H = res->ne[1];
        const int n = ggml_nelements(res);
        k_z6_hc_combine<<<(n + kBlock - 1)/kBlock, kBlock, 0, stream>>>((const float *) res->data,
                (const float *) dst->src[1]->data, (const float *) dst->src[2]->data, d, E, H, n, p0, p1);
    } else if (modo == 1) {
        const int H = dst->op_params[3], E = dst->ne[0];
        const int n = ggml_nelements(dst);
        auto q = E % MATRIX_ROW_PADDING == 0 ? ggml_cuda_z6_q8_producer(ctx, dst) : nullptr;
        if (q) {
            k_z6_hc_gate_sum<true><<<(n + kBlock - 1)/kBlock, kBlock, 0, stream>>>((const float *) dst->src[0]->data,
                (const float *) dst->src[1]->data, d, E, H, n, p0, q);
        } else {
            k_z6_hc_gate_sum<<<(n + kBlock - 1)/kBlock, kBlock, 0, stream>>>((const float *) dst->src[0]->data,
                (const float *) dst->src[1]->data, d, E, H, n, p0);
        }
    } else if (modo == 3) {
        const ggml_tensor * cache = dst->src[0];
        GGML_ASSERT(cache->type == GGML_TYPE_Q8_0);
        const int blocchi_cella = dst->ne[0]*dst->ne[1]/QK8_0;
        k_z6_gather_q8_0_f16<<<dst->ne[2], kBlock, 0, stream>>>((const block_q8_0 *) cache->data,
                (const int *) dst->src[1]->data, (half *) dst->data, blocchi_cella);
    } else {
        const ggml_tensor * x = dst->src[0];
        const int E = x->ne[0], H = x->ne[1];
        const int rows = x->ne[1]*x->ne[2];
        auto q = E % MATRIX_ROW_PADDING == 0 ? ggml_cuda_z6_q8_producer(ctx, dst) : nullptr;
        if (q) {
            k_z6_hc_norm<1024, true><<<rows, 1024, 0, stream>>>((const float *) x->data, (const float *) dst->src[1]->data,
                d, E, H, p0, q);
        } else {
            k_z6_hc_norm<1024><<<rows, 1024, 0, stream>>>((const float *) x->data, (const float *) dst->src[1]->data,
                d, E, H, p0);
        }
    }
}
