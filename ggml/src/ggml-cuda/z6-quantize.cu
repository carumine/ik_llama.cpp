#include "quantize.cuh"

#include <map>
#include <atomic>

struct ggml_cuda_z6_state {
    int device;
    std::unique_ptr<ggml_cuda_pool> pool;
    std::vector<std::pair<void *, size_t>> allocations;
    // Key includes length: graph captures for different shapes must keep their
    // original addresses even when the scheduler reuses the F32 allocation.
    std::map<std::pair<uintptr_t, int64_t>, void *> q8;
    void * gemm = nullptr;

    explicit ggml_cuda_z6_state(int device_id) : device(device_id), pool(ggml_backend_cuda_context::new_pool_for_device(device_id)) {}

    ~ggml_cuda_z6_state() {
        ggml_cuda_set_device(device);
        // VMM requires LIFO frees; the legacy pool also needs the live
        // allocations returned before its destructor can release them.
        for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
            pool->free(it->first, it->second);
        }
    }

    void * alloc(size_t bytes) {
        size_t actual_size;
        void * ptr = pool->alloc(bytes, &actual_size);
        allocations.emplace_back(ptr, actual_size);
        return ptr;
    }
};

static ggml_cuda_z6_state & z6_state(ggml_backend_cuda_context & ctx) {
    if (!ctx.z6_state) {
        ctx.z6_state = std::make_shared<ggml_cuda_z6_state>(ctx.device);
    }
    return *ctx.z6_state;
}

static bool z6_q8_enabled() {
    static const bool enabled = [] { const char * e = getenv("GGML_Z6_Q8_REUSE"); return e && atoi(e) > 0; }();
    return enabled;
}

static bool z6_q8_shape(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nrows(t) == 1 &&
        t->ne[0] > 0 && t->ne[0] % QK8_1 == 0;
}

void ggml_cuda_z6_q8_trace(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
        const ggml_tensor * dst, const ggml_cgraph * graph, int node) {
    static const bool enabled = [] { const char * e = getenv("GGML_Z6_Q8_TRACE"); return e && atoi(e) > 0; }();
    static std::atomic<int> samples{0};
    if (enabled && z6_q8_shape(src) && samples.fetch_add(1) < 2048) {
        fprintf(stderr, "Z6_Q8 ctx=%p graph=%p node=%d src=%p name=%s op=%s K=%lld dst=%s\n",
            static_cast<void *>(&ctx), static_cast<const void *>(graph), node, static_cast<const void *>(src), src->name, ggml_op_name(src->op),
            (long long)src->ne[0], dst->name);
    }
}

block_q8_1 * ggml_cuda_z6_q8_producer(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    if (!z6_q8_enabled() || !z6_q8_shape(dst)) {
        return nullptr;
    }
    auto & state = z6_state(ctx);
    auto & ptr = state.q8[{reinterpret_cast<uintptr_t>(dst->data), dst->ne[0]}];
    if (!ptr) {
        ptr = state.alloc(GGML_PAD(dst->ne[0], MATRIX_ROW_PADDING) * sizeof(block_q8_1) / QK8_1);
    }
    return static_cast<block_q8_1 *>(ptr);
}

block_q8_1 * ggml_cuda_z6_q8_flat_producer(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    if (!z6_q8_enabled() || !ggml_is_contiguous(dst) || ggml_nelements(dst) > 16384) {
        return nullptr;
    }
    auto flat = *dst;
    flat.ne[0] = ggml_nelements(dst);
    flat.ne[1] = flat.ne[2] = flat.ne[3] = 1;
    flat.nb[1] = flat.nb[2] = flat.nb[3] = flat.ne[0]*sizeof(float);
    return ggml_cuda_z6_q8_producer(ctx, &flat);
}

const char * ggml_cuda_z6_q8_consumer(ggml_backend_cuda_context & ctx, const ggml_tensor * src) {
    if (!z6_q8_enabled() || !ctx.z6_state || !z6_q8_shape(src)) {
        return nullptr;
    }
    const ggml_tensor * producer = src;
    while (producer->op == GGML_OP_RESHAPE || producer->op == GGML_OP_VIEW) {
        const auto * parent = producer->src[0];
        if (parent && parent->op == GGML_OP_Z6_HC && parent->op_params[0] == 4 &&
                producer->data == parent->data && ggml_nrows(producer) == 1 &&
                producer->ne[0] == parent->src[0]->ne[0]*parent->src[0]->ne[1]) {
            producer = parent;
            break;
        }
        if (!producer->src[0] || producer->data != producer->src[0]->data ||
            ggml_nelements(producer) != ggml_nelements(producer->src[0])) {
            return nullptr;
        }
        producer = producer->src[0];
    }
    // Never infer validity just from an address. Every accepted producer writes
    // its side buffer on every execution, including CUDA graph replay.
    const bool hc = producer->op == GGML_OP_Z6_HC &&
        (((producer->op_params[0] == 1 || producer->op_params[0] == 2) && z6_q8_shape(producer)) ||
         (producer->op_params[0] == 4 && src->ne[0] == producer->src[0]->ne[0]*producer->src[0]->ne[1])) &&
        (producer->op_params[0] == 1 ? producer->ne[0] : producer->src[0]->ne[0]) % MATRIX_ROW_PADDING == 0;
    const bool silu = producer->op == GGML_OP_UNARY &&
        ggml_get_unary_op(producer) == GGML_UNARY_OP_SILU && z6_q8_shape(producer) &&
        producer->src[0]->type == GGML_TYPE_F32;
    const bool mul = producer->op == GGML_OP_MUL && producer->type == GGML_TYPE_F32 &&
        ggml_nelements(producer) == 6144 && ggml_is_contiguous(producer) &&
        producer->src[0]->type == GGML_TYPE_F32 && producer->src[1]->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(producer->src[0]) && ggml_is_contiguous(producer->src[1]) &&
        ggml_are_same_shape(producer->src[0], producer->src[1]);
    if (!hc && !silu && !mul) {
        return nullptr;
    }
    auto & q8 = ctx.z6_state->q8;
    auto it = q8.find({reinterpret_cast<uintptr_t>(src->data), src->ne[0]});
    return it == q8.end() ? nullptr : static_cast<const char *>(it->second);
}

void * ggml_cuda_z6_gemm_scratch(ggml_backend_cuda_context & ctx) {
    auto & state = z6_state(ctx);
    if (!state.gemm) {
        const size_t rows = ggml_cuda_info().devices[ctx.device].nsm;
        state.gemm = state.alloc(rows * 32*(sizeof(float) + sizeof(unsigned)));
        CUDA_CHECK(cudaMemsetAsync(static_cast<char *>(state.gemm) + rows*32*sizeof(float),
            0, rows*32*sizeof(unsigned), ctx.stream()));
    }
    return state.gemm;
}
