// Z6 (GGML_Z6_FREDDI_SENZA_CALDI): a MoE layer without hot experts computed in two ways through the real scheduler,
// with the real CPU and CUDA kernels, and compared bit for bit.
//   A, normal path (llm_build_moe_ffn with fused_mmad, then + shared expert as build_glm5next does): the scheduler
//      copies x and the IDs (a strided view, like ggml_top_k) to the CPU, the CPU runs moe_up_gate and mul_mat_id,
//      the GPU runs mul_multi_add and the add of the shared expert;
//   B, group with zero hot experts (the new branch): x, contiguous IDs and router weights staged in a pinned packet by
//      z6_caldi_prepare, the shared expert as the GPU piece of the group (first node = hot_begin, output = hot_end),
//      the cold output staged in the packet and uploaded through the ticket, then the same mul_multi_add and add.
// Several epochs with fresh inputs and expert choices, 1 and 2 tokens. The GGML_Z6_* variables of the CPU kernels are
// read once per process: run it also with those of the preset (GGML_Z6_FREDDI_CODA=16 GGML_Z6_FREDDI_CODA_RIF=4
// GGML_Z6_FREDDI_CODA_RIGHE=2 GGML_Z6_PRINCIPALE_LIBERO=1 ...) so that A and B go through the production CPU paths.
//   test-z6-senza-caldi [CUDA0] [thread]      exit 0 and "TUTTO IDENTICO" when every output matches
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace {

void check(bool ok, const char * why) {
    if (!ok) { fprintf(stderr, "FAIL %s\n", why); std::exit(1); }
}
bool is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}
struct pin_free { void operator()(void * p) const { if (p) ggml_backend_cuda_z6_host_free(p); } };

constexpr int n_embd = 1024, n_ff = 512, n_exp = 32, n_used = 8, n_sorted = 32;

struct layer {
    ggml_context_ptr cpu_ctx, gpu_ctx;
    ggml_backend_buffer_ptr cpu_buf, gpu_buf;
    ggml_tensor * up = nullptr, * gate = nullptr, * down = nullptr;   // experts in RAM (CPU)
    ggml_tensor * sh = nullptr;                                        // shared expert in VRAM
};

void quantized(ggml_tensor * t, std::mt19937 & rng, float scale) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> f(ggml_nelements(t));
    for (auto & v : f) v = d(rng);
    std::vector<uint8_t> q(ggml_nbytes(t));
    ggml_quantize_chunk(t->type, f.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr, nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

layer make_layer(ggml_backend_t gpu, ggml_backend_t cpu, std::mt19937 & rng) {
    layer l;
    l.cpu_ctx.reset(ggml_init({4*ggml_tensor_overhead(), nullptr, true}));
    l.up   = ggml_new_tensor_3d(l.cpu_ctx.get(), GGML_TYPE_Q8_0, n_embd, n_ff, n_exp);
    l.gate = ggml_new_tensor_3d(l.cpu_ctx.get(), GGML_TYPE_Q8_0, n_embd, n_ff, n_exp);
    l.down = ggml_new_tensor_3d(l.cpu_ctx.get(), GGML_TYPE_Q8_0, n_ff, n_embd, n_exp);
    l.cpu_buf.reset(ggml_backend_alloc_ctx_tensors(l.cpu_ctx.get(), cpu));
    check(bool(l.cpu_buf), "expert allocation");
    ggml_backend_buffer_set_usage(l.cpu_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    l.gpu_ctx.reset(ggml_init({2*ggml_tensor_overhead(), nullptr, true}));
    l.sh = ggml_new_tensor_2d(l.gpu_ctx.get(), GGML_TYPE_Q8_0, n_embd, n_embd);
    l.gpu_buf.reset(ggml_backend_alloc_ctx_tensors(l.gpu_ctx.get(), gpu));
    check(bool(l.gpu_buf), "shared expert allocation");
    ggml_backend_buffer_set_usage(l.gpu_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    quantized(l.up, rng, 0.05f); quantized(l.gate, rng, 0.05f); quantized(l.down, rng, 0.05f); quantized(l.sh, rng, 0.03f);
    return l;
}

// inputs on the GPU, shared by A and B: the activation before the scale, the sorted expert list, the router weights
struct inputs {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor * act = nullptr, * sorted = nullptr, * w = nullptr;
};
inputs make_inputs(ggml_backend_t gpu, int nt) {
    inputs in;
    in.ctx.reset(ggml_init({4*ggml_tensor_overhead(), nullptr, true}));
    in.act = ggml_new_tensor_3d(in.ctx.get(), GGML_TYPE_F32, n_embd, 1, nt);
    in.sorted = ggml_new_tensor_2d(in.ctx.get(), GGML_TYPE_I32, n_sorted, nt);
    in.w = ggml_new_tensor_3d(in.ctx.get(), GGML_TYPE_F32, 1, n_used, nt);
    in.buf.reset(ggml_backend_alloc_ctx_tensors(in.ctx.get(), gpu));
    check(bool(in.buf), "input allocation");
    return in;
}

void set_unary_params(ggml_tensor * par) {
    // the values the normal path writes: swiglu_limit(il, false), situ_beta, situ_linear_beta (0 for GLM)
    *(float *)(par->op_params + 1) = 0.0f;
    *(float *)(par->op_params + 2) = 0.0f;
    *(float *)(par->op_params + 3) = 0.0f;
}

ggml_tensor * shared_expert(ggml_context * ctx, const layer & l, ggml_tensor * x, int nt) {
    auto s = ggml_mul_mat(ctx, l.sh, ggml_reshape_2d(ctx, x, n_embd, nt));
    return ggml_silu(ctx, s);
}

struct graph_a {
    ggml_context_ptr ctx;
    ggml_backend_sched_ptr sched;
    ggml_cgraph * g = nullptr;
    ggml_tensor * out = nullptr;
    std::vector<ggml_tensor *> on_gpu, on_cpu;
};
graph_a build_a(ggml_backend_t gpu, ggml_backend_t cpu, const layer & l, const inputs & in, int nt) {
    graph_a a;
    ggml_backend_t backends[] = {gpu, cpu};
    a.sched.reset(ggml_backend_sched_new(backends, nullptr, 2, 1024, false));
    a.ctx.reset(ggml_init({256*ggml_tensor_overhead() + ggml_graph_overhead_custom(1024, false), nullptr, true}));
    auto ctx = a.ctx.get();
    a.g = ggml_new_graph_custom(ctx, 1024, false);
    auto x = ggml_scale(ctx, in.act, 1.25f);
    auto sel = ggml_view_2d(ctx, in.sorted, n_used, nt, in.sorted->nb[1], 0);   // strided, like ggml_top_k
    ggml_build_forward_expand(a.g, in.w);
    auto par = ggml_moe_up_gate(ctx, l.up, l.gate, x, sel, GGML_UNARY_OP_SILU);
    set_unary_params(par);
    auto down = ggml_mul_mat_id(ctx, l.down, par, sel);
    auto sum = ggml_mul_multi_add(ctx, down, in.w);
    ggml_build_forward_expand(a.g, sum);
    auto shared = shared_expert(ctx, l, x, nt);
    a.out = ggml_add(ctx, sum, shared);
    ggml_build_forward_expand(a.g, a.out);
    // the normal path leaves the placement to the scheduler; here it is explicit: the experts on the CPU, the rest on
    // the GPU, which is where the model runs them (the weighted sum is uploaded as [n_embd, n_used] and summed there)
    a.on_gpu = {x, sum, a.out};
    for (int i = 0; i < ggml_graph_n_nodes(a.g); ++i) {
        auto t = ggml_graph_node(a.g, i);
        if (t != par && t != down && !is_view(t)) a.on_gpu.push_back(t);
    }
    a.on_cpu = {par, down};
    for (auto t : a.on_gpu) ggml_backend_sched_set_tensor_backend(a.sched.get(), t, gpu);
    for (auto t : a.on_cpu) ggml_backend_sched_set_tensor_backend(a.sched.get(), t, cpu);
    check(ggml_backend_sched_alloc_graph(a.sched.get(), a.g), "graph A allocated");
    return a;
}

struct graph_b {
    ggml_context_ptr ctx;
    ggml_backend_sched_ptr sched;
    std::unique_ptr<void, pin_free> pinned;
    ggml_backend_buffer_ptr host;
    ggml_cgraph * g = nullptr;
    ggml_tensor * out = nullptr;
    std::vector<ggml_tensor *> on_gpu, on_cpu;
};
graph_b build_b(ggml_backend_t gpu, ggml_backend_t cpu, const layer & l, const inputs & in, int nt) {
    graph_b b;
    ggml_backend_t backends[] = {gpu, cpu};
    b.sched.reset(ggml_backend_sched_new(backends, nullptr, 2, 1024, false));
    ggml_backend_sched_z6_begin_graph(b.sched.get());
    constexpr size_t packet = 4*1024*1024;
    b.pinned.reset(ggml_backend_cuda_z6_host_alloc(packet));
    check(bool(b.pinned), "pinned packet");
    b.host.reset(ggml_backend_cpu_buffer_from_ptr(b.pinned.get(), packet));
    b.ctx.reset(ggml_init({256*ggml_tensor_overhead() + ggml_graph_overhead_custom(1024, false), nullptr, true}));
    auto ctx = b.ctx.get();
    auto staged = [&](ggml_tensor * t, size_t offset) {
        t->buffer = b.host.get(); t->data = (char *) b.pinned.get() + offset;
        return t;
    };
    b.g = ggml_new_graph_custom(ctx, 1024, false);
    auto x = ggml_scale(ctx, in.act, 1.25f);
    auto sel = ggml_view_2d(ctx, in.sorted, n_used, nt, in.sorted->nb[1], 0);
    // as the layer branch: the IDs are always copied contiguous on the GPU, with one token too
    auto ids = ggml_reshape_2d(ctx, ggml_cont(ctx, sel), n_used, nt);
    b.on_gpu = {x, ids->src[0]};
    ggml_build_forward_expand(b.g, x);
    ggml_build_forward_expand(b.g, ids);
    ggml_build_forward_expand(b.g, in.w);
    auto cpu_x = staged(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, nt), 0);
    auto cpu_ids = staged(ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_used, nt, 1), 512*1024);
    auto cpu_w = staged(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, nt), 768*1024);
    const int n_before = ggml_graph_n_nodes(b.g);
    auto shared = shared_expert(ctx, l, x, nt);
    ggml_build_forward_expand(b.g, shared);
    ggml_tensor * shared_begin = nullptr;
    for (int i = n_before; i < ggml_graph_n_nodes(b.g) && !shared_begin; ++i) {
        auto t = ggml_graph_node(b.g, i);
        if (!is_view(t)) shared_begin = t;
    }
    check(shared_begin != nullptr, "shared expert first node");
    for (int i = n_before; i < ggml_graph_n_nodes(b.g); ++i) {
        if (!is_view(ggml_graph_node(b.g, i))) b.on_gpu.push_back(ggml_graph_node(b.g, i));
    }
    auto par = ggml_moe_up_gate(ctx, l.up, l.gate, cpu_x, cpu_ids, GGML_UNARY_OP_SILU);
    set_unary_params(par);
    auto down = staged(ggml_mul_mat_id(ctx, l.down, par, cpu_ids), 1024*1024);
    b.on_cpu = {par, down};
    const ggml_z6_caldi_group group = { shared_begin, shared, par, down, {x, ids, in.w}, {cpu_x, cpu_ids, cpu_w}, 0, nullptr, nullptr };
    ggml_backend_sched_z6_add_group(b.sched.get(), &group);
    auto sum = ggml_mul_multi_add(ctx, down, in.w);
    b.out = ggml_add(ctx, sum, shared);
    b.on_gpu.push_back(sum); b.on_gpu.push_back(b.out);
    ggml_build_forward_expand(b.g, b.out);
    for (auto t : b.on_gpu) ggml_backend_sched_set_tensor_backend(b.sched.get(), t, gpu);
    for (auto t : b.on_cpu) ggml_backend_sched_set_tensor_backend(b.sched.get(), t, cpu);
    check(ggml_backend_sched_alloc_graph(b.sched.get(), b.g), "graph B allocated");
    return b;
}

}  // namespace

int main(int argc, char ** argv) {
    setenv("GGML_Z6_TEMPI", "1", 1);   // counts the groups (ggml_backend_sched_z6_stats); no effect on the numbers
    const char * dev = argc > 1 ? argv[1] : "CUDA0";
    const int threads = argc > 2 ? atoi(argv[2]) : 8;
    ggml_backend_ptr gpu(ggml_backend_reg_init_backend_from_str(dev));
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    check(bool(gpu) && bool(cpu), "backend init");
    ggml_backend_cpu_set_n_threads(cpu.get(), threads);
    std::mt19937 rng(1511);
    layer l = make_layer(gpu.get(), cpu.get(), rng);
    bool all = true;
    for (int nt : {1, 2}) {
        inputs in = make_inputs(gpu.get(), nt);
        graph_a a = build_a(gpu.get(), cpu.get(), l, in, nt);
        graph_b b = build_b(gpu.get(), cpu.get(), l, in, nt);
        constexpr int epochs = 16;
        int diff_epochs = 0;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::normal_distribution<float> d(0.0f, 1.0f);
            std::vector<float> act(n_embd*nt), w(n_used*nt);
            for (auto & v : act) v = d(rng);
            // a few exact zeros and negative zeros in the activation: the sums must not touch their sign
            act[3] = 0.0f; act[5] = -0.0f;
            std::vector<int32_t> sorted(n_sorted*nt);
            for (int t = 0; t < nt; ++t) {
                std::vector<int32_t> perm(n_exp);
                for (int e = 0; e < n_exp; ++e) perm[e] = e;
                std::shuffle(perm.begin(), perm.end(), rng);
                for (int k = 0; k < n_sorted; ++k) sorted[t*n_sorted + k] = perm[k % n_exp];
                float tot = 0;
                for (int k = 0; k < n_used; ++k) { w[t*n_used + k] = std::abs(d(rng)) + 0.01f; tot += w[t*n_used + k]; }
                for (int k = 0; k < n_used; ++k) w[t*n_used + k] /= tot;
            }
            ggml_backend_tensor_set(in.act, act.data(), 0, act.size()*sizeof(float));
            ggml_backend_tensor_set(in.sorted, sorted.data(), 0, sorted.size()*sizeof(int32_t));
            ggml_backend_tensor_set(in.w, w.data(), 0, w.size()*sizeof(float));
            for (auto t : a.on_gpu) ggml_backend_sched_set_tensor_backend(a.sched.get(), t, gpu.get());
            for (auto t : a.on_cpu) ggml_backend_sched_set_tensor_backend(a.sched.get(), t, cpu.get());
            for (auto t : b.on_gpu) ggml_backend_sched_set_tensor_backend(b.sched.get(), t, gpu.get());
            for (auto t : b.on_cpu) ggml_backend_sched_set_tensor_backend(b.sched.get(), t, cpu.get());
            check(ggml_backend_sched_graph_compute(a.sched.get(), a.g) == GGML_STATUS_SUCCESS, "compute A");
            check(ggml_backend_sched_graph_compute(b.sched.get(), b.g) == GGML_STATUS_SUCCESS, "compute B");
            std::vector<float> oa(n_embd*nt), ob(n_embd*nt);
            ggml_backend_tensor_get(a.out, oa.data(), 0, oa.size()*sizeof(float));
            ggml_backend_tensor_get(b.out, ob.data(), 0, ob.size()*sizeof(float));
            bool finite = true;
            for (float v : oa) finite &= std::isfinite(v);
            check(finite, "finite output");
            if (memcmp(oa.data(), ob.data(), oa.size()*sizeof(float)) != 0) {
                ++diff_epochs;
                size_t n = 0; double maxd = 0;
                for (size_t i = 0; i < oa.size(); ++i) if (memcmp(&oa[i], &ob[i], 4)) { ++n; maxd = std::max(maxd, (double) std::abs(oa[i] - ob[i])); }
                fprintf(stderr, "token %d epoch %d: %zu valori diversi su %zu, differenza massima %g\n", nt, epoch, n, oa.size(), maxd);
            }
        }
        int64_t groups = 0, overlapping = 0;
        ggml_backend_sched_z6_stats(b.sched.get(), &groups, &overlapping);
        const bool ok = diff_epochs == 0 && groups == epochs;
        printf("token %d: %d epoche, diverse %d, gruppi eseguiti %lld (atteso %d): %s\n", nt, epochs, diff_epochs,
                (long long) groups, epochs, ok ? "identico" : "DIVERSO");
        all &= ok;
        ggml_backend_sched_reset(b.sched.get());   // drain before freeing the pinned packet
    }
    printf("%s\n", all ? "TUTTO IDENTICO" : "DIFFERENZE: vedi sopra");
    return all ? 0 : 1;
}
