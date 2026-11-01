#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static void check(bool ok, const char * why) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", why); std::exit(1); }
}
static void fill(ggml_tensor * t, float scale, int epoch = 0) {
    std::vector<float> v(ggml_nelements(t));
    for (size_t i = 0; i < v.size(); ++i) v[i] = scale*std::sin(float((i*17 + epoch*31)%1009));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, v.data(), 0, v.size()*4);
    } else {
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, v.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr, nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
}
static void compare(ggml_tensor * ref, ggml_tensor * got, float tolerance, const char * name) {
    check(ggml_nelements(ref) == ggml_nelements(got), "comparison shape");
    std::vector<float> a(ggml_nelements(ref)), b(a.size());
    ggml_backend_tensor_get(ref, a.data(), 0, a.size()*4);
    ggml_backend_tensor_get(got, b.data(), 0, b.size()*4);
    float error = 0; size_t changed = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        check(std::isfinite(a[i]) && std::isfinite(b[i]), "finite outputs");
        error = std::max(error, std::abs(a[i] - b[i]));
        changed += std::memcmp(&a[i], &b[i], 4) != 0;
    }
    std::printf("%s: max_abs=%g changed=%zu/%zu\n", name, error, changed, a.size());
    check(error <= tolerance, name);
}
static void hc(ggml_backend_t backend) {
    constexpr int E = 2560, H = 4, N = E*H;
    ggml_context_ptr ctx(ggml_init({16*1024*1024, nullptr, true}));
    auto x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, E, H, 1);
    auto w = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, N);
    auto wi = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, N, H);
    auto mmw = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, N, 32);
    auto nr = ggml_z6_hc_norm(ctx.get(), x, w, 1e-6f);
    auto ir = ggml_mul_mat(ctx.get(), wi, nr);
    auto packed = ggml_z6_hc_norm_inject(ctx.get(), x, w, wi, 1e-6f);
    auto nf = ggml_view_2d(ctx.get(), packed, N, 1, N*4, 0);
    auto inj = ggml_view_2d(ctx.get(), packed, H, 1, H*4, N*4);
    auto mr = ggml_mul_mat(ctx.get(), mmw, nr);
    auto mf = ggml_mul_mat(ctx.get(), mmw, nf);
    auto g = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(g, ir); ggml_build_forward_expand(g, mr);
    ggml_build_forward_expand(g, inj); ggml_build_forward_expand(g, mf);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    check(bool(buffer), "HC allocation");
    fill(w, 0.8f); fill(wi, 0.01f); fill(mmw, 0.01f);
    for (int epoch = 0; epoch < 3; ++epoch) {
        fill(x, 1.5f, epoch);
        check(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS, "HC compute");
        compare(nr, nf, 3e-6f, "HC norm");
        compare(ir, inj, 3e-5f, "HC inject");
        compare(mr, mf, 1e-4f, "HC Q8 consumer/fresh epoch");
    }
}
static void ssm(ggml_backend_t backend) {
    constexpr int D = 128, K = D*4, N = 4*K;
    ggml_context_ptr ctx(ggml_init({8*1024*1024, nullptr, true}));
    auto x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, N, 1);
    auto silu = ggml_silu(ctx.get(), x);
    auto q = ggml_l2_norm(ctx.get(), ggml_view_2d(ctx.get(), silu, D, K/D, D*4, 0), 1e-6f);
    auto k = ggml_l2_norm(ctx.get(), ggml_view_2d(ctx.get(), silu, D, K/D, D*4, K*4), 1e-6f);
    auto v = ggml_view_1d(ctx.get(), silu, N - 2*K, 2*K*4);
    auto ref = ggml_concat(ctx.get(), ggml_concat(ctx.get(), ggml_reshape_1d(ctx.get(), q, K),
            ggml_reshape_1d(ctx.get(), k, K), 0), v, 0);
    auto fused = ggml_z6_ssm_silu_norm(ctx.get(), x, K, D, 1e-6f);
    auto g = ggml_new_graph(ctx.get()); ggml_build_forward_expand(g, ref); ggml_build_forward_expand(g, fused);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    check(bool(buffer), "SSM allocation");
    for (int epoch = 0; epoch < 4; ++epoch) {
        fill(x, epoch == 0 ? 0.0f : (epoch == 1 ? 1e-8f : 8.0f), epoch);
        check(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS, "SSM compute");
        compare(ref, fused, 2e-6f, "SSM SiLU/L2 zero/tiny/large/fresh");
    }
}
static void down(ggml_backend_t cpu, ggml_type type = GGML_TYPE_Q4_K) {
    constexpr int K = 512, E = 256, NK = 3, NE = 4;
    ggml_context_ptr ctx(ggml_init({8*1024*1024, nullptr, true}));
    auto a = ggml_new_tensor_3d(ctx.get(), type, K, E, NE);
    auto x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, K, NK, 1);
    auto ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, NK, 1);
    auto w = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, NK, 1);
    auto ref = ggml_mul_multi_add_id(ctx.get(), ggml_mul_mat_id(ctx.get(), a, x, ids), w, ids);
    auto fused = ggml_z6_moe_down_sum(ctx.get(), a, x, ids, w);
    auto g = ggml_new_graph(ctx.get()); ggml_build_forward_expand(g, ref); ggml_build_forward_expand(g, fused);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu));
    check(bool(buffer), "down allocation");
    fill(a, 0.125f);
    for (int epoch = 0; epoch < 3; ++epoch) {
        fill(x, 0.5f, epoch);
        int32_t id[NK] = {1, epoch == 1 ? 1 : -1, 3};
        float weights[NK] = {0.125f, epoch == 1 ? -0.25f : std::numeric_limits<float>::quiet_NaN(), 0.5f};
        if (epoch == 2) for (int & i : id) i = -1;
        ggml_backend_tensor_set(ids, id, 0, sizeof(id)); ggml_backend_tensor_set(w, weights, 0, sizeof(weights));
        check(ggml_backend_graph_compute(cpu, g) == GGML_STATUS_SUCCESS, "down compute");
        compare(ref, fused, 1e-5f, "down/sum excluded/duplicate/all excluded");
    }
}
static void batch(ggml_backend_t cpu, const char * prefix) {
    // Skewed routing crosses the IQK Ny conversion threshold and leaves masked slots.
    constexpr int K = 512, U = 256, NE = 128, NK = 10;
    int32_t cpus[40];
    for (int i = 0; i < 40; ++i) cpus[i] = i;
    ggml_backend_cpu_set_cpu_affinity(cpu, cpus, 40);
    for (int nt : {174, 550, 1024, 1536}) {
        ggml_context_ptr ctx(ggml_init({16*1024*1024, nullptr, true}));
        auto up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, K, U, NE);
        auto gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, K, U, NE);
        auto dw = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q5_K, U, K, NE);
        auto x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, K, 1, nt);
        auto ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, NK, nt);
        auto par = ggml_moe_up_gate(ctx.get(), up, gate, x, ids, GGML_UNARY_OP_SILU);
        auto out = ggml_mul_mat_id(ctx.get(), dw, par, ids);
        auto g = ggml_new_graph(ctx.get()); ggml_build_forward_expand(g, out);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu));
        check(bool(buffer), "batch allocation");
        fill(up, 0.125f); fill(gate, 0.2f); fill(dw, 0.1f); fill(x, 0.5f);
        std::vector<int32_t> id(NK*nt);
        for (int t = 0; t < nt; ++t) for (int j = 0; j < NK; ++j)
            id[t*NK + j] = j < 3 ? j : (j == 9 && t%4 == 0 ? -1 : 3 + (t*7 + j*17)%(NE - 3));
        ggml_backend_tensor_set(ids, id.data(), 0, id.size()*4);
        check(ggml_backend_graph_compute(cpu, g) == GGML_STATUS_SUCCESS, "batch compute");
        for (auto t : {par, out}) {
            std::vector<float> data(ggml_nelements(t));
            ggml_backend_tensor_get(t, data.data(), 0, data.size()*4);
            for (float v : data) check(std::isfinite(v), "batch finite");
            char path[4096];
            std::snprintf(path, sizeof(path), "%s-%d-%s.bin", prefix, nt, t == par ? "up" : "down");
            FILE * f = std::fopen(path, "wb"); check(f != nullptr, "batch output open");
            check(std::fwrite(data.data(), 4, data.size(), f) == data.size(), "batch output write");
            std::fclose(f);
        }
        std::printf("OK batch fixture %d\n", nt);
    }
}
int main(int argc, char ** argv) {
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    check(bool(cpu), "CPU backend");
    ggml_backend_cpu_set_n_threads(cpu.get(), 40);
    if (argc == 3 && std::strcmp(argv[1], "--batch") == 0) { batch(cpu.get(), argv[2]); return 0; }
    hc(cpu.get()); ssm(cpu.get()); down(cpu.get()); down(cpu.get(), GGML_TYPE_F32);
    ggml_backend_ptr gpu(ggml_backend_reg_init_backend_from_str("CUDA0"));
    check(bool(gpu), "CUDA backend");
    hc(gpu.get()); ssm(gpu.get());
    std::puts("OK Z6 software numerical fixtures");
}
