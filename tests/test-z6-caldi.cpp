#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

static void partition_maps(ggml_backend_t backend, int nt) {
    ggml_context_ptr ctx(ggml_init({1024*1024, nullptr, true}));
    constexpr int ne = 16, nk = 10;
    auto hot = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, ne);
    auto cold = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, ne);
    auto ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, nk, nt);
    auto flat = ggml_reshape_1d(ctx.get(), ids, nk*nt);
    auto hi = ggml_get_rows(ctx.get(), hot, flat);
    auto ci = ggml_get_rows(ctx.get(), cold, flat);
    auto g = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(g, hi); ggml_build_forward_expand(g, ci);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    check(bool(buffer), "buffer map");
    int32_t hm[ne], cm[ne]; int slot = 0;
    for (int e = 0; e < ne; ++e) { hm[e] = e%3 == 0 ? slot++ : -1; cm[e] = e%3 == 0 ? -1 : e; }
    std::vector<int32_t> chosen(nk*nt), h(nk*nt), c(nk*nt);
    for (int i = 0; i < nk*nt; ++i) chosen[i] = i%ne;
    ggml_backend_tensor_set(hot, hm, 0, sizeof(hm)); ggml_backend_tensor_set(cold, cm, 0, sizeof(cm));
    ggml_backend_tensor_set(ids, chosen.data(), 0, chosen.size()*4);
    check(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS, "I32 map compute");
    ggml_backend_tensor_get(hi, h.data(), 0, h.size()*4); ggml_backend_tensor_get(ci, c.data(), 0, c.size()*4);
    for (size_t i = 0; i < chosen.size(); ++i)
        check(h[i] == hm[chosen[i]] && c[i] == cm[chosen[i]] && ((h[i] < 0) != (c[i] < 0)),
                "exactly one valid map entry, preserve I32 minus one");
}

static void masked_sum(ggml_backend_t backend, int nt, int mode, bool scales) {
    ggml_context_ptr ctx(ggml_init({8*1024*1024, nullptr, true}));
    constexpr int nx = 256, nk = 10;
    auto x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, nx, nk, nt);
    auto w = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, nk, nt);
    auto ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, nk, nt);
    auto orig = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, nk, nt);
    auto s = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, nk);
    auto y = ggml_mul_multi_add_id(ctx.get(), x, w, ids);
    if (scales) { y->src[2] = s; y->src[3] = orig; }
    auto g = ggml_new_graph(ctx.get()); ggml_build_forward_expand(g, y);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    check(bool(buffer), "buffer sum");
    std::vector<float> xd(nx*nk*nt), wd(nk*nt), sd(nk), expected(nx*nt, 0);
    std::vector<int32_t> id(nk*nt), original(nk*nt);
    for (int j = 0; j < nk; ++j) sd[j] = 0.5f + j*0.125f;
    for (int t = 0; t < nt; ++t) for (int j = 0; j < nk; ++j) {
        const bool valid = mode == 0 || (mode == 1 && (j + t)%3 == 0);
        id[t*nk+j] = valid ? j : -1;
        original[t*nk+j] = valid ? j : -1; // must not index scales[-1]
        wd[t*nk+j] = valid ? 0.125f : std::numeric_limits<float>::quiet_NaN();
        for (int k = 0; k < nx; ++k) {
            float value = (k%11 - 5)*0.25f + j*0.125f;
            xd[(t*nk+j)*nx+k] = valid ? value : std::numeric_limits<float>::quiet_NaN();
            if (valid) expected[t*nx+k] += value*0.125f*(scales ? sd[j] : 1);
        }
    }
    ggml_backend_tensor_set(x, xd.data(), 0, xd.size()*4);
    ggml_backend_tensor_set(w, wd.data(), 0, wd.size()*4);
    ggml_backend_tensor_set(ids, id.data(), 0, id.size()*4);
    ggml_backend_tensor_set(orig, original.data(), 0, original.size()*4);
    ggml_backend_tensor_set(s, sd.data(), 0, sd.size()*4);
    check(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS, "masked compute");
    std::vector<float> out(expected.size()); ggml_backend_tensor_get(y, out.data(), 0, out.size()*4);
    for (size_t i = 0; i < out.size(); ++i) check(std::isfinite(out[i]) && std::abs(out[i]-expected[i]) < 1e-6f, "masked reference/NaN");
}

// Mimics the token-wise hot path: each CUDA up/gate has one token, each down
// consumes [ff,n_used,1]. For ten experts, down must be adjacent to up/gate:
// the ordinary standalone down fast path only accepts <=8 experts.
// Test that invariant for the real hot path, plus standalone down with eight.
static void token_moe(ggml_backend_t backend, int nt, int mode, bool merged, bool fused_down,
        int nx = 256, int ff = 256, ggml_type up_type = GGML_TYPE_Q4_K) {
    ggml_context_ptr ctx(ggml_init({16*1024*1024, nullptr, true}));
    constexpr int ne = 16;
    const int nk = fused_down ? 10 : 8;
    auto up = ggml_new_tensor_3d(ctx.get(), up_type, nx, merged ? 2*ff : ff, ne);
    auto gate = merged ? nullptr : ggml_new_tensor_3d(ctx.get(), up_type, nx, ff, ne);
    auto down_w = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q5_1, ff, nx, ne);
    auto x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, nx, 1, nt);
    auto ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, nk, nt);
    auto weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, nk, nt);
    std::vector<ggml_tensor *> pars, downs;
    ggml_tensor * result = nullptr;
    auto g = ggml_new_graph(ctx.get());
    for (int t = 0; t < nt; ++t) {
        auto xi = ggml_view_3d(ctx.get(), x, nx, 1, 1, x->nb[1], x->nb[2], t*x->nb[2]);
        auto ii = ggml_view_2d(ctx.get(), ids, nk, 1, ids->nb[1], t*ids->nb[1]);
        auto wi = ggml_view_3d(ctx.get(), weights, 1, nk, 1, weights->nb[1], weights->nb[2], t*weights->nb[2]);
        auto par = ggml_moe_up_gate(ctx.get(), up, gate, xi, ii, GGML_UNARY_OP_SILU);
        auto down = ggml_mul_mat_id(ctx.get(), down_w, par, ii);
        if (!fused_down) {
            // Inserting a copy prevents the CUDA adjacency fusion.
            down->src[1] = ggml_cont(ctx.get(), par);
        }
        auto sum = ggml_mul_multi_add_id(ctx.get(), down, wi, ii);
        check(par->src[2]->ne[1] == 1 && par->src[2]->ne[2] == 1 && down->src[1]->ne[2] == 1,
                "hot matmul token dimensions");
        pars.push_back(par); downs.push_back(down);
        result = result ? ggml_concat(ctx.get(), result, sum, 1) : sum;
    }
    ggml_build_forward_expand(g, result);
    if (fused_down) for (int t = 0; t < nt; ++t) {
        auto it = std::find(g->nodes, g->nodes + g->n_nodes, pars[t]);
        check(it != g->nodes + g->n_nodes && it + 1 != g->nodes + g->n_nodes && it[1] == downs[t],
                "ten-expert down must stay adjacent to up/gate for MMVQ fusion");
    }
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    check(bool(buffer), "buffer moe");
    ggml_backend_buffer_clear(buffer.get(), 0xff); // stale NaNs in all outputs
    auto quantize = [](ggml_tensor * tensor) {
        std::vector<float> data(ggml_nelements(tensor));
        for (size_t i = 0; i < data.size(); ++i) data[i] = std::sin(float(i%1009))*0.01f;
        std::vector<uint8_t> q(ggml_nbytes(tensor));
        ggml_quantize_chunk(tensor->type, data.data(), q.data(), 0, ggml_nrows(tensor), tensor->ne[0], nullptr, nullptr);
        ggml_backend_tensor_set(tensor, q.data(), 0, q.size());
    };
    quantize(up); if (gate) quantize(gate); quantize(down_w);
    std::vector<float> xd(nx*nt, 0.125f), wd(nk*nt, 0.1f);
    std::vector<int32_t> id(nk*nt);
    for (int t = 0; t < nt; ++t) for (int j = 0; j < nk; ++j)
        id[t*nk+j] = mode == 0 || (mode == 1 && (j+t)%3 == 0) ? j : -1;
    ggml_backend_tensor_set(x, xd.data(), 0, xd.size()*4);
    ggml_backend_tensor_set(ids, id.data(), 0, id.size()*4);
    ggml_backend_tensor_set(weights, wd.data(), 0, wd.size()*4);
    check(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS, "moe compute");
    std::vector<float> out(nx*nt); ggml_backend_tensor_get(result, out.data(), 0, out.size()*4);
    for (float f : out) check(std::isfinite(f) && (mode != 2 || f == 0), "moe finite/all excluded zero");
    for (int t = 0; t < nt; ++t) {
        std::vector<float> pd(ff*nk), dd(nx*nk);
        ggml_backend_tensor_get(pars[t], pd.data(), 0, pd.size()*4);
        ggml_backend_tensor_get(downs[t], dd.data(), 0, dd.size()*4);
        for (int j = 0; j < nk; ++j) if (id[t*nk+j] < 0) {
            for (int k = 0; k < ff; ++k) check(pd[j*ff+k] == 0, "excluded up/gate must be zero");
            for (int k = 0; k < nx; ++k) check(dd[j*nx+k] == 0, "excluded down must be zero");
        }
    }
}

int main(int argc, char ** argv) {
    ggml_backend_ptr backend(ggml_backend_reg_init_backend_from_str(argc > 1 ? argv[1] : "CPU"));
    check(bool(backend), "backend init");
    int count = 0;
    for (int nt : {1,2,8,32}) { partition_maps(backend.get(), nt); ++count; }
    for (int nt : {1,2,8,32}) for (int mode : {0,1,2}) {
        for (bool scales : {false,true}) { masked_sum(backend.get(), nt, mode, scales); ++count; }
        for (bool merged : {false,true}) for (bool fused : {false,true}) {
            token_moe(backend.get(), nt, mode, merged, fused); ++count;
        }
    }
    for (auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K}) {
        token_moe(backend.get(), 1, 1, false, true, 2560, 640, type); ++count;
    }
    printf("OK %s: %d casi; batch 1,2,8,32; tutti/misti/nessuno; NaN, scale, up-gate uniti/separati, down fuso/separato\n",
            ggml_backend_name(backend.get()), count);
}
