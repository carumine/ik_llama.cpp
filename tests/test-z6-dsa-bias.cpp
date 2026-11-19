// Z6 (GGML_Z6_DSA_BIAS_F16): il bias dei pool della DSA di GLM vale solo 0 e -inf. Dato in F16 e convertito in F32 a
// blocchi di query, la somma con i punteggi deve dare gli stessi byte di quella con il bias in F32, e il top-k le
// stesse celle. Gira sulla GPU (CUDA0), senza modello:
//
//   test-z6-dsa-bias
//
// Punteggi come in build_glm5next_dsa_top_k: [n_pool, tc] F32, non negativi a meno del segno dei pesi, con zeri
// positivi e negativi (0 + (-0) deve restare quello della strada di oggi) e valori uguali fra pool.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int errori = 0;

void verifica(bool ok, const char * cosa, const char * caso) {
    printf("%s %s: %s\n", ok ? "PASS" : "FAIL", caso, cosa);
    if (!ok) ++errori;
}

std::vector<uint8_t> leggi(ggml_tensor * t) {
    std::vector<uint8_t> v(ggml_nbytes(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size());
    return v;
}

// n_tok query in blocchi da tc, come il ciclo dei punteggi; visibili: quanti pool vede la prima query (uno in più ogni 4)
void prova(ggml_backend_t gpu, ggml_backend_t cpu, const char * nome, int64_t n_pool, int n_tok, int tc, int64_t visibili, int n_sel, unsigned seme) {
    ggml_context_ptr pesi(ggml_init({1024*1024, nullptr, true})), ctx(ggml_init({16*1024*1024, nullptr, true}));
    auto punteggi = ggml_new_tensor_2d(pesi.get(), GGML_TYPE_F32, n_pool, n_tok);
    auto bias32   = ggml_new_tensor_2d(pesi.get(), GGML_TYPE_F32, n_pool, n_tok);
    auto bias16   = ggml_new_tensor_2d(pesi.get(), GGML_TYPE_F16, n_pool, n_tok);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(pesi.get(), gpu));
    if (!buf) { verifica(false, "allocazione sulla GPU", nome); return; }

    std::mt19937 rng(seme);
    std::normal_distribution<float> normale(0.0f, 1.0f);
    std::vector<float> p((size_t) n_pool*n_tok), b32(p.size());
    std::vector<ggml_fp16_t> b16(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        const unsigned u = rng() % 16;
        p[i] = u == 0 ? 0.0f : u == 1 ? -0.0f : u == 2 ? 0.5f : std::fabs(normale(rng));   // zeri dei due segni e pari merito
    }
    for (int j = 0; j < n_tok; ++j) {
        for (int64_t pool = 0; pool < n_pool; ++pool) {
            const bool ok = pool < visibili + j/4;
            b32[(size_t) j*n_pool + pool] = ok ? 0.0f : -INFINITY;
            b16[(size_t) j*n_pool + pool] = ggml_fp32_to_fp16(ok ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(punteggi, p.data(), 0, ggml_nbytes(punteggi));
    ggml_backend_tensor_set(bias32, b32.data(), 0, ggml_nbytes(bias32));
    ggml_backend_tensor_set(bias16, b16.data(), 0, ggml_nbytes(bias16));

    std::vector<ggml_tensor *> nodi, uscite;
    auto nodo = [&](ggml_tensor * t) { nodi.push_back(t); return t; };
    std::vector<ggml_tensor *> s32, s16, k32, k16;
    for (int c0 = 0; c0 < n_tok; c0 += tc) {
        const int n = std::min(tc, n_tok - c0);
        auto sc  = ggml_view_2d(ctx.get(), punteggi, n_pool, n, punteggi->nb[1], (size_t) c0*punteggi->nb[1]);
        // oggi: ggml_cont della vista F32; leva: ggml_cast della vista F16
        auto b_o = nodo(ggml_cont(ctx.get(), ggml_view_2d(ctx.get(), bias32, n_pool, n, bias32->nb[1], (size_t) c0*bias32->nb[1])));
        auto b_l = nodo(ggml_cast(ctx.get(), ggml_view_2d(ctx.get(), bias16, n_pool, n, bias16->nb[1], (size_t) c0*bias16->nb[1]), GGML_TYPE_F32));
        auto a_o = nodo(ggml_add(ctx.get(), nodo(ggml_cont(ctx.get(), sc)), b_o));
        auto a_l = nodo(ggml_add(ctx.get(), nodo(ggml_cont(ctx.get(), sc)), b_l));
        ggml_set_output(a_o); ggml_set_output(a_l);
        s32.push_back(a_o); s16.push_back(a_l);
        auto t_o = nodo(ggml_cont(ctx.get(), nodo(ggml_top_k(ctx.get(), a_o, n_sel))));
        auto t_l = nodo(ggml_cont(ctx.get(), nodo(ggml_top_k(ctx.get(), a_l, n_sel))));
        ggml_set_output(t_o); ggml_set_output(t_l);
        k32.push_back(t_o); k16.push_back(t_l);
    }
    ggml_backend_t backends[] = {gpu, cpu};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 8192, false));
    auto g = ggml_new_graph_custom(ctx.get(), 8192, false);
    for (auto v : {&s32, &s16, &k32, &k16}) for (auto t : *v) ggml_build_forward_expand(g, t);
    for (auto t : nodi) ggml_backend_sched_set_tensor_backend(sched.get(), t, gpu);
    if (!ggml_backend_sched_alloc_graph(sched.get(), g) || ggml_backend_sched_graph_compute(sched.get(), g) != GGML_STATUS_SUCCESS) {
        verifica(false, "calcolo del grafo", nome);
        return;
    }
    bool somme = true, scelte = true, inf_visto = false;
    for (size_t i = 0; i < s32.size(); ++i) {
        const auto a = leggi(s32[i]), b = leggi(s16[i]);
        somme &= a == b;
        for (size_t x = 0; x + 4 <= a.size(); x += 4) { float f; memcpy(&f, a.data() + x, 4); inf_visto |= std::isinf(f); }
        scelte &= leggi(k32[i]) == leggi(k16[i]);
    }
    verifica(somme, "punteggi + bias F16 identici a punteggi + bias F32", nome);
    verifica(scelte, "top-k identico", nome);
    verifica(inf_visto, "controllo: ci sono pool mascherati", nome);
}

}  // namespace

int main() {
    ggml_backend_ptr gpu(ggml_backend_reg_init_backend_from_str("CUDA0"));
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    if (!gpu || !cpu) { fprintf(stderr, "FAIL backend CUDA0 o CPU non disponibile\n"); return 1; }
    ggml_backend_cpu_set_n_threads(cpu.get(), 1);
    prova(gpu.get(), cpu.get(), "65536 pool, 64 query a blocchi di 32", 65536, 64, 32, 55000, 512, 1);
    prova(gpu.get(), cpu.get(), "65536 pool, 48 query a blocchi di 16", 65536, 48, 16, 60000, 512, 2);
    prova(gpu.get(), cpu.get(), "27840 pool, 1 query",                  27840,  1, 32, 27000, 512, 3);
    prova(gpu.get(), cpu.get(), "640 pool, meno visibili del top-k",      640, 33, 32,   100, 512, 4);
    printf("%s: %d errori\n", errori ? "FALLITO" : "TUTTO IDENTICO", errori);
    return errori ? 1 : 0;
}
