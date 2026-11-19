// Z6 (GGML_Z6_DSA_IDX_DIRETTO): la lista delle celle scelte dalla DSA di GLM fatta direttamente dal top-k deve essere,
// byte per byte, quella che esce oggi dalla maschera sparsa (ggml_indexer_mask + ggml_mask_to_index), e l'attenzione
// che la usa con la maschera causale deve dare gli stessi byte di quella che la usa con la maschera sparsa.
// Gira sulla GPU (CUDA0), senza modello:
//
//   test-z6-dsa-indice [--solo-indice]
//
// Parte 1, indice: maschera causale e top-k costruiti come in llama_set_inputs e build_glm5next_dsa_top_k (pool di r
// celle, pool incompleti e coda mancante riempiti con la cella 0, pool non ancora visibili). Tre liste a confronto con
// quella calcolata qui sull'host: la strada di oggi, la diretta con le celle una volta sola (op_params[0] = 1) e, come
// controllo, la diretta senza: con una cella ripetuta deve essere DIVERSA, altrimenti la prova non prova niente.
// Parte 2, attenzione: flash attention MLA (K = V = latente a 512, 64 teste) a 1, 2 e 32 query, cioè i due kernel
// che GLM usa con la lista (ggml_cuda_dsa_attn_ext fino a 7 query, il kernel sparso dalle 8 in su).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

int errori = 0;

void verifica(bool ok, const char * cosa, const char * caso) {
    printf("%s %s: %s\n", ok ? "PASS" : "FAIL", caso, cosa);
    if (!ok) ++errori;
}

struct ingressi {
    int64_t n_kv, righe;         // celle della cache, righe della maschera (n_tok arrotondato a GGML_KQ_MASK_PAD)
    int n_tok, n_topk;
    std::vector<ggml_fp16_t> maschera;   // [n_kv, righe]
    std::vector<int32_t> topk;           // [n_topk, n_tok]
    std::vector<int32_t> attesa;         // [n_pad, righe]: la lista che deve uscire
    int64_t n_pad;
    bool ripetute = false;               // almeno una cella visibile compare due volte in una riga del top-k
};

// La cache è in ordine di posizione (cella i = posizione i), usate le prime n_usate celle; la query j sta alla
// posizione n_usate - n_tok + j. n_sel pool per query presi a caso fra tutti quelli della cache, anche futuri.
ingressi costruisci(int64_t n_kv, int64_t n_usate, int n_tok, int n_sel, int r, int incompleti, unsigned seme) {
    ingressi in;
    in.n_kv = n_kv; in.n_tok = n_tok; in.righe = GGML_PAD(n_tok, GGML_KQ_MASK_PAD);
    in.n_topk = n_sel*r + (r - 1);
    in.n_pad = std::min<int64_t>(n_kv, GGML_PAD(in.n_topk, 256));
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), inf = ggml_fp32_to_fp16(-INFINITY);
    in.maschera.assign(n_kv*in.righe, inf);
    in.topk.assign((size_t) in.n_topk*n_tok, 0);
    in.attesa.assign(in.n_pad*in.righe, -1);
    std::mt19937 rng(seme);
    const int64_t n_pool = n_kv / r;
    for (int j = 0; j < n_tok; ++j) {
        const int64_t q = n_usate - n_tok + j;
        for (int64_t i = 0; i <= q; ++i) in.maschera[j*n_kv + i] = zero;
        // n_sel pool distinti
        std::vector<int32_t> pool(n_pool);
        for (int64_t b = 0; b < n_pool; ++b) pool[b] = (int32_t) b;
        for (int s = 0; s < n_sel; ++s) std::swap(pool[s], pool[s + rng() % (n_pool - s)]);
        int32_t * t = in.topk.data() + (size_t) j*in.n_topk;
        for (int s = 0; s < n_sel; ++s) {
            const bool incompleto = s < incompleti;   // llama_set_inputs mette la cella 0 al posto di ogni membro
            for (int m = 0; m < r; ++m) t[s*r + m] = incompleto ? 0 : pool[s]*r + m;
        }
        // coda: il pool incompleto in fondo, [tail_start, q]; i posti senza cella valgono 0
        const int64_t tail_start = (q + 1) / r * r;
        for (int m = 0; m < r - 1; ++m) t[n_sel*r + m] = tail_start + m <= q ? (int32_t) (tail_start + m) : 0;
        // lista attesa: celle del top-k aperte nella maschera causale, una volta sola, per (cella % 32, cella)
        std::vector<int32_t> celle;
        for (int k = 0; k < in.n_topk; ++k) if (in.maschera[j*n_kv + t[k]] == zero) celle.push_back(t[k]);
        std::sort(celle.begin(), celle.end());
        if (std::adjacent_find(celle.begin(), celle.end()) != celle.end()) in.ripetute = true;
        celle.erase(std::unique(celle.begin(), celle.end()), celle.end());
        std::sort(celle.begin(), celle.end(), [](int32_t a, int32_t b) { return a % 32 != b % 32 ? a % 32 < b % 32 : a < b; });
        if ((int64_t) celle.size() > in.n_pad) { fprintf(stderr, "caso mal costruito: %zu celle\n", celle.size()); std::exit(2); }
        std::copy(celle.begin(), celle.end(), in.attesa.begin() + j*in.n_pad);
    }
    return in;
}

struct banco {
    ggml_backend_t gpu, cpu;
    ggml_context_ptr pesi, ctx;
    ggml_backend_buffer_ptr buf;
    ggml_backend_sched_ptr sched;
    ggml_cgraph * g = nullptr;
    std::vector<ggml_tensor *> nodi;
    banco(ggml_backend_t gpu_, ggml_backend_t cpu_) : gpu(gpu_), cpu(cpu_),
        pesi(ggml_init({1024*1024, nullptr, true})), ctx(ggml_init({16*1024*1024, nullptr, true})) {}
    ggml_tensor * nodo(ggml_tensor * t) { nodi.push_back(t); return t; }
    ggml_tensor * uscita(ggml_tensor * t) { ggml_set_output(t); return nodo(t); }
    bool calcola(const std::vector<ggml_tensor *> & uscite) {
        ggml_backend_t backends[] = {gpu, cpu};
        sched.reset(ggml_backend_sched_new(backends, nullptr, 2, 4096, false));
        g = ggml_new_graph_custom(ctx.get(), 4096, false);
        for (auto t : uscite) ggml_build_forward_expand(g, t);
        for (auto t : nodi) ggml_backend_sched_set_tensor_backend(sched.get(), t, gpu);
        return ggml_backend_sched_alloc_graph(sched.get(), g) && ggml_backend_sched_graph_compute(sched.get(), g) == GGML_STATUS_SUCCESS;
    }
};

std::vector<uint8_t> leggi(ggml_tensor * t) {
    std::vector<uint8_t> v(ggml_nbytes(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size());
    return v;
}

void prova_indice(ggml_backend_t gpu, ggml_backend_t cpu, const char * nome, const ingressi & in) {
    banco b(gpu, cpu);
    auto maschera = ggml_new_tensor_2d(b.pesi.get(), GGML_TYPE_F16, in.n_kv, in.righe);
    auto topk     = ggml_new_tensor_2d(b.pesi.get(), GGML_TYPE_I32, in.n_topk, in.n_tok);
    b.buf.reset(ggml_backend_alloc_ctx_tensors(b.pesi.get(), gpu));
    if (!b.buf) { verifica(false, "allocazione sulla GPU", nome); return; }
    ggml_backend_tensor_set(maschera, in.maschera.data(), 0, ggml_nbytes(maschera));
    ggml_backend_tensor_set(topk, in.topk.data(), 0, ggml_nbytes(topk));

    auto sparsa  = b.nodo(ggml_indexer_mask(b.ctx.get(), maschera, topk));
    auto oggi    = b.uscita(ggml_mask_to_index(b.ctx.get(), sparsa, (int) in.n_pad));
    auto diretta = b.uscita(ggml_indexer_mask_to_index(b.ctx.get(), maschera, topk, (int) in.n_pad));
    diretta->op_params[0] = 1;
    auto doppia  = b.uscita(ggml_indexer_mask_to_index(b.ctx.get(), maschera, topk, (int) in.n_pad));
    if (!b.calcola({oggi, diretta, doppia})) { verifica(false, "calcolo del grafo", nome); return; }

    const auto v_oggi = leggi(oggi), v_dir = leggi(diretta), v_dop = leggi(doppia);
    const size_t n = in.attesa.size()*sizeof(int32_t);
    verifica(v_oggi.size() == n && memcmp(v_oggi.data(), in.attesa.data(), n) == 0, "la strada di oggi dà la lista attesa", nome);
    verifica(v_dir.size()  == n && memcmp(v_dir.data(),  in.attesa.data(), n) == 0, "la lista diretta dà la lista attesa", nome);
    verifica(v_oggi == v_dir, "lista diretta identica a quella di oggi", nome);
    if (in.ripetute) verifica(v_dop != v_oggi, "controllo: senza le celle una volta sola la lista è diversa", nome);
}

void prova_attenzione(ggml_backend_t gpu, ggml_backend_t cpu, const char * nome, const ingressi & in, unsigned seme) {
    constexpr int d = 512, teste = 64;
    banco b(gpu, cpu);
    auto maschera = ggml_new_tensor_2d(b.pesi.get(), GGML_TYPE_F16, in.n_kv, in.righe);
    auto topk     = ggml_new_tensor_2d(b.pesi.get(), GGML_TYPE_I32, in.n_topk, in.n_tok);
    auto k        = ggml_new_tensor_2d(b.pesi.get(), GGML_TYPE_F16, d, in.n_kv);
    auto q        = ggml_new_tensor_3d(b.pesi.get(), GGML_TYPE_F32, d, in.n_tok, teste);
    b.buf.reset(ggml_backend_alloc_ctx_tensors(b.pesi.get(), gpu));
    if (!b.buf) { verifica(false, "allocazione sulla GPU", nome); return; }
    std::mt19937 rng(seme);
    std::normal_distribution<float> normale(0.0f, 1.0f);
    std::vector<ggml_fp16_t> kd((size_t) d*in.n_kv);
    for (auto & x : kd) x = ggml_fp32_to_fp16(normale(rng));
    std::vector<float> qd((size_t) d*in.n_tok*teste);
    for (auto & x : qd) x = 0.2f*normale(rng);
    ggml_backend_tensor_set(maschera, in.maschera.data(), 0, ggml_nbytes(maschera));
    ggml_backend_tensor_set(topk, in.topk.data(), 0, ggml_nbytes(topk));
    ggml_backend_tensor_set(k, kd.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(q, qd.data(), 0, ggml_nbytes(q));
    const float scala = 1.0f/sqrtf(256.0f);

    // oggi: maschera sparsa e lista letta dalla maschera
    auto sparsa   = b.nodo(ggml_indexer_mask(b.ctx.get(), maschera, topk));
    auto idx_oggi = b.nodo(ggml_mask_to_index(b.ctx.get(), sparsa, (int) in.n_pad));
    auto fa_oggi  = b.uscita(ggml_flash_attn_ext(b.ctx.get(), q, k, k, sparsa, scala, 0.0f, 0.0f));
    fa_oggi->src[5] = idx_oggi;
    // leva: maschera causale e lista diretta
    auto idx_dir  = b.nodo(ggml_indexer_mask_to_index(b.ctx.get(), maschera, topk, (int) in.n_pad));
    idx_dir->op_params[0] = 1;
    auto fa_dir   = b.uscita(ggml_flash_attn_ext(b.ctx.get(), q, k, k, maschera, scala, 0.0f, 0.0f));
    fa_dir->src[5] = idx_dir;
    fa_dir->op_params[5] = 1;
    // controllo: maschera causale senza lista, cioè attenzione densa: deve essere diversa dalla sparsa
    auto fa_densa = b.uscita(ggml_flash_attn_ext(b.ctx.get(), q, k, k, maschera, scala, 0.0f, 0.0f));
    if (!b.calcola({fa_oggi, fa_dir, fa_densa})) { verifica(false, "calcolo del grafo", nome); return; }

    const auto v_oggi = leggi(fa_oggi), v_dir = leggi(fa_dir), v_densa = leggi(fa_densa);
    bool finiti = true;
    for (size_t i = 0; i + 4 <= v_oggi.size(); i += 4) { float f; memcpy(&f, v_oggi.data() + i, 4); finiti &= std::isfinite(f); }
    verifica(finiti, "uscita di oggi finita", nome);
    verifica(v_oggi == v_dir, "attenzione con maschera causale e lista diretta identica a quella di oggi", nome);
    verifica(v_oggi != v_densa, "controllo: l'attenzione densa è diversa dalla sparsa", nome);
}

}  // namespace

int main(int argc, char ** argv) {
    const bool solo_indice = argc > 1 && strcmp(argv[1], "--solo-indice") == 0;
    ggml_backend_ptr gpu(ggml_backend_reg_init_backend_from_str("CUDA0"));
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    if (!gpu || !cpu) { fprintf(stderr, "FAIL backend CUDA0 o CPU non disponibile\n"); return 1; }
    ggml_backend_cpu_set_n_threads(cpu.get(), 1);

    // n_kv, celle usate, query, pool scelti, r, pool incompleti
    struct { const char * nome; int64_t n_kv, n_usate; int n_tok, n_sel, r, incompleti; } casi[] = {
        { "111K, 1 query, coda piena",        111360, 111291,   1, 512, 4,  0 },   // q + 1 = 111291: coda di 3 celle
        { "111K, 1 query, coda vuota",        111360, 111292,   1, 512, 4,  0 },   // q + 1 multiplo di 4: tre volte la cella 0
        { "111K, 1 query, 5 pool incompleti", 111360, 111290,   1, 512, 4,  5 },
        { "131K piena, 1 query",              131072, 131072,   1, 512, 4,  0 },
        { "soglia della lista, 2560 celle",     2560,   2400,   1, 512, 4,  0 },
        { "8K, 2 query (verifica MTP)",         8192,   8001,   2, 512, 4,  1 },
        { "48K, 7 query",                      49152,  48007,   7, 512, 4,  2 },
        { "111K, 100 query (prefill)",        111360, 111300, 100, 512, 4,  3 },
        { "16K, tutti i pool incompleti",      16384,  16001,   1, 512, 4, 512 },
    };
    unsigned seme = 20261113;
    for (const auto & c : casi) {
        prova_indice(gpu.get(), cpu.get(), c.nome, costruisci(c.n_kv, c.n_usate, c.n_tok, c.n_sel, c.r, c.incompleti, seme++));
    }
    if (!solo_indice) {
        struct { const char * nome; int64_t n_kv, n_usate; int n_tok, incompleti; } fa[] = {
            { "FA 16K, 1 query",   16384, 16002,  1, 0 },
            { "FA 16K, 1 query, coda vuota e pool incompleti", 16384, 16004, 1, 4 },
            { "FA 16K, 2 query",   16384, 16001,  2, 1 },
            { "FA 16K, 32 query",  16384, 16040, 32, 2 },
            { "FA 4K, 32 query",    4096,  4000, 32, 0 },
        };
        for (const auto & c : fa) {
            prova_attenzione(gpu.get(), cpu.get(), c.nome, costruisci(c.n_kv, c.n_usate, c.n_tok, 512, 4, c.incompleti, seme), seme + 1000);
            ++seme;
        }
    }
    printf("%s: %d errori\n", errori ? "FALLITO" : "TUTTO IDENTICO", errori);
    return errori ? 1 : 0;
}
