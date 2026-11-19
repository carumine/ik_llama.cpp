// Z6 (GGML_Z6_DSA_CAUSALE_COMPATTA): la lista delle celle scelte dalla DSA di GLM fatta dal top-k e da due vettori
// di posizioni (ggml_z6_indexer_pos_to_index) deve essere, byte per byte, quella che esce dalla maschera causale
// [n_kv, n_tokens], e l'attenzione che la usa senza maschera deve dare gli stessi byte di quella di oggi (maschera
// sparsa + lista letta dalla maschera). Gira sulla GPU (CUDA0), senza modello:
//
//   test-z6-dsa-causale [--solo-indice]
//
// Parte 1, indice. Maschera e top-k come in llama_set_inputs e build_glm5next_dsa_top_k: pool di r celle, pool
// incompleti e coda mancante riempiti con la cella 0, pool non ancora visibili, celle vuote in fondo. Tre casi in più
// rispetto a test-z6-dsa-indice: celle non in ordine di posizione (dopo un defrag), una cella vuota nel mezzo, righe
// di riempimento. A confronto con la lista calcolata qui sull'host:
//   - la strada di oggi (ggml_indexer_mask + ggml_mask_to_index);
//   - la lista dalle posizioni sulla GPU;
//   - la lista dalle posizioni sulla CPU, che ordina per cella come fa già l'operatore CPU: stesso insieme di celle.
// Controllo: con la posizione delle query spostata di una cella la lista deve essere DIVERSA.
// Parte 2, attenzione. Flash attention MLA (K = V = latente a 512, 64 teste) a 1, 2, 7, 32 e 100 query: fino a 7
// query lavora ggml_cuda_dsa_attn_ext (senza maschera con op_params[5] = 2), dalle 8 il kernel sparso.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
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
    std::vector<int32_t> pos_celle;      // [n_kv]: posizione della cella, INT32_MAX se vuota
    std::vector<int32_t> pos_query;      // [righe]: posizione della query, -1 nelle righe di riempimento
    std::vector<int32_t> topk;           // [n_topk, n_tok]
    std::vector<int32_t> attesa;         // [n_pad, righe]: la lista che deve uscire, per (cella % 32, cella)
    std::vector<int32_t> attesa_cpu;     // la stessa per cella crescente
    int64_t n_pad;
    bool coda = false;                   // almeno una query ha la propria cella nel top-k (coda non vuota)
};

// n_usate posizioni occupate; la query j sta alla posizione n_usate - n_tok + j. Con disordine la posizione p sta
// nella cella perm[p] (come dopo un defrag), altrimenti nella cella p. buco: una posizione passata senza cella
// (cella vuota nel mezzo). n_sel pool per query presi a caso fra tutti quelli della cache, anche futuri.
ingressi costruisci(int64_t n_kv, int64_t n_usate, int n_tok, int n_sel, int r, int incompleti, bool disordine, bool buco, unsigned seme) {
    ingressi in;
    in.n_kv = n_kv; in.n_tok = n_tok; in.righe = GGML_PAD(n_tok, GGML_KQ_MASK_PAD);
    in.n_topk = n_sel*r + (r - 1);
    in.n_pad = std::min<int64_t>(n_kv, GGML_PAD(in.n_topk, 256));
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), inf = ggml_fp32_to_fp16(-INFINITY);
    std::mt19937 rng(seme);

    // posizione -> cella
    std::vector<int32_t> cella_di(n_kv);
    std::iota(cella_di.begin(), cella_di.end(), 0);
    if (disordine) std::shuffle(cella_di.begin(), cella_di.end(), rng);
    in.pos_celle.assign(n_kv, INT32_MAX);
    const int64_t p_buco = buco ? n_usate/2 : -1;
    for (int64_t p = 0; p < n_usate; ++p) if (p != p_buco) in.pos_celle[cella_di[p]] = (int32_t) p;

    in.maschera.assign(n_kv*in.righe, inf);
    in.pos_query.assign(in.righe, -1);
    in.topk.assign((size_t) in.n_topk*n_tok, 0);
    in.attesa.assign(in.n_pad*in.righe, -1);
    in.attesa_cpu.assign(in.n_pad*in.righe, -1);
    const int64_t n_pool = n_kv / r;
    for (int j = 0; j < n_tok; ++j) {
        const int64_t q = n_usate - n_tok + j;
        in.pos_query[j] = (int32_t) q;
        // la maschera come la riempie llama_set_inputs: aperta se la cella è della sequenza e non è dopo la query
        for (int64_t i = 0; i < n_kv; ++i) if (in.pos_celle[i] <= q) in.maschera[j*n_kv + i] = zero;
        std::vector<int32_t> pool(n_pool);
        for (int64_t b = 0; b < n_pool; ++b) pool[b] = (int32_t) b;
        for (int s = 0; s < n_sel; ++s) std::swap(pool[s], pool[s + rng() % (n_pool - s)]);
        int32_t * t = in.topk.data() + (size_t) j*in.n_topk;
        for (int s = 0; s < n_sel; ++s) {
            const bool incompleto = s < incompleti;   // llama_set_inputs mette la cella 0 al posto di ogni membro
            for (int m = 0; m < r; ++m) t[s*r + m] = incompleto ? 0 : cella_di[pool[s]*r + m];
        }
        // coda: il pool incompleto in fondo, [tail_start, q]; i posti senza cella valgono 0
        const int64_t tail_start = (q + 1) / r * r;
        for (int m = 0; m < r - 1; ++m) t[n_sel*r + m] = tail_start + m <= q ? cella_di[tail_start + m] : 0;
        if (tail_start <= q) in.coda = true;
        std::vector<int32_t> celle;
        for (int k = 0; k < in.n_topk; ++k) if (in.maschera[j*n_kv + t[k]] == zero) celle.push_back(t[k]);
        std::sort(celle.begin(), celle.end());
        celle.erase(std::unique(celle.begin(), celle.end()), celle.end());
        if ((int64_t) celle.size() > in.n_pad) { fprintf(stderr, "caso mal costruito: %zu celle\n", celle.size()); std::exit(2); }
        std::copy(celle.begin(), celle.end(), in.attesa_cpu.begin() + j*in.n_pad);
        std::sort(celle.begin(), celle.end(), [](int32_t a, int32_t b) { return a % 32 != b % 32 ? a % 32 < b % 32 : a < b; });
        std::copy(celle.begin(), celle.end(), in.attesa.begin() + j*in.n_pad);
    }
    return in;
}

struct banco {
    ggml_backend_t gpu, cpu;
    ggml_context_ptr pesi, pesi_cpu, ctx;
    ggml_backend_buffer_ptr buf, buf_cpu;
    ggml_backend_sched_ptr sched;
    ggml_cgraph * g = nullptr;
    std::vector<ggml_tensor *> nodi, nodi_cpu;
    banco(ggml_backend_t gpu_, ggml_backend_t cpu_) : gpu(gpu_), cpu(cpu_),
        pesi(ggml_init({1024*1024, nullptr, true})), pesi_cpu(ggml_init({1024*1024, nullptr, true})),
        ctx(ggml_init({16*1024*1024, nullptr, true})) {}
    ggml_tensor * nodo(ggml_tensor * t) { nodi.push_back(t); return t; }
    ggml_tensor * uscita(ggml_tensor * t) { ggml_set_output(t); return nodo(t); }
    ggml_tensor * uscita_cpu(ggml_tensor * t) { ggml_set_output(t); nodi_cpu.push_back(t); return t; }
    bool calcola(const std::vector<ggml_tensor *> & uscite) {
        ggml_backend_t backends[] = {gpu, cpu};
        sched.reset(ggml_backend_sched_new(backends, nullptr, 2, 4096, false));
        g = ggml_new_graph_custom(ctx.get(), 4096, false);
        for (auto t : uscite) ggml_build_forward_expand(g, t);
        for (auto t : nodi) ggml_backend_sched_set_tensor_backend(sched.get(), t, gpu);
        for (auto t : nodi_cpu) ggml_backend_sched_set_tensor_backend(sched.get(), t, cpu);
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
    auto celle    = ggml_new_tensor_1d(b.pesi.get(), GGML_TYPE_I32, in.n_kv);
    auto query    = ggml_new_tensor_1d(b.pesi.get(), GGML_TYPE_I32, in.righe);
    auto spostate = ggml_new_tensor_1d(b.pesi.get(), GGML_TYPE_I32, in.righe);
    auto topk_c   = ggml_new_tensor_2d(b.pesi_cpu.get(), GGML_TYPE_I32, in.n_topk, in.n_tok);
    auto celle_c  = ggml_new_tensor_1d(b.pesi_cpu.get(), GGML_TYPE_I32, in.n_kv);
    auto query_c  = ggml_new_tensor_1d(b.pesi_cpu.get(), GGML_TYPE_I32, in.righe);
    b.buf.reset(ggml_backend_alloc_ctx_tensors(b.pesi.get(), gpu));
    b.buf_cpu.reset(ggml_backend_alloc_ctx_tensors(b.pesi_cpu.get(), cpu));
    if (!b.buf || !b.buf_cpu) { verifica(false, "allocazione", nome); return; }
    // controllo: ogni query una posizione prima, cioè una cella aperta in meno per riga
    std::vector<int32_t> prima(in.pos_query);
    for (int j = 0; j < in.n_tok; ++j) prima[j] -= 1;
    ggml_backend_tensor_set(maschera, in.maschera.data(), 0, ggml_nbytes(maschera));
    ggml_backend_tensor_set(topk, in.topk.data(), 0, ggml_nbytes(topk));
    ggml_backend_tensor_set(celle, in.pos_celle.data(), 0, ggml_nbytes(celle));
    ggml_backend_tensor_set(query, in.pos_query.data(), 0, ggml_nbytes(query));
    ggml_backend_tensor_set(spostate, prima.data(), 0, ggml_nbytes(spostate));
    ggml_backend_tensor_set(topk_c, in.topk.data(), 0, ggml_nbytes(topk_c));
    ggml_backend_tensor_set(celle_c, in.pos_celle.data(), 0, ggml_nbytes(celle_c));
    ggml_backend_tensor_set(query_c, in.pos_query.data(), 0, ggml_nbytes(query_c));

    auto sparsa   = b.nodo(ggml_indexer_mask(b.ctx.get(), maschera, topk));
    auto oggi     = b.uscita(ggml_mask_to_index(b.ctx.get(), sparsa, (int) in.n_pad));
    auto compatta = b.uscita(ggml_z6_indexer_pos_to_index(b.ctx.get(), celle, query, topk, (int) in.n_pad));
    auto diversa  = b.uscita(ggml_z6_indexer_pos_to_index(b.ctx.get(), celle, spostate, topk, (int) in.n_pad));
    auto sulla_cpu = b.uscita_cpu(ggml_z6_indexer_pos_to_index(b.ctx.get(), celle_c, query_c, topk_c, (int) in.n_pad));
    if (!b.calcola({oggi, compatta, diversa, sulla_cpu})) { verifica(false, "calcolo del grafo", nome); return; }

    const auto v_oggi = leggi(oggi), v_comp = leggi(compatta), v_div = leggi(diversa), v_cpu = leggi(sulla_cpu);
    const size_t n = in.attesa.size()*sizeof(int32_t);
    verifica(v_oggi.size() == n && memcmp(v_oggi.data(), in.attesa.data(), n) == 0, "la strada di oggi dà la lista attesa", nome);
    verifica(v_comp.size() == n && memcmp(v_comp.data(), in.attesa.data(), n) == 0, "la lista dalle posizioni dà la lista attesa", nome);
    verifica(v_oggi == v_comp, "lista dalle posizioni identica a quella di oggi", nome);
    verifica(v_cpu.size() == n && memcmp(v_cpu.data(), in.attesa_cpu.data(), n) == 0, "lista dalle posizioni sulla CPU: stesse celle, per cella crescente", nome);
    // la cella della query sta nel top-k solo con la coda non vuota: senza, spostare la query può non cambiare niente
    if (in.coda) verifica(v_div != v_oggi, "controllo: con le query una posizione prima la lista è diversa", nome);
}

void prova_attenzione(ggml_backend_t gpu, ggml_backend_t cpu, const char * nome, const ingressi & in, unsigned seme) {
    constexpr int d = 512, teste = 64;
    banco b(gpu, cpu);
    auto maschera = ggml_new_tensor_2d(b.pesi.get(), GGML_TYPE_F16, in.n_kv, in.righe);
    auto topk     = ggml_new_tensor_2d(b.pesi.get(), GGML_TYPE_I32, in.n_topk, in.n_tok);
    auto celle    = ggml_new_tensor_1d(b.pesi.get(), GGML_TYPE_I32, in.n_kv);
    auto query    = ggml_new_tensor_1d(b.pesi.get(), GGML_TYPE_I32, in.righe);
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
    ggml_backend_tensor_set(celle, in.pos_celle.data(), 0, ggml_nbytes(celle));
    ggml_backend_tensor_set(query, in.pos_query.data(), 0, ggml_nbytes(query));
    ggml_backend_tensor_set(k, kd.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(q, qd.data(), 0, ggml_nbytes(q));
    const float scala = 1.0f/sqrtf(256.0f);

    // oggi: maschera sparsa e lista letta dalla maschera
    auto sparsa   = b.nodo(ggml_indexer_mask(b.ctx.get(), maschera, topk));
    auto idx_oggi = b.nodo(ggml_mask_to_index(b.ctx.get(), sparsa, (int) in.n_pad));
    auto fa_oggi  = b.uscita(ggml_flash_attn_ext(b.ctx.get(), q, k, k, sparsa, scala, 0.0f, 0.0f));
    fa_oggi->src[5] = idx_oggi;
    // leva: nessuna maschera, lista dalle posizioni
    auto idx_pos  = b.nodo(ggml_z6_indexer_pos_to_index(b.ctx.get(), celle, query, topk, (int) in.n_pad));
    auto fa_pos   = b.uscita(ggml_flash_attn_ext(b.ctx.get(), q, k, k, nullptr, scala, 0.0f, 0.0f));
    fa_pos->src[5] = idx_pos;
    fa_pos->op_params[5] = 2;
    // controllo: maschera causale senza lista, cioè attenzione densa: deve essere diversa dalla sparsa
    auto fa_densa = b.uscita(ggml_flash_attn_ext(b.ctx.get(), q, k, k, maschera, scala, 0.0f, 0.0f));
    if (!b.calcola({fa_oggi, fa_pos, fa_densa})) { verifica(false, "calcolo del grafo", nome); return; }

    const auto v_oggi = leggi(fa_oggi), v_pos = leggi(fa_pos), v_densa = leggi(fa_densa);
    bool finiti = true;
    for (size_t i = 0; i + 4 <= v_oggi.size(); i += 4) { float f; memcpy(&f, v_oggi.data() + i, 4); finiti &= std::isfinite(f); }
    verifica(finiti, "uscita di oggi finita", nome);
    verifica(v_oggi == v_pos, "attenzione senza maschera e lista dalle posizioni identica a quella di oggi", nome);
    verifica(v_oggi != v_densa, "controllo: l'attenzione densa è diversa dalla sparsa", nome);
}

}  // namespace

int main(int argc, char ** argv) {
    const bool solo_indice = argc > 1 && strcmp(argv[1], "--solo-indice") == 0;
    ggml_backend_ptr gpu(ggml_backend_reg_init_backend_from_str("CUDA0"));
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    if (!gpu || !cpu) { fprintf(stderr, "FAIL backend CUDA0 o CPU non disponibile\n"); return 1; }
    ggml_backend_cpu_set_n_threads(cpu.get(), 1);

    // n_kv, posizioni usate, query, pool scelti, r, pool incompleti, celle in disordine, cella vuota nel mezzo
    struct { const char * nome; int64_t n_kv, n_usate; int n_tok, n_sel, r, incompleti; bool disordine, buco; } casi[] = {
        { "262K, 1 query, coda piena",          262144, 222975,   1, 512, 4,   0, false, false },
        { "262K, 1 query, coda vuota",          262144, 222976,   1, 512, 4,   0, false, false },
        { "262K piena, 1 query",                262144, 262144,   1, 512, 4,   0, false, false },
        { "111K, 1 query, 5 pool incompleti",   111360, 111290,   1, 512, 4,   5, false, false },
        { "soglia della lista, 2560 celle",       2560,   2400,   1, 512, 4,   0, false, false },
        { "8K, 2 query (verifica MTP)",           8192,   8001,   2, 512, 4,   1, false, false },
        { "48K, 7 query",                        49152,  48007,   7, 512, 4,   2, false, false },
        { "111K, 100 query (prefill)",          111360, 111300, 100, 512, 4,   3, false, false },
        { "262K, 1024 query (microbatch)",      262144, 222976, 1024, 512, 4,  3, false, false },
        { "16K, tutti i pool incompleti",        16384,  16001,   1, 512, 4, 512, false, false },
        { "48K, celle in disordine, 7 query",    49152,  48007,   7, 512, 4,   2, true,  false },
        { "48K, cella vuota nel mezzo, 33 query", 49152, 48040,  33, 512, 4,   0, false, true  },
        { "48K, disordine e cella vuota, 1 query", 49152, 48003,  1, 512, 4,   1, true,  true  },
    };
    unsigned seme = 20261115;
    for (const auto & c : casi) {
        prova_indice(gpu.get(), cpu.get(), c.nome,
                costruisci(c.n_kv, c.n_usate, c.n_tok, c.n_sel, c.r, c.incompleti, c.disordine, c.buco, seme++));
    }
    if (!solo_indice) {
        struct { const char * nome; int64_t n_kv, n_usate; int n_tok, incompleti; bool disordine; } fa[] = {
            { "FA 16K, 1 query",                                16384, 16002,   1, 0, false },
            { "FA 16K, 1 query, coda vuota e pool incompleti",  16384, 16004,   1, 4, false },
            { "FA 16K, 2 query",                                16384, 16001,   2, 1, false },
            { "FA 16K, 7 query, celle in disordine",            16384, 16007,   7, 1, true  },
            { "FA 16K, 32 query",                               16384, 16040,  32, 2, false },
            { "FA 16K, 100 query, celle in disordine",          16384, 16100, 100, 2, true  },
            { "FA 4K, 32 query",                                 4096,  4000,  32, 0, false },
        };
        for (const auto & c : fa) {
            prova_attenzione(gpu.get(), cpu.get(), c.nome,
                    costruisci(c.n_kv, c.n_usate, c.n_tok, 512, 4, c.incompleti, c.disordine, false, seme), seme + 1000);
            ++seme;
        }
    }
    printf("%s: %d errori\n", errori ? "FALLITO" : "TUTTO IDENTICO", errori);
    return errori ? 1 : 0;
}
