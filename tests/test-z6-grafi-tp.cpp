// Z6: CUDA graph with the exchange of the split dense matrices (GGML_Z6_DENSA_TP) relaunched after its arena was freed
// and allocated again at the same address. Senza modello, DUE GPU, pochi secondi.
//
// test-z6-grafi-epoca usa solo copie D2D nello stesso contesto e non crolla (14/11, 17:16). Qui il grafo catturato
// contiene i nodi che i grafi di generazione hanno davvero (ggml-cuda.cu, z6_tp_try):
//   - una copia D2H eseguita sullo stream della GPU remota che legge l'arena della GPU proprietaria (fra contesti);
//   - una copia H2D a due dimensioni dalla RAM pinned verso la stessa arena;
//   - i kernel delle due meta' della matrice, su due contesti.
// Passi:
//   1. una matrice Q8_0 divisa per righe fra le due GPU e registrata con ggml_backend_cuda_z6_tp_register;
//   2. grafo y = mul_mat(w, x), d = scale(y): x, y e d stanno in un'arena nostra sulla GPU proprietaria;
//   3. calcoli di riferimento (cattura, istanza, lanci), un risultato per ogni giro;
//   4. a ogni giro l'arena si libera e se ne alloca una di poco diversa dentro lo stesso multiplo di 2 MiB (come
//      CUDA1 730,38 -> 731,38 MiB e 732,91 -> 733,62 MiB del 14/11); poi buffer CUDA e blocchi dell'heap nuovi, per far
//      riusare la memoria degli oggetti del driver appena liberati;
//   5. stessi tensori agli stessi offset: la cache dei grafi rilancia l'istanza vecchia.
// Senza GGML_Z6_GRAFI_EPOCA=1 l'attesa e' SIGSEGV in libcuda sotto cuGraphLaunch oppure "CUDA error: invalid argument"
// in cudaGraphLaunch (uscita 134); con GGML_Z6_GRAFI_EPOCA=1 il risultato e' identico al riferimento, bit per bit.
//
//   test-z6-grafi-tp [--owner 1] [--remote 0] [--mib 64] [--giri 6]
//
// Uscita: 0 identico, 1 diverso, 2 inconcludente (mai lo stesso indirizzo), 3 preparazione fallita.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

size_t pad(size_t n) { return (n + 255) & ~(size_t) 255; }

struct tensori { ggml_tensor * x, * y, * d; };

// x in testa, poi y (mul_mat) e d (scale): sempre agli stessi offset dell'arena
void piazza(ggml_backend_buffer_t buf, const tensori & t) {
    char * base = (char *) ggml_backend_buffer_get_base(buf);
    size_t off = 0;
    for (ggml_tensor * v : {t.x, t.y, t.d}) {
        v->buffer = nullptr;
        v->data = nullptr;
        ggml_backend_tensor_alloc(buf, v, base + off);
        off += pad(ggml_nbytes(v));
    }
}

bool calcola(ggml_backend_t gpu, ggml_cgraph * gf, const tensori & t, int seme, std::vector<float> & out) {
    const int64_t k = t.x->ne[0];
    std::vector<float> vx(k);
    for (int64_t i = 0; i < k; ++i) vx[i] = 0.01f*(float) ((i*7 + seme*13) % 97) - 0.4f;
    ggml_backend_tensor_set(t.x, vx.data(), 0, vx.size()*sizeof(float));
    if (ggml_backend_graph_compute(gpu, gf) != GGML_STATUS_SUCCESS) return false;
    ggml_backend_synchronize(gpu);
    out.resize(ggml_nelements(t.d));
    ggml_backend_tensor_get(t.d, out.data(), 0, out.size()*sizeof(float));
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    int owner = 1, remote = 0, mib = 64, giri = 6;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--owner" && i + 1 < argc) owner = atoi(argv[++i]);
        else if (a == "--remote" && i + 1 < argc) remote = atoi(argv[++i]);
        else if (a == "--mib" && i + 1 < argc) mib = atoi(argv[++i]);
        else if (a == "--giri" && i + 1 < argc) giri = atoi(argv[++i]);
        else { fprintf(stderr, "uso: %s [--owner 1] [--remote 0] [--mib 64] [--giri 6]\n", argv[0]); return 3; }
    }
    if (owner == remote || giri < 1) { fprintf(stderr, "FAIL servono due GPU diverse e almeno un giro\n"); return 3; }
    const char * e = getenv("GGML_Z6_GRAFI_EPOCA");
    printf("GGML_Z6_GRAFI_EPOCA=%s, proprietaria CUDA%d, remota CUDA%d\n", e ? e : "(spenta)", owner, remote);

    ggml_backend_t gpu_o = ggml_backend_cuda_init(owner, nullptr, nullptr);
    ggml_backend_t gpu_r = ggml_backend_cuda_init(remote, nullptr, nullptr);
    if (!gpu_o || !gpu_r) { fprintf(stderr, "FAIL backend CUDA\n"); return 3; }
    ggml_backend_buffer_type_t buft_o = ggml_backend_cuda_buffer_type(owner);
    ggml_backend_buffer_type_t buft_r = ggml_backend_cuda_buffer_type(remote);

    // la matrice divisa: k colonne, n righe, r0 sulla proprietaria e le altre sulla remota
    constexpr int64_t k = 4096, n = 4096, r0 = 2048, nr = n - r0;
    const ggml_type tipo = GGML_TYPE_Q8_0;
    if (!ggml_backend_cuda_z6_tp_type_ok(tipo)) { fprintf(stderr, "FAIL tipo non ammesso dalla TP\n"); return 3; }
    ggml_init_params ip = { 8*ggml_tensor_overhead(), nullptr, true };
    ggml_context * cw_o = ggml_init(ip);
    ggml_context * cw_r = ggml_init(ip);
    ggml_tensor * w   = ggml_new_tensor_2d(cw_o, tipo, k, n);
    ggml_tensor * y_l = ggml_new_tensor_2d(cw_o, GGML_TYPE_F32, r0, 16);
    ggml_tensor * w_r = ggml_new_tensor_2d(cw_r, tipo, k, nr);
    ggml_tensor * x_r = ggml_new_tensor_2d(cw_r, GGML_TYPE_F32, k, 16);
    ggml_tensor * y_r = ggml_new_tensor_2d(cw_r, GGML_TYPE_F32, nr, 16);
    ggml_backend_buffer_t pesi_o = ggml_backend_alloc_ctx_tensors(cw_o, gpu_o);
    ggml_backend_buffer_t pesi_r = ggml_backend_alloc_ctx_tensors(cw_r, gpu_r);
    if (!pesi_o || !pesi_r) { fprintf(stderr, "FAIL buffer dei pesi\n"); return 3; }
    {
        std::vector<float> f((size_t) k*n);
        uint32_t s = 12345;
        for (auto & v : f) { s = s*1664525u + 1013904223u; v = ((float) (s >> 8) / 16777216.0f - 0.5f)*0.2f; }
        std::vector<char> q(ggml_nbytes(w));
        const size_t scritti = ggml_quantize_chunk(tipo, f.data(), q.data(), 0, n, k, nullptr, nullptr);
        if (scritti != q.size()) { fprintf(stderr, "FAIL quantizzazione: %zu byte invece di %zu\n", scritti, q.size()); return 3; }
        ggml_backend_tensor_set(w, q.data(), 0, q.size());
        ggml_backend_tensor_set(w_r, q.data() + (size_t) r0*w->nb[1], 0, (size_t) nr*w->nb[1]);
    }
    if (!ggml_backend_cuda_z6_tp_register(w, r0, w_r, x_r, y_r, y_l)) { fprintf(stderr, "FAIL registrazione TP\n"); return 3; }

    // il grafo: un mul_mat sulla matrice divisa (una colonna, come un passo di generazione) e uno scale
    ggml_init_params ig = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ig);
    tensori t;
    t.x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    t.y = ggml_mul_mat(ctx, w, t.x);
    t.d = ggml_scale(ctx, t.y, 2.0f);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t.d);

    // prima arena: mib MiB + 256 KiB; le altre 512 KiB in piu' o uguali, dentro lo stesso multiplo di 2 MiB
    const size_t primo = (size_t) mib*1048576 + 256*1024, secondo = primo + 512*1024;
    ggml_backend_buffer_t arena = ggml_backend_buft_alloc_buffer(buft_o, primo);
    if (!arena) { fprintf(stderr, "FAIL prima arena\n"); return 3; }
    piazza(arena, t);
    std::vector<std::vector<float>> rif(giri);
    bool ok = true;
    for (int g = 0; g < giri; ++g) ok = calcola(gpu_o, gf, t, g, rif[g]) && ok;
    double somma = 0;
    for (float v : rif[0]) somma += v;
    printf("riferimento: %s, %d calcoli, somma del primo %.6f\n", ok ? "calcolato" : "FALLITO", giri, somma);
    if (!ok) return 3;

    int esito = 0;
    bool visto_stesso = false;
    std::vector<void *> spazzatura;
    std::vector<ggml_backend_buffer_t> spazzatura_cuda;
    for (int giro = 0; giro < giri && esito == 0; ++giro) {
        void * vecchio = ggml_backend_buffer_get_base(arena);
        ggml_backend_synchronize(gpu_o);
        ggml_backend_synchronize(gpu_r);
        ggml_backend_buffer_free(arena);
        arena = ggml_backend_buft_alloc_buffer(buft_o, giro % 2 ? primo : secondo);
        if (!arena) { fprintf(stderr, "FAIL arena del giro %d\n", giro); return 3; }
        const bool stesso = ggml_backend_buffer_get_base(arena) == vecchio;
        visto_stesso = visto_stesso || stesso;
        printf("giro %d: nuova arena %s (%p -> %p)\n", giro, stesso ? "allo STESSO indirizzo" : "a un altro indirizzo",
                vecchio, ggml_backend_buffer_get_base(arena));
        // oggetti nuovi del driver e dell'heap dopo la nuova arena: la memoria dell'oggetto liberato viene riusata
        for (int r = 0; r < 8; ++r) {
            if (auto b = ggml_backend_buft_alloc_buffer(buft_o, ((size_t) 1 + r % 4)*1048576)) spazzatura_cuda.push_back(b);
            if (auto b = ggml_backend_buft_alloc_buffer(buft_r, ((size_t) 1 + r % 4)*1048576)) spazzatura_cuda.push_back(b);
            if (void * p = ggml_backend_cuda_z6_host_alloc((size_t) 65536 << (r % 4))) spazzatura.push_back(p);
        }
        for (int r = 0; r < 64; ++r) {
            for (size_t sz = 16; sz <= 4096; sz += 16) {
                void * p = malloc(sz);
                if (p) { memset(p, 0xA5, sz); free(p); }
            }
        }
        fflush(stdout);
        piazza(arena, t);
        // senza GGML_Z6_GRAFI_EPOCA, con lo stesso indirizzo, qui la cache rilancia l'istanza catturata prima
        std::vector<float> out;
        const bool calcolato = calcola(gpu_o, gf, t, giro, out);
        const bool identico = calcolato && out.size() == rif[giro].size() &&
                memcmp(out.data(), rif[giro].data(), out.size()*sizeof(float)) == 0;
        printf("giro %d: risultato %s\n", giro, identico ? "identico" : "DIVERSO");
        if (!identico) esito = 1;
    }
    if (esito == 0 && !visto_stesso) esito = 2;
    ggml_backend_synchronize(gpu_o);
    ggml_backend_synchronize(gpu_r);
    for (auto b : spazzatura_cuda) ggml_backend_buffer_free(b);
    for (void * p : spazzatura) ggml_backend_cuda_z6_host_free(p);
    ggml_backend_buffer_free(arena);
    ggml_backend_cuda_z6_tp_unregister(w);
    ggml_backend_buffer_free(pesi_o);
    ggml_backend_buffer_free(pesi_r);
    ggml_free(ctx);
    ggml_free(cw_o);
    ggml_free(cw_r);
    ggml_backend_free(gpu_o);
    ggml_backend_free(gpu_r);
    printf("esito %d (%s)\n", esito, esito == 0 ? "identico" : esito == 1 ? "DIVERSO" : "inconcludente: indirizzo diverso");
    return esito;
}
