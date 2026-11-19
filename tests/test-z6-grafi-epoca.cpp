// Z6: CUDA graph relaunched after its buffer was freed and allocated again at the same address (GGML_Z6_GRAFI_EPOCA).
// Senza modello, una GPU, pochi secondi. Riproduce il meccanismo dei segfault in libcuda del 14/11:
//   1. un grafo ggml con un CONCAT contiguo (due cudaMemcpyAsync D2D, cioè due nodi memcpy nel grafo CUDA catturato)
//      e uno SCALE, tutti i tensori in un buffer CUDA nostro;
//   2. due calcoli: cattura, istanza e lancio, poi solo lancio;
//   3. il buffer si libera e se ne alloca un altro di poco più grande, dentro lo stesso arrotondamento a 2 MiB
//      (come l'arena CUDA1 732,91 -> 733,62 MiB del 14/11): di solito il driver restituisce lo stesso indirizzo;
//   4. si riempie l'heap con blocchi piccoli scritti a 0xA5, così l'oggetto del driver liberato viene sovrascritto;
//   5. stessi tensori agli stessi offset: ogni indirizzo dei nodi è identico e la cache dei grafi CUDA rilancia
//      l'istanza vecchia, i cui nodi memcpy puntano all'allocazione liberata.
// Senza GGML_Z6_GRAFI_EPOCA=1 ci si aspetta il segfault in libcuda (o, se l'oggetto liberato è ancora intatto, un
// passaggio fortunato); con GGML_Z6_GRAFI_EPOCA=1 il grafo si ricattura e il risultato torna giusto.
//
//   test-z6-grafi-epoca [--device 0] [--mib 64] [--giri 3]
//
// Uscita: 0 risultato giusto, 1 risultato sbagliato, 2 inconcludente (il secondo buffer ha un altro indirizzo).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct tensori {
    ggml_tensor * a, * b, * c, * d;
};

size_t pad(size_t n) { return (n + 255) & ~(size_t) 255; }

// a e b in testa, poi c (concat) e d (scale): sempre agli stessi offset
void piazza(ggml_backend_buffer_t buf, const tensori & t) {
    char * base = (char *) ggml_backend_buffer_get_base(buf);
    size_t off = 0;
    for (ggml_tensor * x : {t.a, t.b, t.c, t.d}) {
        x->buffer = nullptr;
        x->data = nullptr;
        ggml_backend_tensor_alloc(buf, x, base + off);
        off += pad(ggml_nbytes(x));
    }
}

bool calcola_e_verifica(ggml_backend_t gpu, ggml_cgraph * gf, const tensori & t, float seme) {
    const int64_t n = t.a->ne[0], ra = t.a->ne[1], rb = t.b->ne[1];
    std::vector<float> va(n*ra), vb(n*rb), vd(n*(ra + rb));
    for (size_t i = 0; i < va.size(); ++i) va[i] = seme + 0.001f*(float) i;
    for (size_t i = 0; i < vb.size(); ++i) vb[i] = -seme - 0.002f*(float) i;
    ggml_backend_tensor_set(t.a, va.data(), 0, va.size()*sizeof(float));
    ggml_backend_tensor_set(t.b, vb.data(), 0, vb.size()*sizeof(float));
    if (ggml_backend_graph_compute(gpu, gf) != GGML_STATUS_SUCCESS) return false;
    ggml_backend_synchronize(gpu);
    ggml_backend_tensor_get(t.d, vd.data(), 0, vd.size()*sizeof(float));
    for (size_t i = 0; i < va.size(); ++i) if (vd[i] != 2.0f*va[i]) return false;
    for (size_t i = 0; i < vb.size(); ++i) if (vd[va.size() + i] != 2.0f*vb[i]) return false;
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    int device = 0, mib = 64, giri = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) device = atoi(argv[++i]);
        else if (a == "--mib" && i + 1 < argc) mib = atoi(argv[++i]);
        else if (a == "--giri" && i + 1 < argc) giri = atoi(argv[++i]);
        else { fprintf(stderr, "uso: %s [--device 0] [--mib 64] [--giri 3]\n", argv[0]); return 1; }
    }
    const char * e = getenv("GGML_Z6_GRAFI_EPOCA");
    printf("GGML_Z6_GRAFI_EPOCA=%s\n", e ? e : "(spenta)");

    ggml_backend_t gpu = ggml_backend_cuda_init(device, nullptr, nullptr);
    if (!gpu) { fprintf(stderr, "FAIL backend CUDA %d\n", device); return 1; }
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(device);

    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    constexpr int64_t n = 4096, ra = 48, rb = 16;
    tensori t;
    t.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, ra);
    t.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, rb);
    t.c = ggml_concat(ctx, t.a, t.b, 1);   // contiguo, dim 1, ne2 = ne3 = 1: due cudaMemcpyAsync D2D
    t.d = ggml_scale(ctx, t.c, 2.0f);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t.d);

    // primo buffer: mib MiB + 256 KiB; il secondo 512 KiB in più, dentro lo stesso multiplo di 2 MiB
    const size_t primo = (size_t) mib*1048576 + 256*1024, secondo = primo + 512*1024;
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, primo);
    if (!buf) { fprintf(stderr, "FAIL primo buffer\n"); return 1; }
    piazza(buf, t);
    bool ok = true;
    for (int g = 0; g < 2; ++g) ok = calcola_e_verifica(gpu, gf, t, 1.0f + g) && ok;
    printf("prima della liberazione: %s\n", ok ? "giusto" : "SBAGLIATO");

    int esito = ok ? 0 : 1;
    bool visto_stesso = false;
    std::vector<void *> spazzatura;
    for (int giro = 0; giro < giri && esito == 0; ++giro) {
        void * vecchio = ggml_backend_buffer_get_base(buf);
        ggml_backend_synchronize(gpu);
        ggml_backend_buffer_free(buf);
        // l'heap si riempie di blocchi piccoli scritti a 0xA5: gli oggetti del driver appena liberati vengono riusati
        for (int r = 0; r < 64; ++r) {
            for (size_t sz = 16; sz <= 4096; sz += 16) {
                void * p = malloc(sz);
                if (p) { memset(p, 0xA5, sz); spazzatura.push_back(p); }
            }
        }
        buf = ggml_backend_buft_alloc_buffer(buft, giro % 2 ? primo : secondo);
        if (!buf) { fprintf(stderr, "FAIL buffer del giro %d\n", giro); return 1; }
        const bool stesso = ggml_backend_buffer_get_base(buf) == vecchio;
        visto_stesso = visto_stesso || stesso;
        printf("giro %d: nuovo buffer %s (%p -> %p)\n", giro, stesso ? "allo STESSO indirizzo" : "a un altro indirizzo",
                vecchio, ggml_backend_buffer_get_base(buf));
        fflush(stdout);
        piazza(buf, t);
        // senza GGML_Z6_GRAFI_EPOCA, con lo stesso indirizzo, qui la cache rilancia l'istanza catturata prima
        const bool giusto = calcola_e_verifica(gpu, gf, t, 10.0f + giro);
        printf("giro %d: risultato %s\n", giro, giusto ? "giusto" : "SBAGLIATO");
        if (!giusto) esito = 1;
    }
    if (esito == 0 && !visto_stesso) esito = 2;
    for (void * p : spazzatura) free(p);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(gpu);
    printf("esito %d (%s)\n", esito, esito == 0 ? "giusto" : esito == 1 ? "SBAGLIATO" : "inconcludente: indirizzo diverso");
    return esito;
}
