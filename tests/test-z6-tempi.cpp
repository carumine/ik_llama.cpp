// Z6: the host timers of GGML_Z6_TEMPI on the CUDA graph launch path (B3) must not change what the GPU computes.
// Senza modello, una GPU, pochi secondi. Il processo si rilancia due volte, senza e con GGML_Z6_TEMPI=1 (la variabile
// si legge una volta sola per processo); ogni figlio:
//   1. costruisce un grafo con uno SCALE e una CPY (la CPY mette in gioco la copia dei puntatori, cpy.cu), tutto in un
//      buffer CUDA;
//   2. lo calcola 4 volte con gli stessi ingressi: cattura e istanza del grafo CUDA, poi tre lanci;
//   3. controlla i contatori: senza variabile tutti a zero (ggml_backend_cuda_z6_tempi_avvio e
//      ggml_backend_sched_z6_pipeline_tempi); con la variabile almeno un lancio, una cattura e una copia dei puntatori;
//   4. stampa l'impronta FNV-1a dei byte dell'uscita.
// Il padre confronta le due impronte.
//
//   test-z6-tempi [--device 0]
//
// Uscita: 0 TUTTO IDENTICO e contatori come attesi, 1 altrimenti.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

uint64_t fnv1a(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= ((const unsigned char *) p)[i]; h *= 1099511628211ull; }
    return h;
}

int figlio(int device) {
    const char * e = getenv("GGML_Z6_TEMPI");
    const bool tempi = e && atoi(e) > 0;
    int64_t av[10] = {};
    ggml_backend_cuda_z6_tempi_avvio(av, 10);
    for (int64_t v : av) if (v != 0) { printf("FAIL contatori non nulli prima del calcolo\n"); return 1; }

    ggml_backend_t gpu = ggml_backend_cuda_init(device, nullptr, nullptr);
    if (!gpu) { printf("FAIL backend CUDA %d\n", device); return 1; }
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    constexpr int64_t n = 4096, r = 8;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, r);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, r);
    ggml_tensor * s = ggml_scale(ctx, a, 0.37f);
    ggml_tensor * c = ggml_cpy(ctx, s, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    if (!buf) { printf("FAIL buffer\n"); return 1; }

    std::vector<float> va(n*r), vb(n*r);
    for (size_t i = 0; i < va.size(); ++i) va[i] = 1.0f/(1.0f + (float) i) - 0.25f*(float) (i % 7);
    ggml_backend_tensor_set(a, va.data(), 0, va.size()*sizeof(float));
    for (int g = 0; g < 4; ++g) {
        if (ggml_backend_graph_compute(gpu, gf) != GGML_STATUS_SUCCESS) { printf("FAIL calcolo %d\n", g); return 1; }
    }
    ggml_backend_synchronize(gpu);
    ggml_backend_tensor_get(b, vb.data(), 0, vb.size()*sizeof(float));
    const uint64_t h = fnv1a(vb.data(), vb.size()*sizeof(float));

    ggml_backend_cuda_z6_tempi_avvio(av, 10);
    int64_t coppie = -1, us = -1;
    ggml_backend_sched_z6_pipeline_tempi(&coppie, &us);
    int esito = 0;
    if (coppie != 0 || us != 0) { printf("FAIL contatori della pipeline non nulli senza coppie\n"); esito = 1; }
    if (!tempi) {
        for (int64_t v : av) if (v != 0) { printf("FAIL contatori non nulli senza GGML_Z6_TEMPI\n"); esito = 1; break; }
    } else if (av[6] < 3 || av[9] < 1 || av[7] < 1) {
        printf("FAIL con GGML_Z6_TEMPI=1: lanci %" PRId64 ", catture %" PRId64 ", copie dei puntatori %" PRId64
               " (attesi almeno 3, 1, 1)\n", av[6], av[9], av[7]);
        esito = 1;
    }
    printf("GGML_Z6_TEMPI=%s: impronta %016" PRIx64 "; lanci %" PRId64 ", catture %" PRId64 ", dirette %" PRId64
           ", copie dei puntatori %" PRId64 "; us: proprieta %" PRId64 ", compatibilita %" PRId64 ", puntatori %" PRId64
           ", cudaGraphLaunch %" PRId64 ", cattura %" PRId64 ", istanza %" PRId64 "\n",
           tempi ? e : "(spenta)", h, av[6], av[9], av[8], av[7], av[0], av[1], av[2], av[3], av[4], av[5]);
    printf("impronta=%016" PRIx64 " esito=%d\n", h, esito);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(gpu);
    return esito;
}

// runs this executable again with or without the variable and returns the fingerprint line ("" on failure)
std::string rilancia(const std::string & exe, int device, bool tempi) {
    const std::string cmd = std::string(tempi ? "env GGML_Z6_TEMPI=1 " : "env -u GGML_Z6_TEMPI ") + "'" + exe +
            "' --figlio --device " + std::to_string(device);
    FILE * f = popen(cmd.c_str(), "r");
    if (!f) return "";
    std::string riga, ultima;
    char b[1024];
    while (fgets(b, sizeof(b), f)) {
        fputs(b, stdout);
        riga = b;
        if (riga.rfind("impronta=", 0) == 0) ultima = riga;
    }
    const int st = pclose(f);
    return st == 0 ? ultima : "";
}

} // namespace

int main(int argc, char ** argv) {
    int device = 0;
    bool da_figlio = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) device = atoi(argv[++i]);
        else if (a == "--figlio") da_figlio = true;
        else { fprintf(stderr, "uso: %s [--device 0]\n", argv[0]); return 1; }
    }
    if (da_figlio) return figlio(device);

    char exe[4096];
    const ssize_t k = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (k <= 0) { printf("FAIL /proc/self/exe\n"); return 1; }
    exe[k] = 0;
    const std::string senza = rilancia(exe, device, false);
    const std::string con = rilancia(exe, device, true);
    if (senza.empty() || con.empty()) { printf("FAIL: un figlio è fallito\n"); return 1; }
    if (senza != con) { printf("DIVERSO: %s       contro %s", senza.c_str(), con.c_str()); return 1; }
    printf("TUTTO IDENTICO: uscita bit per bit uguale senza e con GGML_Z6_TEMPI, contatori come attesi\n");
    return 0;
}
