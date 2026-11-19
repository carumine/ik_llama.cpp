// Z6 (GGML_Z6_AFFINITA_STABILE=1): mask of the calling thread (worker 0) with and without the variable.
// The GGML_Z6_* variables are read once per process, so the program runs itself twice (variable off, then on) and
// compares what the two children print. Each child computes the same small CPU graph (mul_mat, add, silu, mul_mat)
// with two workers pinned through cplan.cpu_affinity, directly and through a CPU-only scheduler, and checks:
//   - the numbers are bit-for-bit identical between the two runs and between repetitions. This holds by construction
//     (the caller's mask cannot change CPU results), so it is only a guard: the real checks are those on the mask, and
//     the exactness of the lever on the model is proven with glm-logits.py;
//   - outside a hold the caller is back on the launcher mask after every graph, in both runs;
//   - inside a hold (nested too) the caller stays on the CPU of worker 0 only with the variable, and the outermost
//     hold_end, ggml_z6_affinity_release and a graph without CPU list give back the launcher mask;
//   - during a scheduler call (eval callback between the CPU pieces) the caller stays pinned only with the variable,
//     and after the call it is back on the launcher mask.
// Needs two CPUs in the launcher mask (e.g. taskset -c 78,79) and a build where the lever can act (Linux with OpenMP,
// see ggml_z6_affinity_hold_enabled); otherwise it says SALTATO and exits with 0.
//
//   test-z6-affinita
#include "ggml.h"
#include "ggml-backend.h"

#include <sched.h>
#include <pthread.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_errori = 0;
bool g_acceso = false;
cpu_set_t g_lancio;

cpu_set_t maschera() {
    cpu_set_t m;
    CPU_ZERO(&m);
    pthread_getaffinity_np(pthread_self(), sizeof(m), &m);
    return m;
}

bool solo(const cpu_set_t & m, int cpu) {
    return CPU_COUNT(&m) == 1 && CPU_ISSET(cpu, &m);
}

void controlla(bool ok, const char * cosa) {
    if (!ok) {
        ++g_errori;
        printf("FALLITO: %s (variabile %s)\n", cosa, g_acceso ? "accesa" : "spenta");
    }
}

// pinned on `cpu` with the variable, launcher mask without it
void attesa_tenuta(int cpu, const char * cosa) {
    const cpu_set_t m = maschera();
    controlla(g_acceso ? solo(m, cpu) : CPU_EQUAL(&m, &g_lancio), cosa);
}

void attesa_lancio(const char * cosa) {
    const cpu_set_t m = maschera();
    controlla(CPU_EQUAL(&m, &g_lancio), cosa);
}

uint64_t fnv(const void * p, size_t n, uint64_t h) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

struct grafo {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * a = nullptr, * b = nullptr, * c = nullptr, * out = nullptr;
};

void valori(std::vector<float> & v, uint32_t seme) {
    for (auto & x : v) { seme = seme * 1664525u + 1013904223u; x = ((seme >> 8) & 0xffff) / 32768.0f - 1.0f; }
}

grafo costruisci(bool alloc) {
    grafo g;
    ggml_init_params ip = { 16u << 20, nullptr, !alloc };
    g.ctx = ggml_init(ip);
    g.a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 128, 96);
    g.b = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 128, 8);
    g.c = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 96, 64);
    ggml_set_input(g.a); ggml_set_input(g.b); ggml_set_input(g.c);
    ggml_tensor * x = ggml_mul_mat(g.ctx, g.a, g.b);   // 96 x 8
    x = ggml_add(g.ctx, x, x);
    x = ggml_silu(g.ctx, x);
    g.out = ggml_mul_mat(g.ctx, g.c, x);               // 64 x 8
    ggml_set_output(g.out);
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}

void riempi(grafo & g, bool backend) {
    std::vector<float> va(128 * 96), vb(128 * 8), vc(96 * 64);
    valori(va, 1); valori(vb, 2); valori(vc, 3);
    if (backend) {
        ggml_backend_tensor_set(g.a, va.data(), 0, ggml_nbytes(g.a));
        ggml_backend_tensor_set(g.b, vb.data(), 0, ggml_nbytes(g.b));
        ggml_backend_tensor_set(g.c, vc.data(), 0, ggml_nbytes(g.c));
    } else {
        memcpy(g.a->data, va.data(), ggml_nbytes(g.a));
        memcpy(g.b->data, vb.data(), ggml_nbytes(g.b));
        memcpy(g.c->data, vc.data(), ggml_nbytes(g.c));
    }
}

uint64_t calcola(grafo & g, const int32_t * cpus, int n_cpus, uint64_t h) {
    ggml_cplan plan = ggml_graph_plan(g.gf, 2);
    std::vector<uint8_t> work(plan.work_size + 64);
    plan.work_data = plan.work_size ? work.data() : nullptr;
    plan.cpu_affinity = cpus;
    plan.n_cpu_affinity = n_cpus;
    if (ggml_graph_compute(g.gf, &plan) != GGML_STATUS_SUCCESS) controlla(false, "ggml_graph_compute");
    return fnv(g.out->data, ggml_nbytes(g.out), h);
}

struct stato_callback { int cpu; int visti; int tenuti; };

int callback(ggml_tensor * t, bool ask, void * user_data) {
    (void) t;
    auto * s = (stato_callback *) user_data;
    if (ask) return 1;
    const cpu_set_t m = maschera();
    ++s->visti;
    s->tenuti += g_acceso ? solo(m, s->cpu) : CPU_EQUAL(&m, &g_lancio);
    return 1;
}

int figlio() {
    const char * e = getenv("GGML_Z6_AFFINITA_STABILE");
    g_acceso = e && atoi(e) > 0;
    if (g_acceso && !ggml_z6_affinity_hold_enabled()) {
        // Z6 (GGML_Z6_AFFINITA_STABILE): without OpenMP the lever stays off by design, nothing to check
        printf("SALTATO: la leva non si attiva in questa build (serve OpenMP su Linux)\n");
        return 0;
    }
    g_lancio = maschera();
    int32_t cpus[2] = {-1, -1};
    for (int c = 0, k = 0; c < CPU_SETSIZE && k < 2; ++c) if (CPU_ISSET(c, &g_lancio)) cpus[k++] = c;
    if (cpus[1] < 0) {
        printf("SALTATO: servono almeno due CPU nella maschera di lancio\n");
        return 0;
    }
    const int32_t rovescio[2] = {cpus[1], cpus[0]};

    grafo g = costruisci(true);
    riempi(g, false);
    uint64_t h = 1469598103934665603ull, primo = 0;

    // outside a hold: launcher mask after every graph, with and without the variable
    for (int i = 0; i < 3; ++i) {
        const uint64_t r = calcola(g, cpus, 2, 1469598103934665603ull);
        if (i == 0) primo = r; else controlla(r == primo, "ripetizioni diverse");
        h = fnv(&r, sizeof(r), h);
        attesa_lancio("fuori dalla tenuta");
    }

    // nested holds
    ggml_z6_affinity_hold_begin();
    h = calcola(g, cpus, 2, h);
    attesa_tenuta(cpus[0], "tenuta, primo grafo");
    h = calcola(g, cpus, 2, h);
    attesa_tenuta(cpus[0], "tenuta, secondo grafo");
    ggml_z6_affinity_hold_begin();
    h = calcola(g, cpus, 2, h);
    ggml_z6_affinity_hold_end();
    attesa_tenuta(cpus[0], "fine della tenuta interna");
    // a graph without a CPU list runs with the launcher mask, as without the variable
    h = calcola(g, nullptr, 0, h);
    attesa_lancio("grafo senza lista di CPU dentro la tenuta");
    h = calcola(g, cpus, 2, h);
    attesa_tenuta(cpus[0], "di nuovo legato");
    // another CPU list moves worker 0
    h = calcola(g, rovescio, 2, h);
    attesa_tenuta(rovescio[0], "lista di CPU diversa");
    ggml_z6_affinity_release();
    attesa_lancio("ggml_z6_affinity_release");
    h = calcola(g, cpus, 2, h);
    attesa_tenuta(cpus[0], "legato dopo release");
    ggml_z6_affinity_hold_end();
    attesa_lancio("fine della tenuta esterna");
    ggml_free(g.ctx);

    // scheduler: one hold per call, pinned between the CPU pieces only with the variable
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 2);
    ggml_backend_cpu_set_cpu_affinity(cpu, cpus, 2);
    ggml_backend_sched_t sched = ggml_backend_sched_new(&cpu, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false);
    stato_callback sc = { cpus[0], 0, 0 };
    ggml_backend_sched_set_eval_callback(sched, callback, &sc);
    for (int i = 0; i < 2; ++i) {
        grafo gs = costruisci(false);
        if (!ggml_backend_sched_alloc_graph(sched, gs.gf)) { controlla(false, "alloc del grafo"); break; }
        riempi(gs, true);
        if (ggml_backend_sched_graph_compute(sched, gs.gf) != GGML_STATUS_SUCCESS) controlla(false, "sched compute");
        std::vector<float> out(ggml_nelements(gs.out));
        ggml_backend_tensor_get(gs.out, out.data(), 0, ggml_nbytes(gs.out));
        const uint64_t r = fnv(out.data(), out.size() * sizeof(float), 1469598103934665603ull);
        controlla(r == primo, "scheduler e calcolo diretto diversi");
        h = fnv(&r, sizeof(r), h);
        attesa_lancio("dopo una chiamata dello scheduler");
        ggml_backend_sched_reset(sched);
        ggml_free(gs.ctx);
    }
    controlla(sc.visti > 0 && sc.tenuti == sc.visti, "maschera durante la chiamata dello scheduler");
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);

    printf("risultato %016llx\n", (unsigned long long) h);
    printf("callback %d, errori %d\n", sc.visti, g_errori);
    return g_errori ? 1 : 0;
}

std::string lancia(const char * self, const char * acceso, int & esito) {
    std::string cmd = std::string("GGML_Z6_NUMA_AFFINITY=1 GGML_Z6_AFFINITA_STABILE=") + acceso + " '" + self + "' --figlio";
    FILE * p = popen(cmd.c_str(), "r");
    std::string out;
    if (!p) { esito = -1; return out; }
    char buf[512];
    while (fgets(buf, sizeof(buf), p)) out += buf;
    esito = pclose(p);
    return out;
}

std::string riga(const std::string & out, const char * inizio) {
    const size_t i = out.find(inizio);
    if (i == std::string::npos) return "";
    return out.substr(i, out.find('\n', i) - i);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--figlio") == 0) return figlio();
    int e0 = 0, e1 = 0;
    const std::string spento = lancia(argv[0], "0", e0), acceso = lancia(argv[0], "1", e1);
    printf("--- variabile spenta (esito %d)\n%s--- variabile accesa (esito %d)\n%s", e0, spento.c_str(), e1, acceso.c_str());
    if (e0 == 0 && e1 == 0 &&
        (spento.find("SALTATO") != std::string::npos || acceso.find("SALTATO") != std::string::npos)) {
        printf("SALTATO\n");
        return 0;
    }
    const std::string r0 = riga(spento, "risultato "), r1 = riga(acceso, "risultato ");
    if (e0 != 0 || e1 != 0 || r0.empty() || r0 != r1) { printf("DIVERSO o FALLITO\n"); return 1; }
    printf("TUTTO IDENTICO (maschera del thread 0 come attesa con e senza variabile; numeri uguali per costruzione, "
           "l'esattezza sul modello si prova con glm-logits.py)\n");
    return 0;
}
