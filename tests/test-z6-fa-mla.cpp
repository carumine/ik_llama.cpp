// Z6: microbanco dell'attenzione MLA di GLM-5.3-Flash sulla GPU (buco di misura B6) per GGML_Z6_FA_MLA_RIUSO (F1,
// esatta) e GGML_Z6_FA_DECODE_SM (F2, cambia i numeri). Senza modello, una GPU, un paio di minuti.
//   Forma di GLM (build_glm5next.cpp): Q [512, T, 64 teste] f32, K == V lo stesso tensore latente [512, n_kv, 1 testa]
//   f16, maschera causale f16 [n_kv, pad16(T)], scala 1/16 (1/sqrt(256)), senza alibi né softcap. test-backend-ops
//   non copre la forma: crea K e V separati, con tante teste quante Q.
//   T = 1, 2 (decode e livello MTP), 300, 1024 (prefill); contesto 8192, 32768, 50001 (non multiplo di 256: n_kv
//   arrotondato a 50176 e coda mascherata, come la cache vera), 111616.
// Le variabili sono lette una volta per processo, quindi il test rilancia se stesso (fork/exec) per ogni configurazione:
// base, F1, F2, F1+F2. Ogni figlio salva le uscite su file e il padre le confronta byte per byte:
//   - base contro F1 e F2 contro F1+F2 devono essere IDENTICHE (F1 è esatta, anche sopra la griglia di F2);
//   - F2 contro base: elementi diversi e differenza massima (F2 sposta i tagli stream-K); con T > 2 F2 non agisce e
//     le uscite devono restare identiche;
//   - tempi medi per caso di ogni configurazione, in --giri giri a ordine invertito (rumore circa ±1-2%);
//   - controlli: riferimento CPU in double (base, 8192 celle, T 1 e 2), stessa uscita a ogni ripetizione e fra i giri,
//     nessun NaN, riga «z6: ... attiva» nel log dei figli con F1 (prova che la leva ha preso il percorso).
//
//   flock ~/Workspace/z6/.gpu-lock env CUDA_DEVICE_ORDER=PCI_BUS_ID test-z6-fa-mla [--device 0] [--giri 2]
//       [--secondi 0.4] [--solo-decode] [--dir D] [--tieni]
//
// Uscita 0 con «TUTTO IDENTICO» (F1 identica ovunque e ogni controllo passato), 1 altrimenti.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int   D     = 512;      // latente di GLM-5.3-Flash (kv_lora_rank): testa di K, di V e della Q assorbita
constexpr int   H     = 64;       // teste di Q su una sola testa KV
constexpr float SCALA = 0.0625f;  // kq_scale di build_glm5next.cpp: 1/sqrt(256)
constexpr double SOGLIA_RIF = 0.05;  // scarto quadratico relativo massimo dal riferimento in double

const int TT[]  = {1, 2, 300, 1024};
const int CTX[] = {8192, 32768, 50001, 111616};

const char * const VAR_F1   = "GGML_Z6_FA_MLA_RIUSO";
const char * const VAR_F2   = "GGML_Z6_FA_DECODE_SM";
const char * const SEGNO_F1 = "GGML_Z6_FA_MLA_RIUSO attiva";
const char * const SEGNO_F2 = "GGML_Z6_FA_DECODE_SM attiva";

struct conf { const char * tag; bool f1, f2; };
const conf CONF[4] = {{"base", false, false}, {"f1", true, false}, {"f2", false, true}, {"f1f2", true, true}};

struct caso { int T, ctx; };

std::vector<caso> casi(bool solo_decode) {
    std::vector<caso> v;
    for (int T : TT) {
        if (solo_decode && T > 2) continue;
        for (int c : CTX) v.push_back({T, c});
    }
    return v;
}

std::string nome_bin(const std::string & dir, const char * tag, const caso & c) {
    return dir + "/" + tag + "-T" + std::to_string(c.T) + "-c" + std::to_string(c.ctx) + ".bin";
}

std::string nome_ris(const std::string & dir, const char * tag, int giro) {
    return dir + "/" + tag + "-g" + std::to_string(giro) + ".txt";
}

// splitmix64: stessi dati in ogni processo
struct rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30))*0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27))*0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float u() { return (float) (next() >> 40)*(2.0f/16777216.0f) - 1.0f; }  // [-1, 1)
};

uint64_t fnv(const void * p, size_t n) {
    const unsigned char * b = (const unsigned char *) p;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

double ora() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct dati {
    std::vector<float>       q;  // [D, T, H]
    std::vector<ggml_fp16_t> k;  // [D, n_kv]
    std::vector<ggml_fp16_t> m;  // [n_kv, pad16(T)]
};

void genera(const caso & c, int64_t n_kv, int64_t righe_m, dati & x) {
    rng r{0x5A6E15F00Dull + (uint64_t) c.T*7919ull + (uint64_t) c.ctx*104729ull};
    x.q.resize((size_t) D*c.T*H);
    for (float & v : x.q) v = 2.0f*r.u();
    x.k.resize((size_t) D*n_kv);
    std::vector<float> riga(D);
    for (int64_t i = 0; i < n_kv; ++i) {
        for (float & v : riga) v = 1.7f*r.u();
        ggml_fp32_to_fp16_row(riga.data(), x.k.data() + (size_t) i*D, D);
    }
    // le T righe sono gli ultimi token del contesto: la riga j vede le celle 0 .. ctx-T+j; il resto, la coda oltre ctx
    // e le righe di riempimento fino a pad16(T) valgono -inf, come la maschera di llama.cpp
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), meno_inf = ggml_fp32_to_fp16(-INFINITY);
    x.m.assign((size_t) n_kv*righe_m, meno_inf);
    for (int j = 0; j < c.T; ++j) {
        const size_t r0 = (size_t) j*n_kv;
        std::fill(x.m.begin() + r0, x.m.begin() + r0 + (size_t) (c.ctx - c.T + j + 1), zero);
    }
}

// scarto quadratico relativo dell'uscita GPU da un riferimento in double (softmax esatta sulle celle visibili)
double riferimento(const dati & x, const caso & c, const std::vector<float> & o) {
    std::vector<float> kf((size_t) c.ctx*D);
    ggml_fp16_to_fp32_row(x.k.data(), kf.data(), (int64_t) c.ctx*D);
    std::vector<double> l(c.ctx), acc(D);
    double se = 0.0, sr = 0.0;
    for (int j = 0; j < c.T; ++j) {
        const int ultima = c.ctx - c.T + j;
        for (int h = 0; h < H; ++h) {
            const float * qv = x.q.data() + ((size_t) h*c.T + j)*D;
            double mx = -1e300;
            for (int i = 0; i <= ultima; ++i) {
                const float * kv = kf.data() + (size_t) i*D;
                double s = 0.0;
                for (int d = 0; d < D; ++d) s += (double) qv[d]*kv[d];
                l[i] = s*SCALA;
                mx = std::max(mx, l[i]);
            }
            double somma = 0.0;
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int i = 0; i <= ultima; ++i) {
                const double p = std::exp(l[i] - mx);
                const float * kv = kf.data() + (size_t) i*D;
                somma += p;
                for (int d = 0; d < D; ++d) acc[d] += p*kv[d];
            }
            const float * og = o.data() + ((size_t) j*H + h)*D;  // uscita [D, H, T]
            for (int d = 0; d < D; ++d) {
                const double r = acc[d]/somma;
                se += (og[d] - r)*(og[d] - r);
                sr += r*r;
            }
        }
    }
    return std::sqrt(se/sr);
}

struct esito {
    int T = 0, ctx = 0, reps = 0, det = 0;
    long long n_kv = 0, nan = 0;
    double us_stima = 0, us_medio = 0, rif = -1;
    unsigned long long hash = 0;
};

int figlio(const std::string & dir, const char * tag, int giro, int device, double secondi, bool salva,
           bool solo_decode, bool con_rif) {
    ggml_backend_t gpu = ggml_backend_cuda_init(device, nullptr, nullptr);
    if (!gpu) { printf("FAIL backend CUDA %d\n", device); return 1; }
    char descr[256] = {0};
    ggml_backend_cuda_get_device_description(device, descr, sizeof(descr));
    const char * e1 = getenv(VAR_F1), * e2 = getenv(VAR_F2);
    printf("dispositivo %d: %s; %s=%s %s=%s\n", device, descr, VAR_F1, e1 ? e1 : "(spenta)", VAR_F2, e2 ? e2 : "(spenta)");
    FILE * fr = fopen(nome_ris(dir, tag, giro).c_str(), "w");
    if (!fr) { printf("FAIL file dei risultati\n"); return 1; }
    for (const caso & c : casi(solo_decode)) {
        const int64_t n_kv = (c.ctx + 255)/256*256, righe_m = GGML_PAD(c.T, GGML_KQ_MASK_PAD);
        dati x;
        genera(c, n_kv, righe_m, x);
        ggml_init_params ip = {8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, c.T, H);
        ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, n_kv);
        ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, righe_m);
        ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, k, m, SCALA, 0.0f, 0.0f);  // come GLM: K e V lo stesso tensore
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
        if (!buf) { printf("FAIL memoria GPU per T %d contesto %d\n", c.T, c.ctx); return 1; }
        ggml_backend_tensor_set(q, x.q.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(k, x.k.data(), 0, ggml_nbytes(k));
        ggml_backend_tensor_set(m, x.m.data(), 0, ggml_nbytes(m));

        // primo calcolo (carica il modulo CUDA): fissa l'uscita di questo processo
        if (ggml_backend_graph_compute(gpu, gf) != GGML_STATUS_SUCCESS) { printf("FAIL calcolo T %d\n", c.T); return 1; }
        std::vector<float> uscita(ggml_nelements(o)), dopo(ggml_nelements(o));
        ggml_backend_tensor_get(o, uscita.data(), 0, ggml_nbytes(o));
        esito e;
        e.T = c.T; e.ctx = c.ctx; e.n_kv = n_kv;
        e.hash = fnv(uscita.data(), ggml_nbytes(o));
        for (float v : uscita) e.nan += !std::isfinite(v);
        if (salva) {
            FILE * fb = fopen(nome_bin(dir, tag, c).c_str(), "wb");
            if (!fb || fwrite(uscita.data(), 1, ggml_nbytes(o), fb) != ggml_nbytes(o)) { printf("FAIL scrittura\n"); return 1; }
            fclose(fb);
        }
        if (con_rif && c.T <= 2 && c.ctx == CTX[0]) e.rif = riferimento(x, c, uscita);

        // secondo calcolo: stima del tempo per scegliere le ripetizioni; poi ripetizioni senza attese intermedie
        double t0 = ora();
        ggml_backend_graph_compute(gpu, gf);
        e.us_stima = (ora() - t0)*1e6;
        e.reps = std::max(3, std::min(2000, (int) (secondi*1e6/std::max(e.us_stima, 1.0))));
        t0 = ora();
        for (int r = 0; r < e.reps; ++r) ggml_backend_graph_compute_async(gpu, gf);
        ggml_backend_synchronize(gpu);
        e.us_medio = (ora() - t0)*1e6/e.reps;
        ggml_backend_tensor_get(o, dopo.data(), 0, ggml_nbytes(o));
        e.det = memcmp(uscita.data(), dopo.data(), ggml_nbytes(o)) == 0;

        fprintf(fr, "%d %d %lld %d %.3f %.3f %llx %d %lld %.6g\n", e.T, e.ctx, e.n_kv, e.reps, e.us_stima, e.us_medio,
                e.hash, e.det, e.nan, e.rif);
        fflush(fr);
        printf("T %4d contesto %6d (n_kv %6lld): %10.1f us medi su %4d chiamate%s%s\n", c.T, c.ctx, e.n_kv, e.us_medio,
               e.reps, e.det ? "" : ", USCITA CAMBIATA fra le ripetizioni", e.nan ? ", NaN o inf" : "");
        fflush(stdout);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    fclose(fr);
    ggml_backend_free(gpu);
    return 0;
}

std::vector<esito> leggi_ris(const std::string & nome) {
    std::vector<esito> v;
    FILE * f = fopen(nome.c_str(), "r");
    if (!f) return v;
    esito e;
    while (fscanf(f, "%d %d %lld %d %lf %lf %llx %d %lld %lf", &e.T, &e.ctx, &e.n_kv, &e.reps, &e.us_stima, &e.us_medio,
                  &e.hash, &e.det, &e.nan, &e.rif) == 10) {
        v.push_back(e);
    }
    fclose(f);
    return v;
}

// rilancia questo eseguibile con le variabili della configurazione; stdout e stderr del figlio vanno nel log
int lancia(const conf & cf, const std::vector<std::string> & argomenti, const std::string & log) {
    fflush(stdout);
    fflush(stderr);
    const pid_t pid = fork();
    if (pid < 0) { perror("fork"); return -1; }
    if (pid == 0) {
        unsetenv(VAR_F1);
        unsetenv(VAR_F2);
        if (cf.f1) setenv(VAR_F1, "1", 1);
        if (cf.f2) setenv(VAR_F2, "1", 1);
        const int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) { dup2(fd, 1); dup2(fd, 2); close(fd); }
        std::vector<char *> av;
        av.push_back((char *) "test-z6-fa-mla");
        for (const std::string & a : argomenti) av.push_back((char *) a.c_str());
        av.push_back(nullptr);
        execv("/proc/self/exe", av.data());
        _exit(127);
    }
    int st = 0;
    if (waitpid(pid, &st, 0) < 0) { perror("waitpid"); return -1; }
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
}

std::string leggi_testo(const std::string & nome) {
    std::string s;
    FILE * f = fopen(nome.c_str(), "r");
    if (!f) return s;
    char b[4096];
    size_t n;
    while ((n = fread(b, 1, sizeof(b), f)) > 0) s.append(b, n);
    fclose(f);
    return s;
}

struct confronto {
    bool letto = false, identico = false;
    long long diversi = 0, nan_diversi = 0;
    double max_diff = 0.0, max_val = 0.0;
};

confronto confronta(const std::string & a, const std::string & b) {
    confronto r;
    FILE * fa = fopen(a.c_str(), "rb");
    FILE * fb = fopen(b.c_str(), "rb");
    if (!fa || !fb) {
        if (fa) fclose(fa);
        if (fb) fclose(fb);
        return r;
    }
    std::vector<float> va(1 << 22), vb(1 << 22);
    bool uguali = true, ok = true;
    for (;;) {
        const size_t na = fread(va.data(), sizeof(float), va.size(), fa);
        const size_t nb = fread(vb.data(), sizeof(float), vb.size(), fb);
        if (na != nb) { ok = false; break; }
        if (na == 0) break;
        if (memcmp(va.data(), vb.data(), na*sizeof(float)) != 0) {
            uguali = false;
            for (size_t i = 0; i < na; ++i) {
                if (memcmp(&va[i], &vb[i], sizeof(float)) == 0) continue;
                ++r.diversi;
                if (!std::isfinite(va[i]) || !std::isfinite(vb[i])) { ++r.nan_diversi; continue; }
                r.max_diff = std::max(r.max_diff, (double) std::fabs(va[i] - vb[i]));
            }
        }
        for (size_t i = 0; i < na; ++i) {
            if (std::isfinite(va[i])) r.max_val = std::max(r.max_val, (double) std::fabs(va[i]));
        }
    }
    fclose(fa);
    fclose(fb);
    r.letto = ok;
    r.identico = ok && uguali;
    return r;
}

int padre(const std::string & dir_arg, int device, int giri, double secondi, bool solo_decode, bool tieni) {
    std::string dir = dir_arg;
    if (dir.empty()) {
        const char * t = getenv("TMPDIR");
        std::string modello = std::string(t && *t ? t : "/tmp") + "/test-z6-fa-mla-XXXXXX";
        std::vector<char> b(modello.begin(), modello.end());
        b.push_back(0);
        if (!mkdtemp(b.data())) { perror("mkdtemp"); return 1; }
        dir = b.data();
    } else {
        mkdir(dir.c_str(), 0755);
    }
    printf("test-z6-fa-mla: GPU %d, %d giri, %.2f s per caso, cartella %s\n", device, giri, secondi, dir.c_str());

    const std::vector<caso> lista = casi(solo_decode);
    const size_t nc = lista.size();
    std::vector<std::array<std::vector<esito>, 4>> ris(giri);
    std::vector<confronto> c_f1(nc), c_f2(nc), c_f1f2(nc);
    bool segno_f1[4] = {false}, segno_f2[4] = {false};
    std::vector<std::string> problemi;

    for (int g = 0; g < giri && problemi.empty(); ++g) {
        for (int s = 0; s < 4 && problemi.empty(); ++s) {
            const int ic = g % 2 == 0 ? s : 3 - s;  // giro dispari a ordine invertito
            const conf & cf = CONF[ic];
            std::vector<std::string> a = {"--figlio", cf.tag, "--giro", std::to_string(g), "--dir", dir,
                                          "--device", std::to_string(device), "--secondi", std::to_string(secondi)};
            if (g == 0) a.push_back("--salva");
            if (g == 0 && ic == 0) a.push_back("--rif");
            if (solo_decode) a.push_back("--solo-decode");
            const std::string log = dir + "/" + cf.tag + "-g" + std::to_string(g) + ".log";
            printf("\ngiro %d, configurazione %s (%s=%d %s=%d)\n", g + 1, cf.tag, VAR_F1, cf.f1, VAR_F2, cf.f2);
            const int rc = lancia(cf, a, log);
            const std::string testo = leggi_testo(log);
            size_t p0 = 0;
            while (p0 < testo.size()) {
                size_t p1 = testo.find('\n', p0);
                if (p1 == std::string::npos) p1 = testo.size();
                const std::string riga = testo.substr(p0, p1 - p0);
                if (riga.find("z6") != std::string::npos || riga.find("dispositivo") != std::string::npos ||
                    riga.find("FAIL") != std::string::npos || riga.find("T ") == 0 || rc != 0) {
                    printf("    %s\n", riga.c_str());
                }
                p0 = p1 + 1;
            }
            segno_f1[ic] = segno_f1[ic] || testo.find(SEGNO_F1) != std::string::npos;
            segno_f2[ic] = segno_f2[ic] || testo.find(SEGNO_F2) != std::string::npos;
            if (rc != 0) { problemi.push_back(std::string("figlio ") + cf.tag + " uscito con " + std::to_string(rc)); break; }
            ris[g][ic] = leggi_ris(nome_ris(dir, cf.tag, g));
            if (ris[g][ic].size() != nc) { problemi.push_back(std::string("risultati incompleti di ") + cf.tag); break; }
            if (g != 0) continue;
            // confronti byte per byte appena possibili, poi via i file che non servono più
            for (size_t i = 0; i < nc; ++i) {
                const caso & c = lista[i];
                if (ic == 1) c_f1[i] = confronta(nome_bin(dir, "base", c), nome_bin(dir, "f1", c));
                if (ic == 2) c_f2[i] = confronta(nome_bin(dir, "base", c), nome_bin(dir, "f2", c));
                if (ic == 3) c_f1f2[i] = confronta(nome_bin(dir, "f2", c), nome_bin(dir, "f1f2", c));
                if (tieni) continue;
                if (ic == 1) remove(nome_bin(dir, "f1", c).c_str());
                if (ic == 2) remove(nome_bin(dir, "base", c).c_str());
                if (ic == 3) { remove(nome_bin(dir, "f2", c).c_str()); remove(nome_bin(dir, "f1f2", c).c_str()); }
            }
        }
    }
    if (!problemi.empty()) {
        for (const std::string & p : problemi) printf("ERRORE: %s\n", p.c_str());
        printf("esito: FALLITO (log in %s)\n", dir.c_str());
        return 1;
    }

    bool f1_ok = true, f2_prefill_ok = true, f2_cambia = false, det_ok = true, nan_ok = true, rif_ok = true, file_ok = true;
    for (size_t i = 0; i < nc; ++i) file_ok = file_ok && c_f1[i].letto && c_f2[i].letto && c_f1f2[i].letto;
    auto stato = [](const confronto & c) { return !c.letto ? "MANCANTI" : c.identico ? "IDENTICHE" : "DIVERSE"; };
    printf("\n== F1 (%s): uscite confrontate byte per byte ==\n", VAR_F1);
    printf("   T  contesto  base / F1    F2 / F1+F2\n");
    for (size_t i = 0; i < nc; ++i) {
        f1_ok = f1_ok && c_f1[i].identico && c_f1f2[i].identico;
        printf("%4d  %8d  %-11s  %s\n", lista[i].T, lista[i].ctx, stato(c_f1[i]), stato(c_f1f2[i]));
    }
    printf("\n== F2 (%s): differenza dalla base ==\n", VAR_F2);
    printf("   T  contesto  elementi diversi  diff. massima  max |uscita|  relativa\n");
    for (size_t i = 0; i < nc; ++i) {
        const confronto & c = c_f2[i];
        if (!c.letto) {
            printf("%4d  %8d  uscite MANCANTI o di dimensione diversa\n", lista[i].T, lista[i].ctx);
            continue;
        }
        if (c.identico) {
            printf("%4d  %8d  identiche%s\n", lista[i].T, lista[i].ctx, lista[i].T > 2 ? " (F2 non agisce sopra 2 righe)" : "");
            continue;
        }
        if (lista[i].T > 2) f2_prefill_ok = false;
        f2_cambia = true;
        nan_ok = nan_ok && c.nan_diversi == 0;
        printf("%4d  %8d  %16lld  %13.3e  %12.3e  %8.2e%s\n", lista[i].T, lista[i].ctx, c.diversi, c.max_diff, c.max_val,
               c.max_val > 0 ? c.max_diff/c.max_val : 0.0, c.nan_diversi ? "  NaN/inf diversi" : "");
    }

    printf("\n== tempi medi in us per chiamata (media di %d giri; fra parentesi il rapporto con la base) ==\n", giri);
    printf("   T  contesto        base             F1             F2          F1+F2\n");
    for (size_t i = 0; i < nc; ++i) {
        double t[4] = {0, 0, 0, 0};
        for (int ic = 0; ic < 4; ++ic) {
            for (int g = 0; g < giri; ++g) {
                const esito & e = ris[g][ic][i];
                t[ic] += e.us_medio/giri;
                det_ok = det_ok && e.det && e.hash == ris[0][ic][i].hash;
                nan_ok = nan_ok && e.nan == 0;
            }
        }
        printf("%4d  %8d  %10.1f  %8.1f (%.3f)  %8.1f (%.3f)  %8.1f (%.3f)\n", lista[i].T, lista[i].ctx, t[0], t[1],
               t[1]/t[0], t[2], t[2]/t[0], t[3], t[3]/t[0]);
    }
    int n_rif = 0;
    for (const esito & e : ris[0][0]) {
        if (e.rif < 0) continue;
        ++n_rif;
        rif_ok = rif_ok && e.rif < SOGLIA_RIF;
        printf("riferimento in double, T %d contesto %d: scarto quadratico relativo %.2e (soglia %.2g)\n", e.T, e.ctx, e.rif,
               SOGLIA_RIF);
    }
    rif_ok = rif_ok && n_rif > 0;

    const bool segni_ok = segno_f1[1] && segno_f1[3] && !segno_f1[0] && !segno_f1[2] && !segno_f2[0] && !segno_f2[1];
    printf("\n== controlli ==\n");
    printf("riga «%s» nei log: f1 %s, f1f2 %s, base %s, f2 %s\n", SEGNO_F1, segno_f1[1] ? "sì" : "NO", segno_f1[3] ? "sì" : "NO",
           segno_f1[0] ? "SÌ (sbagliato)" : "no", segno_f1[2] ? "SÌ (sbagliato)" : "no");
    printf("riga «%s» nei log: f2 %s, f1f2 %s%s\n", SEGNO_F2, segno_f2[2] ? "sì" : "no", segno_f2[3] ? "sì" : "no",
           !segno_f2[2] && !f2_cambia ? " (F2 senza effetto: numero di SM già potenza di 2?)" : "");
    printf("stessa uscita a ogni ripetizione e fra i giri: %s\n", det_ok ? "sì" : "NO");
    printf("uscite salvate e rilette: %s\n", file_ok ? "tutte" : "ALCUNE MANCANTI");
    printf("NaN o inf: %s\n", nan_ok ? "nessuno" : "PRESENTI");
    printf("riferimento in double: %s\n", rif_ok ? "coerente" : "SCOSTAMENTO OLTRE SOGLIA");
    printf("F2 sopra 2 righe: %s\n", f2_prefill_ok ? "uscite identiche alla base" : "CAMBIA ANCHE IL PREFILL (sbagliato)");

    const bool tutto = f1_ok && f2_prefill_ok && det_ok && nan_ok && rif_ok && segni_ok && file_ok;
    if (tutto && !tieni) {
        for (int g = 0; g < giri; ++g) {
            for (const conf & cf : CONF) {
                remove(nome_ris(dir, cf.tag, g).c_str());
                remove((dir + "/" + cf.tag + "-g" + std::to_string(g) + ".log").c_str());
            }
        }
        rmdir(dir.c_str());
    } else {
        printf("risultati e log in %s%s\n", dir.c_str(), tieni ? "" : " (per tenere anche le uscite: --tieni)");
    }
    if (f1_ok) {
        printf("\nF1: uscite identiche in tutti i %zu casi, anche sopra la griglia di F2\n", nc);
    } else {
        printf("\nF1: USCITE DIVERSE\n");
    }
    double max_rel = 0.0;
    for (const confronto & c : c_f2) if (!c.identico && c.max_val > 0) max_rel = std::max(max_rel, c.max_diff/c.max_val);
    printf("F2: %s, differenza massima relativa %.2e (attesa diversa da zero: cambia i numeri, va validata con la KLD)\n",
           f2_cambia ? "cambia le uscite del decode" : "nessuna differenza", max_rel);
    printf("esito: %s\n", tutto ? "TUTTO IDENTICO" : "FALLITO");
    return tutto ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    int device = 0, giri = 2, giro = 0;
    double secondi = 0.4;
    bool solo_decode = false, tieni = false, salva = false, con_rif = false;
    std::string dir, tag;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) device = atoi(argv[++i]);
        else if (a == "--giri" && i + 1 < argc) giri = std::max(1, atoi(argv[++i]));
        else if (a == "--secondi" && i + 1 < argc) secondi = atof(argv[++i]);
        else if (a == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--solo-decode") solo_decode = true;
        else if (a == "--tieni") tieni = true;
        else if (a == "--figlio" && i + 1 < argc) tag = argv[++i];
        else if (a == "--giro" && i + 1 < argc) giro = atoi(argv[++i]);
        else if (a == "--salva") salva = true;
        else if (a == "--rif") con_rif = true;
        else {
            fprintf(stderr, "uso: %s [--device 0] [--giri 2] [--secondi 0.4] [--solo-decode] [--dir D] [--tieni]\n", argv[0]);
            return 1;
        }
    }
    if (!tag.empty()) {
        for (const conf & cf : CONF) {
            if (tag == cf.tag) return figlio(dir, cf.tag, giro, device, secondi, salva, solo_decode, con_rif);
        }
        fprintf(stderr, "configurazione sconosciuta: %s\n", tag.c_str());
        return 1;
    }
    return padre(dir, device, giri, secondi, solo_decode, tieni);
}
