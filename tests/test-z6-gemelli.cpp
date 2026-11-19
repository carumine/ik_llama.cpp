// Z6: esperti MoE freddi di due microbatch del prefill sulla CPU, da soli o come nodi gemelli (GGML_Z6_PP_GEMELLI=1).
// Costruisce un livello MoE con pesi casuali (up e gate IQ3_S, down IQ4_XS, forme di GLM-5.3-Flash) e due lotti di
// token con esperti scelti da una distribuzione sbilanciata, poi calcola up/gate e down dei due lotti nello stesso
// grafo CPU, nell'ordine [up/gate A, up/gate B, down A, down B]. Le variabili GGML_Z6_* si leggono una volta per
// processo: l'identità si prova lanciando il programma due volte (spento e acceso) e confrontando i file con cmp.
//
//   test-z6-gemelli --out <file> [--giri N] [--thread 40] [--cpu 0] [--tok0 1024] [--tok1 1024]
//                   [--embd 4096] [--ff 2048] [--esperti 288] [--usati 8] [--seme 1303] [--caldi 9]
//
// Scrive le quattro uscite (float grezzi) in <file> e stampa il tempo medio di un giro. Servono le variabili del
// percorso per righe (GGML_Z6_NUMA_EXPERTS=1 GGML_Z6_NUMA_ROWS=1 GGML_Z6_NUMA_QUEUE=1 GGML_Z6_BATCH_CHUNKS=640) e
// thread legati metà per socket: il programma li lega alle CPU --cpu, --cpu+1, ...
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct opzioni {
    std::string out;
    int giri = 1, thread = 40, cpu = 0, tok[2] = {1024, 1024};
    int embd = 4096, ff = 2048, esperti = 288, usati = 8, caldi = 9;
    unsigned seme = 1303;
};

// blocchi casuali validi: il primo campo dei due tipi è la scala in mezza precisione
void pesi_casuali(ggml_tensor * t, std::mt19937 & rng) {
    const size_t blocco = ggml_type_size(t->type), n = ggml_nbytes(t);
    auto p = (uint8_t *) t->data;
    for (size_t i = 0; i + 8 <= n; i += 8) { const uint64_t v = ((uint64_t) rng() << 32) | rng(); memcpy(p + i, &v, 8); }
    for (size_t i = n & ~(size_t) 7; i < n; ++i) p[i] = (uint8_t) rng();
    for (size_t i = 0; i + blocco <= n; i += blocco) {
        const uint16_t d = (uint16_t) (0x2400 + rng() % 0x0400);   // circa 0,016-0,031: nessun NaN né infinito
        memcpy(p + i, &d, 2);
    }
}

// usati esperti distinti per token. La popolarità di un esperto è log-normale (sigma circa 1), come nel prefill di GLM
// a 1024 token: mediana 17 token per esperto, un quinto degli esperti sotto gli 8 della soglia di conversione, pochi
// sopra 128. I due lotti condividono la parte `comune` della popolarità e ne hanno una propria, come due finestre
// vicine dello stesso testo.
void scegli(ggml_tensor * ids, const std::vector<double> & comune, int usati, int caldi, std::mt19937 & rng) {
    const int esperti = (int) comune.size();
    std::normal_distribution<double> normale(0.0, 1.0);
    std::vector<double> peso(esperti);
    for (int a = 0; a < esperti; ++a) peso[a] = std::exp(0.9 * comune[a] + 0.45 * normale(rng));
    std::discrete_distribution<int> dist(peso.begin(), peso.end());
    auto d = (int32_t *) ids->data;
    for (int64_t t = 0; t < ids->ne[1]; ++t) {
        int scelti[64];
        for (int j = 0; j < usati; ++j) {
            int a;
            bool nuovo;
            do {
                a = dist(rng);
                nuovo = true;
                for (int l = 0; l < j; ++l) nuovo = nuovo && scelti[l] != a;
            } while (!nuovo);
            scelti[j] = a;
            // i primi `caldi` esperti stanno sulla GPU: alla CPU arriva -1 e la riga va azzerata
            d[t * usati + j] = a < caldi ? -1 : a;
        }
    }
}

bool leggi(int argc, char ** argv, opzioni & o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char * { return i + 1 < argc ? argv[++i] : nullptr; };
        const char * v = nullptr;
        if      (a == "--out"     && (v = val())) o.out = v;
        else if (a == "--giri"    && (v = val())) o.giri = atoi(v);
        else if (a == "--thread"  && (v = val())) o.thread = atoi(v);
        else if (a == "--cpu"     && (v = val())) o.cpu = atoi(v);
        else if (a == "--tok0"    && (v = val())) o.tok[0] = atoi(v);
        else if (a == "--tok1"    && (v = val())) o.tok[1] = atoi(v);
        else if (a == "--embd"    && (v = val())) o.embd = atoi(v);
        else if (a == "--ff"      && (v = val())) o.ff = atoi(v);
        else if (a == "--esperti" && (v = val())) o.esperti = atoi(v);
        else if (a == "--usati"   && (v = val())) o.usati = atoi(v);
        else if (a == "--caldi"   && (v = val())) o.caldi = atoi(v);
        else if (a == "--seme"    && (v = val())) o.seme = (unsigned) strtoul(v, nullptr, 10);
        else { fprintf(stderr, "argomento non valido: %s\n", a.c_str()); return false; }
    }
    return !o.out.empty() && o.giri > 0 && o.thread > 0 && o.thread <= 256 && o.embd % 256 == 0 && o.ff % 256 == 0 &&
           o.esperti > o.usati && o.esperti <= 512 && o.usati > 0 && o.usati <= 64 && o.tok[0] > 0 && o.tok[1] > 0;
}

}

int main(int argc, char ** argv) {
    opzioni o;
    if (!leggi(argc, argv, o)) {
        fprintf(stderr, "uso: %s --out <file> [--giri N] [--thread 40] [--cpu 0] [--tok0 1024] [--tok1 1024] "
                "[--embd 4096] [--ff 2048] [--esperti 288] [--usati 8] [--caldi 9] [--seme 1303]\n", argv[0]);
        return 2;
    }
    ggml_time_init();
    const size_t pesi = (size_t) o.esperti * ((size_t) 2 * o.ff * (o.embd / 256) * ggml_type_size(GGML_TYPE_IQ3_S) +
                                              (size_t) o.embd * (o.ff / 256) * ggml_type_size(GGML_TYPE_IQ4_XS));
    size_t dati = pesi;
    for (int k = 0; k < 2; ++k) dati += (size_t) o.tok[k] * ((size_t) o.embd * 4 + (size_t) o.usati * 4 + (size_t) o.usati * (o.ff + o.embd) * 4);
    ggml_init_params ip = { dati + 64 * 1024 * 1024, nullptr, false };
    auto ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init fallita (%zu byte)\n", ip.mem_size); return 2; }

    std::mt19937 rng(o.seme);
    auto up   = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ3_S,  o.embd, o.ff,   o.esperti);
    auto gate = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ3_S,  o.embd, o.ff,   o.esperti);
    auto down = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ4_XS, o.ff,   o.embd, o.esperti);
    ggml_set_name(up, "blk.3.ffn_up_exps.weight");
    ggml_set_name(gate, "blk.3.ffn_gate_exps.weight");
    ggml_set_name(down, "blk.3.ffn_down_exps.weight");
    pesi_casuali(up, rng); pesi_casuali(gate, rng); pesi_casuali(down, rng);

    ggml_tensor * x[2], * ids[2], * ug[2], * dn[2];
    std::normal_distribution<float> normale(0.0f, 1.0f);
    std::vector<double> comune(o.esperti);
    for (auto & v : comune) v = normale(rng);
    for (int k = 0; k < 2; ++k) {
        x[k]   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, o.embd, 1, o.tok[k]);
        ids[k] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, o.usati, o.tok[k]);
        auto f = (float *) x[k]->data;
        for (int64_t i = 0; i < ggml_nelements(x[k]); ++i) f[i] = normale(rng);
        scegli(ids[k], comune, o.usati, o.caldi, rng);
    }
    auto gf = ggml_new_graph_custom(ctx, 64, false);
    for (int k = 0; k < 2; ++k) {
        ug[k] = ggml_moe_up_gate(ctx, up, gate, x[k], ids[k], GGML_UNARY_OP_SILU);
        ggml_build_forward_expand(gf, ug[k]);
    }
    for (int k = 0; k < 2; ++k) {
        dn[k] = ggml_mul_mat_id(ctx, down, ug[k], ids[k]);
        ggml_build_forward_expand(gf, dn[k]);
    }
    const int n_nodi = ggml_graph_n_nodes(gf);
    if (n_nodi != 4 || ggml_graph_node(gf, 0) != ug[0] || ggml_graph_node(gf, 1) != ug[1] ||
            ggml_graph_node(gf, 2) != dn[0] || ggml_graph_node(gf, 3) != dn[1]) {
        fprintf(stderr, "ordine dei nodi inatteso (%d nodi): i gemelli devono essere consecutivi\n", n_nodi);
        return 2;
    }

    std::vector<int32_t> cpu(o.thread);
    for (int i = 0; i < o.thread; ++i) cpu[i] = o.cpu + i;
    auto piano = ggml_graph_plan(gf, o.thread);
    std::vector<uint8_t> lavoro(piano.work_size + 64);
    piano.work_data = lavoro.data();
    piano.cpu_affinity = cpu.data();
    piano.n_cpu_affinity = o.thread;

    double somma_ms = 0, minimo_ms = 1e30;
    for (int g = 0; g < o.giri; ++g) {
        for (int k = 0; k < 2; ++k) {   // valori sentinella: una riga non calcolata si vede nel confronto
            std::fill_n((float *) ug[k]->data, ggml_nelements(ug[k]), 12345.0f);
            std::fill_n((float *) dn[k]->data, ggml_nelements(dn[k]), 54321.0f);
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_graph_compute(gf, &piano) != GGML_STATUS_SUCCESS) { fprintf(stderr, "calcolo fallito\n"); return 1; }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (g > 0 || o.giri == 1) { somma_ms += ms; minimo_ms = std::min(minimo_ms, ms); }
    }

    // un valore sentinella rimasto o un NaN indicano righe saltate o esperti letti male
    long long sentinelle = 0, non_finiti = 0;
    for (int k = 0; k < 2; ++k) {
        const auto d = (const int32_t *) ids[k]->data;
        for (ggml_tensor * t : { ug[k], dn[k] }) {
            const auto f = (const float *) t->data;
            const int64_t ne0 = t->ne[0];
            for (int64_t tk = 0; tk < t->ne[2]; ++tk) for (int j = 0; j < o.usati; ++j) {
                const float * r = f + (tk * o.usati + j) * ne0;
                if (d[tk * o.usati + j] < 0) continue;   // riga di un esperto caldo: azzerata in up/gate, non usata
                for (int64_t i = 0; i < ne0; ++i) {
                    sentinelle += r[i] == 12345.0f || r[i] == 54321.0f;
                    non_finiti += !std::isfinite(r[i]);
                }
            }
        }
    }

    FILE * fo = fopen(o.out.c_str(), "wb");
    if (!fo) { fprintf(stderr, "non posso scrivere %s\n", o.out.c_str()); return 2; }
    for (int k = 0; k < 2; ++k) for (ggml_tensor * t : { ug[k], dn[k] }) {
        const auto d = (const int32_t *) ids[k]->data;
        const auto f = (const float *) t->data;
        const int64_t ne0 = t->ne[0];
        // le righe dei caldi non sono definite nel down (le somma la GPU): fuori dal file
        for (int64_t r = 0; r < t->ne[1] * t->ne[2]; ++r) {
            if (t == dn[k] && d[r] < 0) continue;
            fwrite(f + r * ne0, sizeof(float), ne0, fo);
        }
    }
    fclose(fo);

    const int misurati = o.giri == 1 ? 1 : o.giri - 1;
    printf("test-z6-gemelli: gemelli=%s furto=%s ordine=%s, %d+%d token, %d thread, %d esperti: medio %.2f ms, minimo %.2f ms "
           "su %d giri; sentinelle %lld, non finiti %lld\n",
           getenv("GGML_Z6_PP_GEMELLI") ? getenv("GGML_Z6_PP_GEMELLI") : "0",
           getenv("GGML_Z6_PP_FURTO") ? getenv("GGML_Z6_PP_FURTO") : "0",
           getenv("GGML_Z6_PP_ORDINE") ? getenv("GGML_Z6_PP_ORDINE") : "0",
           o.tok[0], o.tok[1], o.thread, o.esperti, somma_ms / misurati, minimo_ms, misurati, sentinelle, non_finiti);
    ggml_free(ctx);
    return sentinelle || non_finiti ? 1 : 0;
}
