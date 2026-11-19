// Z6 (GGML_Z6_MTP_LANCI_NASCOSTI): the MTP pipeline pair of two one-token graphs through the real scheduler, with the
// real CUDA and CPU kernels, without a model. The process runs itself again (the variables are read once per process):
//   seriale:  GGML_Z6_PP_SOVRAPPOSTO_SERIALE=1, the first graph and then the second (control order of z6_pair_compute);
//   coppia:   the pipeline pair as before;
//   nascosti: the pipeline pair with GGML_Z6_MTP_LANCI_NASCOSTI=1;
// with one GPU and, when there are more, with the layers in turn on every GPU (copies between the GPUs).
// Each graph has 8 layers like a GLM layer reduced to its skeleton: an "attention" that reads and rewrites in place a
// recurrent state shared by the two graphs (only stream order protects it, as the KDA state of the model), a hot piece
// on the GPU, the cold experts on the CPU staged in a pinned packet shared by the layers of the slot (z6_caldi_prepare,
// tickets, last_upload), the weighted sum on the GPU; layer 4 stays outside the plan, its experts on the CPU behind the
// scheduler's copies, so its segment must run in series. `between` copies every state after the first graph, as
// z6_per_step_capture does. Each child computes 24 pairs with fresh inputs and prints the FNV-1a fingerprint of the
// outputs of both graphs, of the states after the pair and of the copies `between` took; the parent compares them.
// The GGML_Z6_* variables of the CPU kernels come from the environment: thread 0 is free in the cold pieces only with
// those of the preset, so run it with them (on the Z6, under the GPU lock), on one line:
//   flock ~/Workspace/z6/.gpu-lock env GGML_Z6_NUMA_EXPERTS=1 GGML_Z6_NUMA_ROWS=1 GGML_Z6_NUMA_AFFINITY=1
//     GGML_Z6_NUMA_BARRIER=1 GGML_Z6_NUMA_UNIT=8 GGML_Z6_FREDDI_CODA=16 GGML_Z6_FREDDI_CODA_RIF=4
//     GGML_Z6_FREDDI_CODA_RIGHE=2 GGML_Z6_PRINCIPALE_LIBERO=1 GGML_Z6_H2D_ASYNC=1 GGML_Z6_GRAFI_EPOCA=1
//     numactl --physcpubind=0-39 --localalloc build-cuda/bin/test-z6-lanci-nascosti 40
// Exit 0 with "TUTTO IDENTICO" when the three fingerprints match for every GPU count and, with thread 0 free, the
// nascosti child hid segments and accounted for all of them; 1 otherwise.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

constexpr int n_embd = 1024, n_ff = 512, n_exp = 32, n_used = 8, n_out = 512, n_layer = 8, senza_caldi = 4;
constexpr int n_gruppi = n_layer - 1, n_coppie = 24;
constexpr size_t pacchetto = 1 << 20, off_x = 0, off_ids = 64 << 10, off_w = 96 << 10, off_out = 128 << 10;

void check(bool ok, const char * why) {
    if (!ok) { printf("FAIL %s\n", why); fflush(stdout); std::exit(1); }
}
uint64_t fnv1a(uint64_t h, const void * p, size_t n) {
    for (size_t i = 0; i < n; ++i) { h ^= ((const unsigned char *) p)[i]; h *= 1099511628211ull; }
    return h;
}
struct pin_free { void operator()(void * p) const { if (p) ggml_backend_cuda_z6_host_free(p); } };

void quantized(ggml_tensor * t, std::mt19937 & rng, float scale) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> f(ggml_nelements(t));
    for (auto & v : f) v = d(rng);
    std::vector<uint8_t> q(ggml_nbytes(t));
    ggml_quantize_chunk(t->type, f.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr, nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

// weights: the experts in RAM, shared by the layers; per layer an "attention" and a "hot" matrix on its GPU, the
// recurrent state both graphs rewrite in place and the copy `between` takes of it
struct pesi {
    ggml_context_ptr cpu_ctx;
    ggml_backend_buffer_ptr cpu_buf;
    ggml_tensor * up = nullptr, * gate = nullptr, * down = nullptr;
    std::vector<ggml_context_ptr> ctx;
    std::vector<ggml_backend_buffer_ptr> buf;
    ggml_tensor * wa[n_layer] = {}, * wh[n_layer] = {}, * wo = nullptr;
    ggml_tensor * stato[n_layer] = {}, * copia[n_layer] = {};
};

void crea_pesi(pesi & p, ggml_backend_t * gpu, int n_gpu, ggml_backend_t cpu, std::mt19937 & rng) {
    p.cpu_ctx.reset(ggml_init({4*ggml_tensor_overhead(), nullptr, true}));
    p.up   = ggml_new_tensor_3d(p.cpu_ctx.get(), GGML_TYPE_Q8_0, n_embd, n_ff, n_exp);
    p.gate = ggml_new_tensor_3d(p.cpu_ctx.get(), GGML_TYPE_Q8_0, n_embd, n_ff, n_exp);
    p.down = ggml_new_tensor_3d(p.cpu_ctx.get(), GGML_TYPE_Q8_0, n_ff, n_embd, n_exp);
    p.cpu_buf.reset(ggml_backend_alloc_ctx_tensors(p.cpu_ctx.get(), cpu));
    check(bool(p.cpu_buf), "esperti in RAM");
    ggml_backend_buffer_set_usage(p.cpu_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    quantized(p.up, rng, 0.05f); quantized(p.gate, rng, 0.05f); quantized(p.down, rng, 0.05f);
    for (int g = 0; g < n_gpu; ++g) {
        // weights and states in separate buffers: the states are not weights for the scheduler (like the KV cache)
        ggml_context_ptr cw(ggml_init({64*ggml_tensor_overhead(), nullptr, true}));
        ggml_context_ptr cs(ggml_init({64*ggml_tensor_overhead(), nullptr, true}));
        for (int l = g; l < n_layer; l += n_gpu) {
            p.wa[l] = ggml_new_tensor_2d(cw.get(), GGML_TYPE_Q8_0, n_embd, n_embd);
            p.wh[l] = ggml_new_tensor_2d(cw.get(), GGML_TYPE_Q8_0, n_embd, n_embd);
            p.stato[l] = ggml_new_tensor_2d(cs.get(), GGML_TYPE_F32, n_embd, 1);
            p.copia[l] = ggml_new_tensor_2d(cs.get(), GGML_TYPE_F32, n_embd, 1);
        }
        if ((n_layer - 1) % n_gpu == g) p.wo = ggml_new_tensor_2d(cw.get(), GGML_TYPE_Q8_0, n_embd, n_out);
        ggml_backend_buffer_ptr bw(ggml_backend_alloc_ctx_tensors(cw.get(), gpu[g]));
        ggml_backend_buffer_ptr bs(ggml_backend_alloc_ctx_tensors(cs.get(), gpu[g]));
        check(bw && bs, "pesi e stati in VRAM");
        ggml_backend_buffer_set_usage(bw.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        p.ctx.push_back(std::move(cw)); p.ctx.push_back(std::move(cs));
        p.buf.push_back(std::move(bw)); p.buf.push_back(std::move(bs));
    }
    for (int l = 0; l < n_layer; ++l) { quantized(p.wa[l], rng, 0.03f); quantized(p.wh[l], rng, 0.03f); }
    quantized(p.wo, rng, 0.03f);
}

// one token of the pair: its graph and scheduler, its inputs on the GPUs, its pinned packet
struct slot_t {
    ggml_backend_sched_ptr sched;
    std::unique_ptr<void, pin_free> pinned;
    ggml_backend_buffer_ptr host;
    std::vector<ggml_context_ptr> in_ctx;
    std::vector<ggml_backend_buffer_ptr> in_buf;
    ggml_tensor * inp = nullptr, * ids[n_layer] = {}, * w[n_layer] = {};
    ggml_context_ptr ctx;
    ggml_cgraph * g = nullptr;
    ggml_tensor * out = nullptr;
};

void imposta_unario(ggml_tensor * par) {
    // the values the model writes: swiglu_limit(il, false), situ_beta, situ_linear_beta (0 for GLM)
    *(float *)(par->op_params + 1) = 0.0f;
    *(float *)(par->op_params + 2) = 0.0f;
    *(float *)(par->op_params + 3) = 0.0f;
}

void costruisci(slot_t & s, const pesi & p, ggml_backend_t * gpu, int n_gpu, ggml_backend_t cpu) {
    std::vector<ggml_backend_t> be(gpu, gpu + n_gpu);
    be.push_back(cpu);
    s.sched.reset(ggml_backend_sched_new(be.data(), nullptr, (int) be.size(), 2048, false));
    ggml_backend_sched_z6_begin_graph(s.sched.get());
    for (int g = 0; g < n_gpu; ++g) {
        ggml_context_ptr c(ggml_init({64*ggml_tensor_overhead(), nullptr, true}));
        if (g == 0) s.inp = ggml_new_tensor_2d(c.get(), GGML_TYPE_F32, n_embd, 1);
        for (int l = g; l < n_layer; l += n_gpu) {
            s.ids[l] = ggml_new_tensor_2d(c.get(), GGML_TYPE_I32, n_used, 1);
            s.w[l] = ggml_new_tensor_3d(c.get(), GGML_TYPE_F32, 1, n_used, 1);
        }
        ggml_backend_buffer_ptr b(ggml_backend_alloc_ctx_tensors(c.get(), gpu[g]));
        check(bool(b), "ingressi in VRAM");
        s.in_ctx.push_back(std::move(c)); s.in_buf.push_back(std::move(b));
    }
    s.pinned.reset(ggml_backend_cuda_z6_host_alloc(pacchetto));
    check(bool(s.pinned), "pacchetto pinned");
    s.host.reset(ggml_backend_cpu_buffer_from_ptr(s.pinned.get(), pacchetto));
    s.ctx.reset(ggml_init({1024*ggml_tensor_overhead() + ggml_graph_overhead_custom(2048, false), nullptr, true}));
    auto ctx = s.ctx.get();
    // the packet is shared by the layers of the slot, as in llama_z6_caldi::staging
    auto staged = [&](ggml_tensor * t, size_t offset) {
        t->buffer = s.host.get(); t->data = (char *) s.pinned.get() + offset;
        return t;
    };
    std::vector<std::pair<ggml_tensor *, ggml_backend_t>> posto;
    auto su = [&](ggml_tensor * t, ggml_backend_t b) { posto.emplace_back(t, b); return t; };
    s.g = ggml_new_graph_custom(ctx, 2048, false);
    ggml_tensor * cur = s.inp;
    for (int l = 0; l < n_layer; ++l) {
        ggml_backend_t gl = gpu[l % n_gpu];
        auto cn = su(ggml_rms_norm(ctx, cur, 1e-6f), gl);
        auto att = su(ggml_mul_mat(ctx, p.wa[l], cn), gl);
        auto st = su(ggml_add(ctx, att, p.stato[l]), gl);
        // the recurrent state, rewritten in place: the other graph reads it at the same layer after this one
        ggml_build_forward_expand(s.g, su(ggml_cpy(ctx, st, p.stato[l]), gl));
        auto x = su(ggml_add(ctx, cur, su(ggml_scale(ctx, st, 0.25f), gl)), gl);
        auto x3 = ggml_reshape_3d(ctx, x, n_embd, 1, 1);
        ggml_build_forward_expand(s.g, x3);
        if (l == senza_caldi) {
            // outside the plan: the scheduler copies x and the IDs to the CPU, the CPU runs the experts
            auto par = su(ggml_moe_up_gate(ctx, p.up, p.gate, x3, s.ids[l], GGML_UNARY_OP_SILU), cpu);
            imposta_unario(par);
            auto down = su(ggml_mul_mat_id(ctx, p.down, par, s.ids[l]), cpu);
            auto sum = su(ggml_mul_multi_add_id(ctx, down, s.w[l], s.ids[l]), gl);
            cur = su(ggml_add(ctx, sum, x), gl);
            ggml_build_forward_expand(s.g, cur);
            continue;
        }
        auto cpu_x = staged(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, 1), off_x);
        auto cpu_ids = staged(ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_used, 1, 1), off_ids);
        auto cpu_w = staged(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, 1), off_w);
        // the hot piece: its first node starts a split, z6_caldi_prepare stages x, IDs and weights before it
        auto hs = su(ggml_mul_mat(ctx, p.wh[l], x), gl);
        auto hot = su(ggml_silu(ctx, hs), gl);
        ggml_build_forward_expand(s.g, hot);
        auto par = su(ggml_moe_up_gate(ctx, p.up, p.gate, cpu_x, cpu_ids, GGML_UNARY_OP_SILU), cpu);
        imposta_unario(par);
        auto down = staged(su(ggml_mul_mat_id(ctx, p.down, par, cpu_ids), cpu), off_out);
        const ggml_z6_caldi_group group = { hs, hot, par, down, {x3, s.ids[l], s.w[l]}, {cpu_x, cpu_ids, cpu_w}, l,
                                            nullptr, nullptr };
        ggml_backend_sched_z6_add_group(s.sched.get(), &group);
        auto sum = su(ggml_mul_multi_add_id(ctx, down, s.w[l], s.ids[l]), gl);
        cur = su(ggml_add(ctx, su(ggml_add(ctx, sum, hot), gl), x), gl);
        ggml_build_forward_expand(s.g, cur);
    }
    ggml_backend_t ultima = gpu[(n_layer - 1) % n_gpu];
    s.out = su(ggml_mul_mat(ctx, p.wo, su(ggml_rms_norm(ctx, cur, 1e-6f), ultima)), ultima);
    ggml_build_forward_expand(s.g, s.out);
    for (auto & [t, b] : posto) ggml_backend_sched_set_tensor_backend(s.sched.get(), t, b);
    check(ggml_backend_sched_alloc_graph(s.sched.get(), s.g), "grafo allocato");
}

struct cattura_t {
    const pesi * p = nullptr;
    std::vector<ggml_backend_t> gpu;   // layer l on gpu[l % gpu.size()]
    int chiamate = 0;
};
// as z6_per_step_capture: the states of the layers (from, to] after the first graph, on the stream of their GPU
void tra(int from, int to, void * ud) {
    auto c = (cattura_t *) ud;
    c->chiamate++;
    for (int l = std::max(0, from + 1); l <= to && l < n_layer; ++l) {
        ggml_backend_t b = c->gpu[l % c->gpu.size()];
        ggml_backend_tensor_copy_async(b, b, c->p->stato[l], c->p->copia[l]);
    }
}

int figlio(const std::string & modo, int n_gpu, int threads) {
    std::vector<ggml_backend_t> gpu(n_gpu);
    for (int g = 0; g < n_gpu; ++g) {
        gpu[g] = ggml_backend_cuda_init(g, nullptr, nullptr);
        check(gpu[g] != nullptr, "backend CUDA");
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    check(cpu != nullptr, "backend CPU");
    ggml_backend_cpu_set_n_threads(cpu, threads);
    if (threads >= 2 && (int) std::thread::hardware_concurrency() >= threads) {
        // as --cpu-range 0-<threads-1> of the preset: worker i on CPU i (the local path wants half per socket)
        std::vector<int32_t> cpus(threads);
        for (int i = 0; i < threads; ++i) cpus[i] = i;
        ggml_backend_cpu_set_cpu_affinity(cpu, cpus.data(), threads);
    }
    std::mt19937 rng(1511);
    pesi p;
    crea_pesi(p, gpu.data(), n_gpu, cpu, rng);
    slot_t sl[2];   // the two tokens of the pair: x (sampled) and d (its draft)
    for (auto & s : sl) costruisci(s, p, gpu.data(), n_gpu, cpu);
    cattura_t cattura;
    cattura.p = &p; cattura.gpu = gpu;

    uint64_t h = 1469598103934665603ull;
    int diverse = 0;   // layers whose state the second graph changed after the copy: the test sees the order
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int k = 0; k < n_coppie; ++k) {
        for (auto & s : sl) {
            std::vector<float> v(n_embd);
            for (auto & a : v) a = nd(rng);
            v[3] = 0.0f; v[5] = -0.0f;
            ggml_backend_tensor_set(s.inp, v.data(), 0, v.size()*sizeof(float));
            for (int l = 0; l < n_layer; ++l) {
                std::vector<int32_t> perm(n_exp);
                for (int e = 0; e < n_exp; ++e) perm[e] = e;
                std::shuffle(perm.begin(), perm.end(), rng);
                std::vector<int32_t> ids(perm.begin(), perm.begin() + n_used);
                // as the cold IDs of the model: the hot experts of the token are -1 (the layer outside the plan has none)
                if (l != senza_caldi) { ids[1] = -1; ids[6] = -1; }
                std::vector<float> w(n_used);
                float tot = 0.0f;
                for (auto & a : w) { a = std::abs(nd(rng)) + 0.01f; tot += a; }
                for (auto & a : w) a /= tot;
                ggml_backend_tensor_set(s.ids[l], ids.data(), 0, ids.size()*sizeof(int32_t));
                ggml_backend_tensor_set(s.w[l], w.data(), 0, w.size()*sizeof(float));
            }
        }
        for (int l = 0; l < n_layer; ++l) {
            std::vector<float> v(n_embd);
            for (auto & a : v) a = 0.5f*nd(rng);
            ggml_backend_tensor_set(p.stato[l], v.data(), 0, v.size()*sizeof(float));
            std::fill(v.begin(), v.end(), 0.0f);
            ggml_backend_tensor_set(p.copia[l], v.data(), 0, v.size()*sizeof(float));
        }
        const auto st = ggml_backend_sched_z6_pair_compute_pipeline(sl[0].sched.get(), sl[1].sched.get(), tra, &cattura);
        check(st == GGML_STATUS_SUCCESS, "coppia della pipeline eseguita (GGML_Z6_PP_SOVRAPPOSTO=1 e grafi ammessi)");
        for (auto & s : sl) {
            std::vector<float> o(n_out);
            ggml_backend_tensor_get(s.out, o.data(), 0, o.size()*sizeof(float));
            for (float a : o) check(std::isfinite(a), "uscita finita");
            h = fnv1a(h, o.data(), o.size()*sizeof(float));
        }
        for (int l = 0; l < n_layer; ++l) {
            std::vector<float> a(n_embd), b(n_embd);
            ggml_backend_tensor_get(p.stato[l], a.data(), 0, a.size()*sizeof(float));
            ggml_backend_tensor_get(p.copia[l], b.data(), 0, b.size()*sizeof(float));
            diverse += memcmp(a.data(), b.data(), a.size()*sizeof(float)) != 0;
            h = fnv1a(h, a.data(), a.size()*sizeof(float));
            h = fnv1a(h, b.data(), b.size()*sizeof(float));
        }
    }
    int64_t nascosti = -1, serie = -1;
    ggml_backend_sched_z6_lanci_nascosti(&nascosti, &serie);
    check(diverse == n_coppie*n_layer, "il secondo grafo riscrive ogni stato dopo la copia di between");
    const int attese = modo == "seriale" ? n_coppie : n_coppie*(n_gruppi + 1);
    check(cattura.chiamate == attese, "chiamate di between come attese");
    printf("figlio %s, %d GPU, %d thread: between %d, segmenti nascosti %" PRId64 ", in serie %" PRId64 "\n",
           modo.c_str(), n_gpu, threads, cattura.chiamate, nascosti, serie);
    printf("impronta=%016" PRIx64 " nascosti=%" PRId64 " serie=%" PRId64 "\n", h, nascosti, serie);
    fflush(stdout);
    for (auto & s : sl) ggml_backend_sched_reset(s.sched.get());
    for (auto & s : sl) s = slot_t();
    p = pesi();
    ggml_backend_free(cpu);
    for (int g = 0; g < n_gpu; ++g) ggml_backend_free(gpu[g]);
    return 0;
}

// spento: the child printed the scheduler's line "il thread 0 non era libero", the lever switched itself off
struct esito_t { bool ok = false, spento = false; std::string impronta; int64_t nascosti = -1, serie = -1; };

esito_t rilancia(const std::string & exe, const std::string & modo, int n_gpu, int threads) {
    std::string cmd = "env -u GGML_Z6_MTP_LANCI_NASCOSTI -u GGML_Z6_PP_SOVRAPPOSTO_SERIALE GGML_Z6_PP_SOVRAPPOSTO=1 ";
    if (modo == "seriale") cmd += "GGML_Z6_PP_SOVRAPPOSTO_SERIALE=1 ";
    if (modo == "nascosti") cmd += "GGML_Z6_MTP_LANCI_NASCOSTI=1 ";
    cmd += "'" + exe + "' --figlio " + modo + " " + std::to_string(n_gpu) + " " + std::to_string(threads) + " 2>&1";
    esito_t e;
    FILE * f = popen(cmd.c_str(), "r");
    if (!f) return e;
    char b[4096];
    while (fgets(b, sizeof(b), f)) {
        const std::string riga = b;
        if (riga.rfind("impronta=", 0) == 0) {
            char hex[32] = {};
            long long n = -1, s = -1;
            if (sscanf(b, "impronta=%31s nascosti=%lld serie=%lld", hex, &n, &s) == 3) {
                e.impronta = hex; e.nascosti = n; e.serie = s;
            }
        }
        if (riga.find("il thread 0 non era libero") != std::string::npos) e.spento = true;
        // the child's own lines, the scheduler's lines about the pair included
        if (riga.rfind("figlio ", 0) == 0 || riga.rfind("FAIL", 0) == 0 || riga.find("z6 lanci nascosti") != std::string::npos ||
                riga.find("z6 pipeline") != std::string::npos) fputs(b, stdout);
    }
    e.ok = pclose(f) == 0 && !e.impronta.empty();
    return e;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc >= 5 && std::string(argv[1]) == "--figlio") return figlio(argv[2], atoi(argv[3]), atoi(argv[4]));
    const int threads = argc > 1 ? atoi(argv[1]) : 8;
    if (threads < 1) { fprintf(stderr, "uso: %s [thread]\n", argv[0]); return 1; }
    char exe[4096];
    const ssize_t k = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (k <= 0) { printf("FAIL /proc/self/exe\n"); return 1; }
    exe[k] = 0;
    const int n_dev = ggml_backend_cuda_get_device_count();
    if (n_dev < 1) { printf("FAIL nessuna GPU CUDA\n"); return 1; }
    // thread 0 is free in the cold pieces only with the variables of the preset (see the top of this file)
    const bool libero = ggml_z6_principale_libero_attivo();
    bool tutto = true;
    // one GPU, then every GPU with the layers in turn on each (copies between the GPUs, as the model split by layers)
    std::vector<int> prove = {1};
    if (n_dev > 1) prove.push_back(std::min(n_dev, n_layer));
    for (int n_gpu : prove) {
        const esito_t ser = rilancia(exe, "seriale", n_gpu, threads);
        const esito_t cop = rilancia(exe, "coppia", n_gpu, threads);
        const esito_t nas = rilancia(exe, "nascosti", n_gpu, threads);
        bool ok = ser.ok && cop.ok && nas.ok;
        if (!ok) printf("%d GPU: un figlio è fallito\n", n_gpu);
        if (ok && (ser.impronta != cop.impronta || cop.impronta != nas.impronta)) {
            printf("%d GPU: DIVERSO: seriale %s, coppia %s, nascosti %s\n", n_gpu, ser.impronta.c_str(), cop.impronta.c_str(),
                   nas.impronta.c_str());
            ok = false;
        }
        if (ok && (ser.nascosti != 0 || ser.serie != 0 || cop.nascosti != 0 || cop.serie != 0)) {
            printf("%d GPU: contatori dei lanci nascosti non nulli senza la variabile\n", n_gpu);
            ok = false;
        }
        const int64_t segmenti = int64_t(n_coppie)*2*n_gruppi;
        if (ok && libero) {
            // every segment of every pair counted once and at least one hidden. The sum alone does not show that the
            // lever stayed on (a segment not taken and those after the switch-off count in series), so the switch-off
            // line of the scheduler fails the test by itself. The bound on the segments in series is on the total:
            // the two of the layer outside the plan and the last one, 3 per pair, every pair having the same splits
            if (nas.spento) {
                printf("%d GPU: LA LEVA SI È SPENTA: il thread 0 non era libero in un pezzo freddo\n", n_gpu);
                ok = false;
            } else if (nas.nascosti <= 0 || nas.nascosti + nas.serie != segmenti || nas.serie < 3*n_coppie) {
                printf("%d GPU: NESSUN SEGMENTO NASCOSTO O CONTI SBAGLIATI: nascosti %" PRId64 ", in serie %" PRId64
                       " su %" PRId64 " (il thread 0 era libero? servono anche GGML_Z6_NUMA_EXPERTS/ROWS/AFFINITY e"
                       " i thread su due nodi)\n", n_gpu, nas.nascosti, nas.serie, segmenti);
                ok = false;
            }
        } else if (ok) {
            printf("%d GPU: thread 0 non libero per configurazione (GGML_Z6_PRINCIPALE_LIBERO=1 e GGML_Z6_FREDDI_CODA"
                   " mancano): controllata solo l'identità\n", n_gpu);
        }
        printf("%d GPU: %s (impronta %s; nascosti %" PRId64 ", in serie %" PRId64 " su %" PRId64 " segmenti)\n", n_gpu,
               ok ? "identico" : "ERRORE", nas.impronta.c_str(), nas.nascosti, nas.serie, segmenti);
        tutto &= ok;
    }
    printf("%s\n", tutto ? "TUTTO IDENTICO" : "DIFFERENZE O ERRORI: vedi sopra");
    return tutto ? 0 : 1;
}
