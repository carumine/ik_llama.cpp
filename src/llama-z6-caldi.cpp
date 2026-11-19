#include "llama-z6-caldi.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "ggml-cpp.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <tuple>

namespace {
uint64_t env_integer(const char * name, uint64_t fallback) {
    const char * s = std::getenv(name);
    if (!s) return fallback;
    char * end = nullptr;
    errno = 0;
    unsigned long long value = std::strtoull(s, &end, 10);
    if (!*s || *s == '-' || errno || !end || *end) throw std::runtime_error(std::string(name) + " non valido");
    return value;
}

std::vector<ggml_tensor *> weights(const llama_layer & l) {
    if (l.ffn_up_gate_exps) return {l.ffn_up_gate_exps, l.ffn_down_exps};
    return {l.ffn_up_exps, l.ffn_gate_exps, l.ffn_down_exps};
}

// GGML_Z6_IQ3S_LUT=1: in RAM the IQ3_S experts are reordered (GGML_TYPE_IQ3_S_Z6, 134 bytes per block). The banks in
// VRAM keep IQ3_S, the only layout the CUDA kernels read: every expert that goes to a bank passes through the inverse
// of the reordering, which rebuilds the bytes of the file.
ggml_type bank_type(const ggml_tensor * src) { return src->type == GGML_TYPE_IQ3_S_Z6 ? GGML_TYPE_IQ3_S : src->type; }
size_t bank_expert_bytes(const ggml_tensor * src) { return ggml_row_size(bank_type(src), src->ne[0])*src->ne[1]; }
// One expert of src in the layout of the bank: src itself, or its restored copy in tmp
const void * bank_expert(const ggml_tensor * src, int expert, std::vector<uint8_t> & tmp) {
    const char * data = (const char *)src->data + (size_t)expert*src->nb[2];
    if (src->type != GGML_TYPE_IQ3_S_Z6) return data;
    const size_t bytes = bank_expert_bytes(src);
    tmp.resize(bytes);
    if (!ggml_z6_iq3s_lut_to_iq3s(data, tmp.data(), (int64_t)(bytes/ggml_type_size(GGML_TYPE_IQ3_S))))
        throw std::runtime_error("z6 caldi: esperto riordinato non riportabile a IQ3_S");
    return tmp.data();
}
}

bool llama_z6_caldi::requested() {
    return std::getenv("GGML_Z6_CALDI_FILE") || std::getenv("GGML_Z6_CALDI_MIB") || std::getenv("GGML_Z6_CALDI_DEVICE");
}

int llama_z6_caldi::device() {
    const uint64_t id = env_integer("GGML_Z6_CALDI_DEVICE", 0);
    if (id > INT32_MAX) throw std::runtime_error("GGML_Z6_CALDI_DEVICE troppo grande");
    return (int)id;
}

struct llama_z6_caldi::impl {
    struct pinned_deleter {
        void operator()(void * ptr) const {
#ifdef GGML_USE_CUDA
            if (ptr) ggml_backend_cuda_z6_host_free(ptr);
#else
            (void)ptr;
#endif
        }
    };
    struct layer {
        std::vector<ggml_tensor *> sources;
        std::vector<int32_t> selected;
        ggml_tensor * hot_map = nullptr;
        ggml_tensor * cold_map = nullptr;
        // GGML_Z6_CALDI_DINAMICI: mappe lato host, proprietario di ogni slot, ultimo token in cui l'esperto è stato freddo
        std::vector<int32_t> hot, cold, owner, seen;
        size_t next_slot = 0, slots_fissi = 0;
        bool dirty = false;
        std::vector<double> score;
        std::vector<int64_t> score_token;
        int64_t observed = 0, cold_uses = 0, loads = 0, rejected = 0;
    };
    struct pending_upload { int il, expert, slot; };
    // Le copie grandi occupano il motore di copia e fanno aspettare quelle piccole del calcolo: un thread proprio,
    // sul fratello HT dell'ultimo core, le manda a pezzi piccoli
    // tensor/expert set: source reordered by GGML_Z6_IQ3S_LUT, restored by the upload thread before the copy
    struct upload_job { ggml_backend_t gpu; ggml_tensor * dst; size_t offset; const char * src; size_t size;
                        const ggml_tensor * lut_src = nullptr; int lut_expert = 0; };
    static upload_job make_job(ggml_backend_t gpu, ggml_tensor * dst, const ggml_tensor * src, int expert, int slot) {
        upload_job j{gpu, dst, (size_t)slot*dst->nb[2], (const char *)src->data + (size_t)expert*src->nb[2], (size_t)dst->nb[2]};
        if (src->type == GGML_TYPE_IQ3_S_Z6) { j.lut_src = src; j.lut_expert = expert; }
        return j;
    }
    std::thread uploader;
    std::mutex upload_m;
    std::condition_variable upload_cv;
    std::deque<upload_job> upload_q;
    std::atomic<int64_t> upload_left{0};
    bool upload_stop = false;
    size_t upload_chunk = 256*1024;
    std::vector<uint8_t> upload_tmp;   // upload thread only
    void upload_loop() {
#ifdef GGML_USE_CUDA
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(79, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        for (;;) {
            upload_job j;
            {
                std::unique_lock<std::mutex> lock(upload_m);
                upload_cv.wait(lock, [&] { return upload_stop || !upload_q.empty(); });
                if (upload_q.empty()) return;
                j = upload_q.front();
                upload_q.pop_front();
            }
            // each piece is awaited before the next one: the restored copy outlives its uploads
            if (j.lut_src) j.src = (const char *)bank_expert(j.lut_src, j.lut_expert, upload_tmp);
            for (size_t done = 0; done < j.size; done += upload_chunk) {
                ggml_backend_cuda_z6_carica_async(j.gpu, j.dst, j.offset + done, j.src + done, std::min(upload_chunk, j.size - done));
                ggml_backend_cuda_z6_carica_attendi(j.gpu);
            }
            upload_left.fetch_sub(1);
        }
#endif
    }
    void report_usage() const {
        if (!class_debug || !token) return;
        int64_t calls = 0, cold = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            const auto & l = layers[il];
            if (!l.observed) continue;
            calls += l.observed; cold += l.cold_uses;
            LLAMA_LOG_INFO("z6 classlru livello: L=%zu chiamate=%lld freddi=%lld caricamenti=%lld rifiuti=%lld\n",
                    il, (long long)l.observed, (long long)l.cold_uses, (long long)l.loads, (long long)l.rejected);
        }
        LLAMA_LOG_INFO("z6 classlru finale: emivita=%llu token=%lld chiamate=%lld freddi_per_livello=%.6f caricamenti_per_token=%.6f\n",
                (unsigned long long)class_half, (long long)token, (long long)calls,
                calls ? double(cold)/calls : 0.0, double(dyn_up)/token);
    }
    ~impl() {
        report_usage();
        if (uploader.joinable()) {
            { std::lock_guard<std::mutex> lock(upload_m); upload_stop = true; }
            upload_cv.notify_all();
            uploader.join();
        }
    }
    uint64_t dyn_period = 0;
    int dyn_max = 20;
    int64_t token = 0;
    size_t turno = 0;
    std::vector<std::pair<int,int>> wanted;
    std::vector<pending_upload> pending;
    int64_t dyn_tot = 0, dyn_up = 0;
    uint64_t class_half = 0;
    bool class_debug = false;
    int64_t class_mancanti = 0; // passi in cui un livello non aveva osservazione dei caldi (grafo non parallelo)
    double class_decay = 0;
    // The clock follows the existing policy: successful decode calls of exactly one token.
    double score_at(const layer & l, int e) const {
        return e < 0 ? 0.0 : l.score[e]*std::exp(-class_decay*double(token - l.score_token[e]));
    }
    void touch(layer & l, int e) {
        GGML_ASSERT(e >= 0 && (size_t)e < l.score.size());
        l.score[e] = score_at(l, e) + 1.0;
        l.score_token[e] = token;
    }

    // GGML_Z6_CALDI_RIPRISTINO: in generazione gli slot dinamici seguono gli esperti di un token alla volta; un prefill
    // grande che arriva dopo trova un insieme caldi peggiore per i blocchi (14,6K: 490 contro 515 t/s, 07/11). Prima
    // del batch gli slot dinamici tornano agli esperti statici della selezione iniziale, se non sono già caldi altrove.
    // Gli stessi caricamenti dei dinamici, ma aspettati qui: il prefill parte con mappe e banchi coerenti.
    int64_t rip_volte = 0, rip_slot = 0, rip_us = 0;
    void prima_del_prefill(int64_t n_tokens) {
#ifdef GGML_USE_CUDA
        const int64_t t0 = ggml_time_us();
        bool diverso = !pending.empty() || upload_left.load() > 0;
        for (size_t il = 0; !diverso && il < layers.size(); ++il) {
            const auto & l = layers[il];
            for (size_t s = l.slots_fissi; s < l.owner.size(); ++s) if (l.owner[s] != l.selected[s]) { diverso = true; break; }
        }
        if (!diverso) return;
        // i caricamenti in corso finiscono prima di riscrivere slot e mappe
        for (auto & b : gpus) ggml_backend_synchronize(b.gpu);
        while (upload_left.load() > 0) std::this_thread::yield();
        for (auto & p : pending) {
            auto & l = layers[p.il];
            l.hot[p.expert] = p.slot; l.cold[p.expert] = -1; l.dirty = true;
        }
        pending.clear();
        wanted.clear();
        std::vector<pending_upload> rimessi;
        for (size_t il = 0; il < layers.size(); ++il) {
            auto & l = layers[il];
            for (size_t s = l.slots_fissi; s < l.owner.size(); ++s) {
                const int e = l.selected[s], vecchio = l.owner[s];
                if (vecchio == e || l.hot[e] >= 0) continue;
                if (vecchio >= 0) { l.hot[vecchio] = -1; l.cold[vecchio] = vecchio; }
                l.owner[s] = e; l.hot[e] = (int32_t)s; l.cold[e] = -1; l.dirty = true;
                rimessi.push_back({(int)il, e, (int)s});
            }
        }
        for (auto & q : rimessi) {
            auto & l = layers[q.il];
            auto gpu = gpus[layer_bank[q.il]].gpu;
            for (auto src : l.sources) {
                auto dst = banks.at(src);
                upload_left.fetch_add(1);
                std::lock_guard<std::mutex> lock(upload_m);
                upload_q.push_back(make_job(gpu, dst, src, q.expert, q.slot));
            }
        }
        upload_cv.notify_all();
        while (upload_left.load() > 0) std::this_thread::yield();
        for (auto & l : layers) {
            if (!l.dirty) continue;
            l.dirty = false;
            ggml_backend_tensor_set(l.hot_map, l.hot.data(), 0, l.hot.size()*sizeof(int32_t));
            ggml_backend_tensor_set(l.cold_map, l.cold.data(), 0, l.cold.size()*sizeof(int32_t));
        }
        for (auto & b : gpus) ggml_backend_synchronize(b.gpu);
        rip_volte++; rip_slot += (int64_t)rimessi.size(); rip_us += ggml_time_us() - t0;
        LLAMA_LOG_INFO("z6 ripristino: batch da %lld token, %zu slot tornati statici in %.1f ms (in tutto %lld volte, %.1f slot e %.1f ms in media)\n",
                (long long)n_tokens, rimessi.size(), (ggml_time_us() - t0)/1e3, (long long)rip_volte,
                1.0*rip_slot/rip_volte, rip_us/1e3/rip_volte);
#else
        (void)n_tokens;
#endif
    }
    void dopo_token() {
        ++token;
        int scelti[16];
        for (size_t il = 0; il < layers.size(); ++il) {
            auto & l = layers[il];
            if (l.selected.empty()) continue;
            const int n = ggml_z6_freddi_scelti((int)il, scelti, 16);
            dyn_tot += n;
            if (class_debug) { ++l.observed; l.cold_uses += n; }
            if (class_half) {
                // ready was already waited by the cold CPU split. Map ownership cannot change
                // until below: pending uploads have not yet been published or replaced.
                auto slots = (int32_t *)((char *)usage_pinned.get() + il*usage_stride);
                bool mancante = true;
                for (int64_t k = 0; k < n_used; ++k) mancante = mancante && slots[k] == -2;
                if (mancante) { ++class_mancanti; goto z6_classlru_fatto; }
                {
                int hot_uses = 0;
                for (int64_t k = 0; k < n_used; ++k) {
                    const int s = slots[k];
                    GGML_ASSERT(s == -1 || (s >= 0 && (size_t)s < l.owner.size()));
                    if (s >= 0) { touch(l, l.owner[s]); ++hot_uses; }
                    slots[k] = -2; // detect a missed observation on the next decode
                }
                GGML_ASSERT(hot_uses + n == n_used);
                for (int k = 0; k < n; ++k) touch(l, scelti[k]);
                }
            }
            z6_classlru_fatto:
            for (int i = 0; i < n; ++i) {
                const int e = scelti[i];
                if (token - l.seen[e] <= 2) wanted.emplace_back((int)il, e);   // seconda vista
                l.seen[e] = (int32_t)token;
            }
        }
        if (token % (int64_t)dyn_period) return;
#ifdef GGML_USE_CUDA
        // i caricamenti del periodo prima sono finiti: gli esperti diventano caldi
        for (auto & b : gpus) ggml_backend_synchronize(b.gpu);
        while (upload_left.load() > 0) std::this_thread::yield();
        for (auto & p : pending) {
            auto & l = layers[p.il];
            l.hot[p.expert] = p.slot; l.cold[p.expert] = -1; l.dirty = true;
        }
        pending.clear();
        // FIFO: lo slot più vecchio torna freddo subito, prima che il caricamento lo scriva
        std::vector<pending_upload> next;
        const size_t limite = (size_t)dyn_max*dyn_period;
        std::vector<size_t> nel_giro(layers.size(), 0);
        // candidati unici, i più recenti prima e a turno fra i livelli: a tetto pieno la lista premiava i primi livelli
        std::vector<std::vector<int>> per_livello(layers.size());
        for (auto w = wanted.rbegin(); w != wanted.rend(); ++w) {
            auto & v = per_livello[w->first];
            if (std::find(v.begin(), v.end(), w->second) == v.end()) v.push_back(w->second);
        }
        turno = (turno + 1) % layers.size();
        bool altri = true;
        for (size_t p = 0; altri && next.size() < limite; ++p) {
            altri = false;
            for (size_t i = 0; i < layers.size() && next.size() < limite; ++i) {
                const int il = (int)((turno + i) % layers.size());
                if (p >= per_livello[il].size()) continue;
                altri = true;
                auto & l = layers[il];
                const int e = per_livello[il][p];
                if (l.hot[e] >= 0) continue;
                // al massimo un nuovo esperto per slot a ogni giro, e i più frequenti restano fissi
                const size_t dinamici = l.owner.size() - l.slots_fissi;
                if (nel_giro[il] >= dinamici) continue;
                int slot;
                if (!class_half) {
                    // Keep the original FIFO and its cursor exactly when opt-in is absent/zero.
                    slot = (int)(l.slots_fissi + l.next_slot++ % dinamici);
                } else {
                    slot = -1;
                    double lowest = std::numeric_limits<double>::infinity();
                    for (size_t s = l.slots_fissi; s < l.owner.size(); ++s) {
                        bool reserved = false;
                        for (const auto & q : next) if (q.il == il && q.slot == (int)s) { reserved = true; break; }
                        if (reserved) continue;
                        const double value = score_at(l, l.owner[s]);
                        if (value < lowest) { lowest = value; slot = (int)s; }
                    }
                    // Strictly better only; rejection must not consume the per-layer load budget.
                    if (slot < 0 || lowest >= score_at(l, e)) {
                        if (class_debug) ++l.rejected;
                        continue;
                    }
                }
                nel_giro[il]++;
                if (class_debug) ++l.loads;
                const int vecchio = l.owner[slot];
                if (vecchio >= 0) { l.hot[vecchio] = -1; l.cold[vecchio] = vecchio; l.dirty = true; }
                l.owner[slot] = e;
                next.push_back({il, e, slot});
            }
        }
        wanted.clear();
        for (auto & l : layers) {
            if (!l.dirty) continue;
            l.dirty = false;
            ggml_backend_tensor_set(l.hot_map, l.hot.data(), 0, l.hot.size()*sizeof(int32_t));
            ggml_backend_tensor_set(l.cold_map, l.cold.data(), 0, l.cold.size()*sizeof(int32_t));
        }
        for (auto & q : next) {
            auto & l = layers[q.il];
            auto gpu = gpus[layer_bank[q.il]].gpu;
            for (auto src : l.sources) {
                auto dst = banks.at(src);
                upload_left.fetch_add(1);
                std::lock_guard<std::mutex> lock(upload_m);
                upload_q.push_back(make_job(gpu, dst, src, q.expert, q.slot));
            }
        }
        upload_cv.notify_all();
        dyn_up += (int64_t)next.size();
        pending = std::move(next);
        if (token % 200 == 0) LLAMA_LOG_INFO("z6 dinamici: token %lld, freddi %.1f%%, caricati %.1f per token\n",
                (long long)token, 100.0*dyn_tot/(token*480.0), 1.0*dyn_up/token);
#endif
    }
    struct bank {
        ggml_backend_t gpu = nullptr;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
        size_t allocated = 0, n_tensors = 0;
    };
    std::vector<bank> gpus;
    std::vector<int> layer_bank; // GGML_Z6_CALDI_PER_LIVELLO: ogni livello usa la scheda del suo denso
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> banks;
    std::unique_ptr<void, pinned_deleter> pinned;
    std::unique_ptr<void, pinned_deleter> usage_pinned;
    size_t usage_stride = 0;
    ggml_backend_buffer_ptr host_buffer;
    std::unique_ptr<void, pinned_deleter> pair_pinned;
    ggml_backend_buffer_ptr pair_host_buffer;
    // GGML_Z6_PP_QUARTETTO: lo staging del terzo e del quarto slot, allocati al primo uso
    std::unique_ptr<void, pinned_deleter> quarto_pinned[2];
    ggml_backend_buffer_ptr quarto_host_buffer[2];
    int pair_slot = 0;
    ggml_backend_buffer_t slot_buffer() const {
        return pair_slot >= 2 ? quarto_host_buffer[pair_slot - 2].get() : pair_slot ? pair_host_buffer.get() : host_buffer.get();
    }
    char * slot_pinned() const {
        return (char *) (pair_slot >= 2 ? quarto_pinned[pair_slot - 2].get() : pair_slot ? pair_pinned.get() : pinned.get());
    }
    size_t ids_offset = 0, weights_offset = 0, output_offset = 0, packet_size = 0;
    int64_t n_embd = 0, n_used = 0, staging_tokens = 32;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends) : layers(model.layers.size()) {
        const char * file = std::getenv("GGML_Z6_CALDI_FILE");
        const uint64_t mib = env_integer("GGML_Z6_CALDI_MIB", 0);
        if (!file || !*file || !mib || mib > SIZE_MAX/(1024*1024))
            throw std::runtime_error("servono GGML_Z6_CALDI_FILE e GGML_Z6_CALDI_MIB positivi");
        const size_t budget = mib*1024*1024;
        auto add_bank = [&](int device) {
            const std::string name = "CUDA" + std::to_string(device);
            ggml_backend_t found = nullptr;
            for (auto b : backends) if (name == ggml_backend_name(b)) found = b;
            if (!found) throw std::runtime_error("backend " + name + " non disponibile");
            for (size_t i = 0; i < gpus.size(); ++i) if (gpus[i].gpu == found) return (int)i;
            gpus.emplace_back();
            gpus.back().gpu = found;
            return (int)gpus.size() - 1;
        };
        layer_bank.assign(layers.size(), 0);
        if (env_integer("GGML_Z6_CALDI_PER_LIVELLO", 0) != 0) {
            if (model.split_mode != LLAMA_SPLIT_MODE_LAYER) throw std::runtime_error("GGML_Z6_CALDI_PER_LIVELLO richiede -sm layer");
            for (size_t il = 0; il < layers.size(); ++il) layer_bank[il] = add_bank(model.devices[model.default_layer_device[il]]);
        } else {
            add_bank(llama_z6_caldi::device());
        }
        const auto buft = ggml_backend_get_default_buffer_type(gpus[0].gpu);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        const int ne = model.hparams.n_expert;
        n_embd = model.hparams.n_embd; n_used = model.hparams.n_expert_used;
        if (ne <= 0 || !model.rpc_servers.empty()) throw std::runtime_error("modello MoE locale richiesto");

        // This first implementation deliberately accepts only the bias-free fused FFN.
        // Unsupported layouts fail explicitly instead of changing their semantics.
        for (size_t il = 0; il < layers.size(); ++il) {
            const auto & l = model.layers[il];
            if (!l.ffn_down_exps) continue;
            // MTP (nextn) layers are loaded only with speculative MTP and are not in the counts file:
            // they stay on the normal path instead of aborting context creation.
            if (model.hparams.nextn_predict_layers > 0 && il >= model.hparams.n_layer - model.hparams.nextn_predict_layers) {
                LLAMA_LOG_INFO("z6 caldi: livello %zu MTP escluso\n", il);
                continue;
            }
            if (l.ffn_up_exps_b || l.ffn_gate_exps_b || l.ffn_up_gate_exps_b || l.ffn_down_exps_b || l.ffn_down_exps_s)
                throw std::runtime_error("esperti con bias/scale non supportati nella prima versione");
            auto ws = weights(l);
            for (auto t : ws) {
                if (!t || !t->buffer || !ggml_backend_buffer_is_host(t->buffer) ||
                        !ggml_is_quantized(t->type) || t->ne[2] != ne || t->ne[3] != 1 || !ggml_is_contiguous(t))
                    throw std::runtime_error("esperti quantizzati contigui tutti in RAM richiesti");
                if (model.hadamard_rotation(t)) throw std::runtime_error("esperti con rotazione Hadamard non supportati");
            }
            if (!l.ffn_up_gate_exps && l.ffn_up_exps->type != l.ffn_gate_exps->type)
                throw std::runtime_error("up e gate devono avere lo stesso tipo per MMVQ fuso");
            size_t bytes_per_expert = 0;
            for (auto t : ws) bytes_per_expert += bank_expert_bytes(t);
            LLAMA_LOG_INFO("z6 caldi: costo livello %zu byte_esperto %zu\n", il, bytes_per_expert);
            layers[il].sources = std::move(ws);
        }

        struct count { int il, expert; uint64_t n; };
        std::vector<count> counts;
        std::vector<std::vector<bool>> seen(layers.size(), std::vector<bool>(ne));
        std::ifstream input(file);
        if (!input) throw std::runtime_error("file conteggi illeggibile");
        std::string line;
        while (std::getline(input, line)) {
            std::istringstream row(line);
            int il, expert; int64_t n; std::string extra;
            if (!(row >> il >> expert >> n) || row >> extra || il < 0 || il >= (int)layers.size() ||
                    expert < 0 || expert >= ne || n < 0 || layers[il].sources.empty() || seen[il][expert])
                throw std::runtime_error("riga conteggi non valida o duplicata: " + line);
            seen[il][expert] = true;
            counts.push_back({il, expert, (uint64_t)n});
        }
        std::sort(counts.begin(), counts.end(), [](const count & a, const count & b) {
            if (a.n != b.n) return a.n > b.n;
            return std::tie(a.il,a.expert) < std::tie(b.il,b.expert);
        });
        auto pad = [alignment](size_t n) { return (n + alignment - 1)/alignment*alignment; };
        auto layer_size = [&](int il, size_t slots) {
            if (!slots) return size_t(0);
            size_t bytes = 2*pad(ne*sizeof(int32_t));
            for (auto src : layers[il].sources) {
                auto t = *src;
                t.type = bank_type(src);
                t.nb[0] = ggml_type_size(t.type); t.nb[1] = ggml_row_size(t.type, t.ne[0]); t.nb[2] = t.nb[1]*t.ne[1];
                t.ne[2] = slots; t.nb[3] = t.nb[2]*slots;
                bytes += pad(ggml_backend_buft_get_alloc_size(buft, &t));
            }
            return bytes;
        };
        size_t selected = 0;
        uint64_t total = 0, covered = 0;
        for (const auto & c : counts) total += c.n;
        // GGML_Z6_CALDI_MIN_LIVELLO=<n>: the n most used experts of every layer are taken first, then the budget is
        // filled by count as before. A layer without hot experts has no parallel group: its cold experts run as an
        // ordinary CPU piece with the GPU waiting, and the pair of GGML_Z6_MTP_PIPELINE loses the overlap there. With
        // a small budget (when the GPU, not the CPU, sets the pace) this keeps every layer in the parallel path.
        const uint64_t min_livello = env_integer("GGML_Z6_CALDI_MIN_LIVELLO", 0);
        std::vector<std::vector<bool>> preso(layers.size(), std::vector<bool>(ne));
        auto prendi = [&](const count & c) {
            auto & l = layers[c.il];
            const size_t delta = layer_size(c.il, l.selected.size() + 1) - layer_size(c.il, l.selected.size());
            auto & b = gpus[layer_bank[c.il]];
            if (delta > budget - b.allocated) return;
            b.allocated += delta; l.selected.push_back(c.expert); covered += c.n; ++selected;
            preso[c.il][c.expert] = true;
        };
        if (min_livello) for (const auto & c : counts) {
            if (!c.n) break;
            if (layers[c.il].selected.size() < min_livello) prendi(c);
        }
        for (const auto & c : counts) {
            if (!c.n) break;
            if (!preso[c.il][c.expert]) prendi(c);
        }
        if (min_livello) {
            size_t senza = 0;
            for (size_t il = 0; il < layers.size(); ++il) senza += !layers[il].sources.empty() && layers[il].selected.empty();
            LLAMA_LOG_INFO("z6 caldi: almeno %llu esperti per livello richiesti, livelli rimasti senza caldi %zu\n",
                    (unsigned long long)min_livello, senza);
        }
        if (!selected) throw std::runtime_error("budget insufficiente o conteggi vuoti");
        for (size_t il = 0; il < layers.size(); ++il)
            if (!layers[il].selected.empty()) gpus[layer_bank[il]].n_tensors += layers[il].sources.size() + 2;
        for (auto & b : gpus) {
            b.ctx.reset(ggml_init({b.n_tensors*ggml_tensor_overhead(), nullptr, true}));
            if (!b.ctx) throw std::runtime_error("metadati esperti caldi: memoria esaurita");
        }
        for (size_t il = 0; il < layers.size(); ++il) {
            auto & l = layers[il];
            if (l.selected.empty()) continue;
            auto ctx = gpus[layer_bank[il]].ctx.get();
            l.hot_map = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ne);
            l.cold_map = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ne);
            ggml_format_name(l.hot_map, "z6_caldi.map-%zu", il);
            ggml_format_name(l.cold_map, "z6_freddi.map-%zu", il);
            for (auto src : l.sources) {
                auto dst = ggml_new_tensor_3d(ctx, bank_type(src), src->ne[0], src->ne[1], l.selected.size());
                ggml_format_name(dst, "z6_caldi.%s", src->name);
                banks[src] = dst;
            }
        }
        size_t actual = 0;
        std::string names;
        for (auto & b : gpus) {
            if (!b.n_tensors) continue;
            b.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(b.ctx.get(), ggml_backend_get_default_buffer_type(b.gpu)));
            if (!b.buffer) throw std::runtime_error("allocazione VRAM esperti caldi fallita");
            ggml_backend_buffer_set_usage(b.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            const size_t size = ggml_backend_buffer_get_size(b.buffer.get());
            if (size > budget) throw std::runtime_error("allocazione effettiva oltre il budget");
            ggml_backend_buffer_clear(b.buffer.get(), 0); // includes CUDA padding
            actual += size;
            names += std::string(names.empty() ? "" : "+") + ggml_backend_name(b.gpu);
        }
        std::vector<uint8_t> restored;   // GGML_Z6_IQ3S_LUT: one expert back in the IQ3_S layout
        size_t restored_experts = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            auto & l = layers[il];
            if (l.selected.empty()) continue;
            std::vector<int32_t> hot(ne, -1), cold(ne);
            for (int e = 0; e < ne; ++e) cold[e] = e;
            for (size_t slot = 0; slot < l.selected.size(); ++slot) {
                const int e = l.selected[slot]; hot[e] = slot; cold[e] = -1;
                for (auto src : l.sources) {
                    auto dst = banks.at(src);
                    ggml_backend_tensor_set(dst, bank_expert(src, e, restored), slot*dst->nb[2], dst->nb[2]);
                }
                for (auto src : l.sources) restored_experts += src->type == GGML_TYPE_IQ3_S_Z6;
                LLAMA_LOG_INFO("z6 caldi: scelto %zu %d slot %zu\n", il, e, slot);
            }
            ggml_backend_tensor_set(l.hot_map, hot.data(), 0, hot.size()*sizeof(int32_t));
            ggml_backend_tensor_set(l.cold_map, cold.data(), 0, cold.size()*sizeof(int32_t));
            l.hot = std::move(hot); l.cold = std::move(cold);
            l.owner = l.selected;
            l.seen.assign(ne, -1000000);
        }
        if (restored_experts) LLAMA_LOG_INFO("z6 caldi: %zu matrici di esperti riportate da IQ3_S riordinato a IQ3_S per la VRAM\n", restored_experts);
        dyn_period = env_integer("GGML_Z6_CALDI_DINAMICI", 0);
        class_half = env_integer("GGML_Z6_CALDI_CLASSLRU", 0);
        class_debug = env_integer("GGML_Z6_CALDI_CLASSLRU_DEBUG", 0) != 0;
        if (class_half && (!dyn_period || n_used > 16 || layers.size() > 128))
            throw std::runtime_error("ClassLRU richiede dinamici, <=16 esperti usati e <=128 livelli");
        if (class_half) {
            class_decay = std::log(2.0)/double(class_half);
            for (auto & l : layers) {
                if (l.selected.empty()) continue;
                l.score.assign(ne, 0.0);
                l.score_token.assign(ne, 0);
            }
        }
        if (dyn_period) {
            dyn_max = (int)env_integer("GGML_Z6_CALDI_DIN_MAX", 20);
            upload_chunk = env_integer("GGML_Z6_CALDI_DIN_PEZZO_KIB", 256)*1024;
            if (!upload_chunk) throw std::runtime_error("GGML_Z6_CALDI_DIN_PEZZO_KIB deve essere positivo");
            const uint64_t fissi = env_integer("GGML_Z6_CALDI_DIN_FISSI", 50);   // percento degli slot mai sostituiti
            for (auto & l : layers) {
                l.slots_fissi = l.owner.size()*fissi/100;
                if (l.slots_fissi >= l.owner.size() && !l.owner.empty()) l.slots_fissi = l.owner.size() - 1;
            }
        }
        for (auto & b : gpus) ggml_backend_synchronize(b.gpu);
        if (env_integer("GGML_Z6_CALDI_PARALLELO", 0) != 0) {
#ifdef GGML_USE_CUDA
            const auto align = [](size_t n) { return (n + 127)/128*128; };
            // GGML_Z6_CALDI_BATCH_PAR=<token massimi>: pacchetto anche per i batch, che consegnano la somma già fatta
            staging_tokens = std::max<int64_t>(32, env_integer("GGML_Z6_CALDI_BATCH_PAR", 0));
            ids_offset = align(staging_tokens*n_embd*sizeof(float));
            weights_offset = ids_offset + align(staging_tokens*n_used*sizeof(int32_t));
            output_offset = weights_offset + align(staging_tokens*n_used*sizeof(float));
            packet_size = output_offset + align(std::max<int64_t>(32*n_used, staging_tokens)*n_embd*sizeof(float));
            pinned.reset(ggml_backend_cuda_z6_host_alloc(packet_size));
            if (pinned) host_buffer.reset(ggml_backend_cpu_buffer_from_ptr(pinned.get(), packet_size));
            if (!host_buffer) pinned.reset();
            LLAMA_LOG_INFO("z6 parallelo: staging %zu byte, %s; pesi CPU non registrati\n", packet_size,
                    pinned ? "attivo" : "allocazione fallita, ritorno sequenziale");
#else
            throw std::runtime_error("GGML_Z6_CALDI_PARALLELO richiede CUDA");
#endif
        }
        if (class_half) {
#ifdef GGML_USE_CUDA
            if (!pinned) throw std::runtime_error("ClassLRU richiede lo staging parallelo pinned");
            usage_stride = (n_used*sizeof(int32_t) + 127)/128*128;
            usage_pinned.reset(ggml_backend_cuda_z6_host_alloc(layers.size()*usage_stride));
            if (!usage_pinned) throw std::runtime_error("ClassLRU: allocazione osservazioni fallita");
            for (size_t il = 0; il < layers.size(); ++il) {
                auto ptr = (int32_t *)((char *)usage_pinned.get() + il*usage_stride);
                std::fill(ptr, ptr + n_used, -2);
            }
#else
            throw std::runtime_error("ClassLRU richiede CUDA");
#endif
        }
        // ultimo passo: un'eccezione del costruttore non troverebbe il distruttore a fermare il thread
        if (dyn_period) uploader = std::thread([this] { upload_loop(); });
        LLAMA_LOG_INFO("z6 caldi: %zu esperti, budget %llu MiB, allocati %zu byte su %s, copertura sul file %.4f%%; "
                "attivi solo per n_tokens<=32, MMVQ per singolo token, nessun upload in inferenza\n",
                selected, (unsigned long long)mib, actual, names.c_str(), total ? 100.0*covered/total : 0.0);
    }
};

llama_z6_caldi::llama_z6_caldi(const llama_model & model, const std::vector<ggml_backend_t> & backends)
    : pimpl(new impl(model, backends)) {}
llama_z6_caldi::~llama_z6_caldi() = default;
bool llama_z6_caldi::has_layer(int il) const {
    return il >= 0 && il < (int)pimpl->layers.size() && !pimpl->layers[il].selected.empty();
}
bool llama_z6_caldi::solo_freddi(int il) const {
    return il >= 0 && il < (int)pimpl->layers.size() && !pimpl->layers[il].sources.empty() && pimpl->layers[il].selected.empty();
}
ggml_backend_t llama_z6_caldi::backend(int il) const { return pimpl->gpus[pimpl->layer_bank.at(il)].gpu; }
ggml_tensor * llama_z6_caldi::map(int il, bool hot) const {
    return hot ? pimpl->layers.at(il).hot_map : pimpl->layers.at(il).cold_map;
}
ggml_tensor * llama_z6_caldi::experts(const ggml_tensor * source) const {
    return source ? pimpl->banks.at(source) : nullptr;
}
bool llama_z6_caldi::parallel() const { return bool(pimpl->pinned); }
int64_t llama_z6_caldi::staging_tokens() const { return pimpl->staging_tokens; }
void llama_z6_caldi::pair_slot(int slot) {
    GGML_ASSERT(slot >= 0 && slot <= 3);
#ifdef GGML_USE_CUDA
    if (slot >= 2 && !pimpl->quarto_host_buffer[slot - 2]) {
        auto & pin = pimpl->quarto_pinned[slot - 2];
        pin.reset(ggml_backend_cuda_z6_host_alloc(pimpl->packet_size));
        if (pin) pimpl->quarto_host_buffer[slot - 2].reset(ggml_backend_cpu_buffer_from_ptr(pin.get(), pimpl->packet_size));
        if (!pimpl->quarto_host_buffer[slot - 2]) throw std::runtime_error("z6 quartetto: staging del terzo e quarto slot non disponibile");
    }
    if (slot == 1 && !pimpl->pair_host_buffer) {
        pimpl->pair_pinned.reset(ggml_backend_cuda_z6_host_alloc(pimpl->packet_size));
        if (pimpl->pair_pinned) pimpl->pair_host_buffer.reset(
                ggml_backend_cpu_buffer_from_ptr(pimpl->pair_pinned.get(), pimpl->packet_size));
        if (!pimpl->pair_host_buffer) throw std::runtime_error("z6 sovrapposto: secondo staging non disponibile");
    }
#endif
    pimpl->pair_slot = slot;
}
int32_t * llama_z6_caldi::usage_slots(int il) const {
    return pimpl->class_half ? (int32_t *)((char *)pimpl->usage_pinned.get() + il*pimpl->usage_stride) : nullptr;
}
void llama_z6_caldi::dopo_token() { if (pimpl->dyn_period) pimpl->dopo_token(); }
void llama_z6_caldi::prima_del_prefill(int64_t n_tokens) { if (pimpl->dyn_period) pimpl->prima_del_prefill(n_tokens); }
void llama_z6_caldi::staging(ggml_context * ctx, int layer, int64_t nt, ggml_tensor ** x, ggml_tensor ** ids, ggml_tensor ** w) const {
    GGML_ASSERT(parallel() && nt > 0 && nt <= pimpl->staging_tokens);
    auto make = [&](ggml_type type, int64_t a, int64_t b, int64_t c, size_t offset, const char * name) {
        auto t = ggml_new_tensor_3d(ctx, type, a, b, c);
        GGML_ASSERT(offset + ggml_nbytes(t) <= pimpl->packet_size);
        t->buffer = pimpl->slot_buffer();
        t->data = pimpl->slot_pinned() + offset;
        ggml_format_name(t, "%s-%d", name, layer);
        return t;
    };
    *x = make(GGML_TYPE_F32, pimpl->n_embd, 1, nt, 0, "z6_cpu_x");
    *ids = make(GGML_TYPE_I32, pimpl->n_used, nt, 1, pimpl->ids_offset, "z6_cpu_ids");
    *w = make(GGML_TYPE_F32, 1, pimpl->n_used, nt, pimpl->weights_offset, "z6_cpu_weights");
}
void llama_z6_caldi::stage_output(ggml_tensor * out) const {
    GGML_ASSERT(parallel() && pimpl->output_offset + ggml_nbytes(out) <= pimpl->packet_size);
    out->buffer = pimpl->slot_buffer();
    out->data = pimpl->slot_pinned() + pimpl->output_offset;
}
