#include "llama-z6-tp.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "ggml-cpp.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

bool llama_z6_tp::requested() {
    const char * e = std::getenv("GGML_Z6_DENSA_TP");
    return e && std::atoi(e) > 0;
}

struct llama_z6_tp::impl {
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> buffers;
    std::vector<const ggml_tensor *> registrate;   // only these leave the CUDA registry with this object

    // The CUDA registry is keyed by the weight. A second context on the same model (--spec-type mtp creates one) used
    // to allocate the copies again, have every registration refused, and clear the registry on its way out: the first
    // context lost the split while its captured CUDA graphs still copied through the freed pinned buffers. Now the
    // contexts of a model share one split, released with the last of them. The exchange buffers are shared too:
    // contexts of one model must not decode at the same time (the server never does).
    static std::mutex & mutex() { static std::mutex m; return m; }
    static std::map<const llama_model *, std::weak_ptr<impl>> & modelli() {
        static std::map<const llama_model *, std::weak_ptr<impl>> m;
        return m;
    }
    static std::shared_ptr<impl> per(const llama_model & model, const std::vector<ggml_backend_t> & backends) {
        std::lock_guard<std::mutex> lock(mutex());
        auto & slot = modelli()[&model];
        auto p = slot.lock();
        if (p) {
            LLAMA_LOG_INFO("z6 densa tp: altro contesto sullo stesso modello, divisione condivisa\n");
            return p;
        }
        p = std::make_shared<impl>(model, backends);
        slot = p;
        return p;
    }
    void rilascia() {
#ifdef GGML_USE_CUDA
        for (auto w : registrate) ggml_backend_cuda_z6_tp_unregister(w);
#endif
        registrate.clear();
    }

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends) {
#ifndef GGML_USE_CUDA
        throw std::runtime_error("GGML_Z6_DENSA_TP richiede CUDA");
#else
        // a throw skips ~impl: the registry must not keep entries whose buffers die with this object
        try {
            carica(model, backends);
        } catch (...) {
            rilascia();
            throw;
        }
#endif
    }
#ifdef GGML_USE_CUDA
    void carica(const llama_model & model, const std::vector<ggml_backend_t> & backends) {
        if ((model.arch != LLM_ARCH_QWEN4EXP && model.arch != LLM_ARCH_GLM5NEXT) ||
                model.split_mode != LLAMA_SPLIT_MODE_LAYER || model.devices.size() != 2) {
            throw std::runtime_error("GGML_Z6_DENSA_TP richiede Qwen4Exp o GLM5NEXT, due GPU, -sm layer");
        }
        // GGML_Z6_DENSA_TP_QUOTA: share of the rows computed by the other GPU (default 0.5)
        const char * qe = std::getenv("GGML_Z6_DENSA_TP_QUOTA");
        const double quota = qe ? std::atof(qe) : 0.5;
        if (!(quota > 0.0 && quota < 1.0)) throw std::runtime_error("GGML_Z6_DENSA_TP_QUOTA fuori da (0, 1)");
        ggml_backend_t gpu[2] = {};
        for (int i = 0; i < 2; ++i) {
            const auto name = "CUDA" + std::to_string(model.devices[i]);
            for (auto b : backends) if (name == ggml_backend_name(b)) gpu[i] = b;
            if (!gpu[i]) throw std::runtime_error("densa tp: backend " + name + " assente");
        }
        // one metadata context per GPU for all the matrices: ggml keeps at most GGML_MAX_CONTEXTS (64) contexts alive,
        // two per matrix ran out on Qwen ("metadati esauriti")
        struct cand { const ggml_tensor * w; int owner; int64_t r0, nr, k; ggml_tensor * w_r, * x_r, * y_r, * y_l; };
        std::vector<cand> cs;
        auto candidata = [&](const ggml_tensor * w, int64_t righe_min = 4096) {
            // big projections only: below 4096 rows the exchange (about 55 us) costs more than half the work
            // types without an MMVQ kernel are left out here, before any copy is allocated for them
            if (!w || !w->buffer || !ggml_backend_cuda_z6_tp_type_ok(w->type) || !ggml_is_contiguous(w) || w->ne[0] % 512 != 0 ||
                    w->ne[1] < righe_min || w->ne[2] != 1 || w->ne[3] != 1 || w->extra) return;
            int owner = -1;
            for (int d = 0; d < 2; ++d) if (ggml_backend_supports_buft(gpu[d], ggml_backend_buffer_get_type(w->buffer))) owner = d;
            if (owner < 0 || ggml_backend_buffer_is_host(w->buffer)) return;
            const int64_t n = w->ne[1], k = w->ne[0];
            const int64_t nr = (int64_t) std::floor(n*quota/128.0)*128;   // whole 128-row tiles, same kernels per row
            const int64_t r0 = n - nr;
            if (nr < 128 || r0 < 128) return;
            cs.push_back({w, owner, r0, nr, k, nullptr, nullptr, nullptr, nullptr});
        };
        // GGML_Z6_DENSA_TP_ALTRE=1: also the output projection of the attention where it is big enough to pay for an
        // exchange of its own (48 MiB: the 68 MiB ones of the MLA layers of GLM-5.3-Flash, not the 34 MiB of the
        // others) and the vocabulary head. Same op and same kernels per row as the matrices above.
        const char * ae = std::getenv("GGML_Z6_DENSA_TP_ALTRE");
        const bool altre = ae && std::atoi(ae) > 0;
        // GGML_Z6_DENSA_TP_PICCOLE=<minimum rows, e.g. 2048>: also the output projections left out because one exchange
        // costs more than half of their work (ssm_out and attn_output, 16 MiB each on Qwen3.8-Flash-Next: 47 us whole
        // against an exchange of about 55). Worth it only with a shorter exchange (GGML_Z6_TP_INGRESSO_S0,
        // GGML_Z6_TP_MAPPATA); same op and same kernels per row, so the numbers do not change.
        const char * pe = std::getenv("GGML_Z6_DENSA_TP_PICCOLE");
        const int64_t piccole = pe ? std::atoll(pe) : 0;
        // GGML_Z6_DENSA_TP_MLA=<minimum rows, e.g. 8192>: also the second projection of the queries of the MLA layers
        // (wq_b: 16384 rows on GLM-5.3-Flash, 34 MiB, about 115 us whole; 11 of them a token). It is a plain mul_mat
        // by one column like the matrices above: same op, same kernels per row, same numbers.
        const char * me = std::getenv("GGML_Z6_DENSA_TP_MLA");
        const int64_t mla = me ? std::atoll(me) : 0;
        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & layer = model.layers[il];
            for (const ggml_tensor * w : {layer.wqkv, layer.wqkv_gate, layer.wq, layer.wk, layer.wv}) {
                candidata(w);
            }
            if (altre && layer.wo && ggml_nbytes(layer.wo) >= (size_t) 48*1024*1024) candidata(layer.wo);
            else if (piccole >= 256) candidata(layer.wo, piccole);
            if (piccole >= 256) candidata(layer.ssm_out, piccole);
            if (mla >= 256) candidata(layer.wq_b, mla);
        }
        if (altre) candidata(model.output);
        if (cs.empty()) throw std::runtime_error("densa tp: nessuna matrice adatta");
        ggml_context * cx[2];
        for (int d = 0; d < 2; ++d) {
            ggml_context_ptr c(ggml_init({4*cs.size()*ggml_tensor_overhead(), nullptr, true}));
            if (!c) throw std::runtime_error("densa tp: metadati esauriti");
            cx[d] = c.get();
            ctxs.push_back(std::move(c));
        }
        for (auto & c : cs) {
            ggml_context * cr = cx[1-c.owner];
            c.w_r = ggml_new_tensor_2d(cr, c.w->type, c.k, c.nr);
            c.x_r = ggml_new_tensor_2d(cr, GGML_TYPE_F32, c.k, 16);
            c.y_r = ggml_new_tensor_2d(cr, GGML_TYPE_F32, c.nr, 16);
            c.y_l = ggml_new_tensor_2d(cx[c.owner], GGML_TYPE_F32, c.r0, 16);
            ggml_format_name(c.w_r, "z6_tp.%s", c.w->name);
        }
        size_t bytes[2] = {};
        for (int d = 0; d < 2; ++d) {
            ggml_backend_buffer_ptr b(ggml_backend_alloc_ctx_tensors(cx[d], gpu[d]));
            if (!b) throw std::runtime_error("densa tp: VRAM esaurita");
            ggml_backend_buffer_set_usage(b.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            bytes[d] = ggml_backend_buffer_get_size(b.get());
            buffers.push_back(std::move(b));
        }
        int count = 0;
        std::vector<char> staging;
        for (auto & c : cs) {
            const size_t off = (size_t) c.r0*c.w->nb[1], len = (size_t) c.nr*c.w->nb[1];
            staging.resize(len);
            ggml_backend_tensor_get(c.w, staging.data(), off, len);
            ggml_backend_tensor_set(c.w_r, staging.data(), 0, len);
            if (!ggml_backend_cuda_z6_tp_register(c.w, c.r0, c.w_r, c.x_r, c.y_r, c.y_l)) {
                // registered by another model object on the same weights, or a shape the op refuses: it stays whole
                LLAMA_LOG_WARN("z6 densa tp: %s (%s) resta intera\n", c.w->name, ggml_type_name(c.w->type));
                continue;
            }
            registrate.push_back(c.w);
            ++count;
        }
        if (count == 0) throw std::runtime_error("densa tp: nessuna matrice adatta");
        LLAMA_LOG_INFO("z6 densa tp: %d matrici divise, quota remota %.3f, VRAM in più %.1f / %.1f MiB\n",
                count, quota, bytes[0]/1048576.0, bytes[1]/1048576.0);
    }
#endif
    ~impl() {
        rilascia();
    }
};

llama_z6_tp::llama_z6_tp(const llama_model & model, const std::vector<ggml_backend_t> & backends)
    : pimpl(impl::per(model, backends)) {}
llama_z6_tp::~llama_z6_tp() {
    std::lock_guard<std::mutex> lock(impl::mutex());
    pimpl.reset();
}
