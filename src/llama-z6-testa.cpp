#include "llama-z6-testa.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "ggml-cpp.h"
#include <cstdlib>
#include <stdexcept>
#include <string>

bool llama_z6_testa::requested() {
    const char * e = std::getenv("GGML_Z6_TESTA_DIVISA");
    return e && std::atoi(e) > 0;
}

struct llama_z6_testa::impl {
    struct shard {
        ggml_backend_t backend = nullptr;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
        ggml_tensor * weight = nullptr;
    } shards[2];
    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends) {
        const auto w = model.output;
        if (model.arch != LLM_ARCH_QWEN4EXP || model.mtp ||
                model.split_mode != LLAMA_SPLIT_MODE_LAYER || model.devices.size() != 2 ||
                !w || !w->buffer || !ggml_backend_buffer_is_host(w->buffer) ||
                !ggml_is_contiguous(w) || !ggml_is_quantized(w->type) ||
                w->ne[1] % 2 || w->ne[2] != 1 || w->ne[3] != 1) {
            throw std::runtime_error("GGML_Z6_TESTA_DIVISA richiede Qwen4Exp quantizzato, due GPU, -sm layer, senza MTP");
        }
        for (int i = 0; i < 2; ++i) {
            auto & s = shards[i];
            const std::string name = "CUDA" + std::to_string(model.devices[i]);
            for (auto b : backends) if (name == ggml_backend_name(b)) s.backend = b;
            if (!s.backend) throw std::runtime_error("testa divisa: backend " + name + " assente");
            s.ctx.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
            if (!s.ctx) throw std::runtime_error("testa divisa: metadati esauriti");
            s.weight = ggml_new_tensor_2d(s.ctx.get(), w->type, w->ne[0], w->ne[1]/2);
            ggml_format_name(s.weight, "z6_testa.weight-%d", i);
            s.buffer.reset(ggml_backend_alloc_ctx_tensors(s.ctx.get(), s.backend));
            if (!s.buffer) throw std::runtime_error("testa divisa: VRAM esaurita");
            ggml_backend_buffer_set_usage(s.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_clear(s.buffer.get(), 0);
            const size_t bytes = ggml_nbytes(s.weight);
            ggml_backend_tensor_set(s.weight, (const char *)w->data + i*bytes, 0, bytes);
            LLAMA_LOG_INFO("z6 testa: righe %lld..%lld su %s, %zu byte\n",
                    (long long)(i*w->ne[1]/2), (long long)((i+1)*w->ne[1]/2), name.c_str(), bytes);
        }
    }
};
llama_z6_testa::llama_z6_testa(const llama_model & model, const std::vector<ggml_backend_t> & backends)
    : pimpl(new impl(model, backends)) {}
llama_z6_testa::~llama_z6_testa() = default;
ggml_tensor * llama_z6_testa::build(ggml_context * ctx, ggml_cgraph * graph, ggml_backend_sched_t sched,
        ggml_backend_t cpu, ggml_tensor * hidden) const {
    // The exit mixer remains on the last device. Do not let the first shard
    // pull its producer to GPU0 during the scheduler's backward propagation.
    ggml_backend_sched_set_tensor_backend(sched, hidden, pimpl->shards[1].backend);
    ggml_tensor * partial[2];
    for (int i = 0; i < 2; ++i) {
        partial[i] = ggml_mul_mat(ctx, pimpl->shards[i].weight, hidden);
        ggml_format_name(partial[i], "z6_testa.logits-%d", i);
        ggml_backend_sched_set_tensor_backend(sched, partial[i], pimpl->shards[i].backend);
        // Copy hidden to GPU0 before enqueueing either head. GPU1's head has
        // no dependency on GPU0. Join on CPU only after both have launched.
        ggml_build_forward_expand(graph, partial[i]);
    }
    auto result = ggml_concat(ctx, partial[0], partial[1], 0);
    ggml_backend_sched_set_tensor_backend(sched, result, cpu);
    return result;
}
