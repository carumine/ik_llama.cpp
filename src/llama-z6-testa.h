#pragma once
#include "ggml-backend.h"
#include <memory>
#include <vector>
struct llama_model;

// Immutable vocabulary shards, enabled only by GGML_Z6_TESTA_DIVISA=1.
class llama_z6_testa {
public:
    static bool requested();
    llama_z6_testa(const llama_model & model, const std::vector<ggml_backend_t> & backends);
    ~llama_z6_testa();
    ggml_tensor * build(ggml_context * ctx, ggml_cgraph * graph, ggml_backend_sched_t sched,
            ggml_backend_t cpu, ggml_tensor * hidden) const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
