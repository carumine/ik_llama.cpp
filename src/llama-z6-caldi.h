#pragma once

#include "ggml-backend.h"
#include <memory>
#include <vector>

struct llama_model;

// Expert banks. Immutable unless GGML_Z6_CALDI_DINAMICI replaces slots between tokens.
class llama_z6_caldi {
public:
    static bool requested();
    static int device(); // raw CUDA device ordinal, independent of model layer split
    llama_z6_caldi(const llama_model & model, const std::vector<ggml_backend_t> & backends);
    ~llama_z6_caldi();
    bool has_layer(int il) const;
    ggml_backend_t backend(int il) const;
    ggml_tensor * map(int il, bool hot) const;
    ggml_tensor * experts(const ggml_tensor * source) const;
    bool parallel() const;
    int64_t staging_tokens() const;
    void pair_slot(int slot);
    void staging(ggml_context * ctx, int layer, int64_t n_tokens, ggml_tensor ** x, ggml_tensor ** ids, ggml_tensor ** weights) const;
    void stage_output(ggml_tensor * output) const;
    // GGML_Z6_CALDI_DINAMICI=<periodo in token>: da chiamare dopo ogni decode a un token
    void dopo_token();
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
