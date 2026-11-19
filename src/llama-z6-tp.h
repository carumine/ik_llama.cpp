#pragma once
#include "ggml-backend.h"
#include <memory>
#include <vector>
struct llama_model;

// Z6 (GGML_Z6_DENSA_TP=1): large dense projections split by rows over the two GPUs, computed inside one CUDA op
// (ggml_backend_cuda_z6_tp_register). The graph does not change: the owner's mul_mat recognises the matrix.
class llama_z6_tp {
public:
    static bool requested();
    llama_z6_tp(const llama_model & model, const std::vector<ggml_backend_t> & backends);
    ~llama_z6_tp();
private:
    struct impl;
    // one split per model, shared by its contexts (target and MTP draft context, more slots)
    std::shared_ptr<impl> pimpl;
};
