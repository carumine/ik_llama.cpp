#pragma once

#include "common.cuh"

// Z6: hyper-connection fuse di qwen4exp (GGML_OP_Z6_HC, vedi ggml_z6_hc_* in ggml.h)
void ggml_cuda_op_z6_hc(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
