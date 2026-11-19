#include "common.cuh"

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context & ctx,
                           const ggml_tensor *         logits,
                           ggml_tensor *               weights,
                           ggml_tensor *               top_k,
                           ggml_tensor *               bias = nullptr,
                           ggml_tensor *               map0 = nullptr,
                           ggml_tensor *               map1 = nullptr);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * softmax, const ggml_tensor * weights);

// Returns the last fused mapping node, or start when the strict pattern does not match.
int ggml_cuda_z6_topk_maps(const ggml_cgraph * graph, int start,
                          const ggml_tensor * ids, const ggml_tensor * weights, ggml_tensor ** maps);
