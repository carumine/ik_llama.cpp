#include "llama-model.h"
#include <cstdio>

// Separate TU: common.h and llama-arch.h have colliding historical KV names.
bool z6_glm_print_layout(const llama_model * model) {
    if (model->arch != LLM_ARCH_GLM5NEXT) return false;
    for (size_t il = 0; il < model->layers.size(); ++il) {
        const auto & layer = model->layers[il];
        if (!layer.ffn_down_exps) continue;
        const size_t ug = layer.ffn_up_gate_exps ? layer.ffn_up_gate_exps->nb[2] :
                layer.ffn_up_exps->nb[2] + layer.ffn_gate_exps->nb[2];
        std::printf("Z6_GLM_LAYER %zu %d %zu\n", il,
                    model->devices[model->default_layer_device[il]], ug+layer.ffn_down_exps->nb[2]);
    }
    return true;
}
