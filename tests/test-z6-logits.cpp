#include "common.h"
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct trace_state { const char * prefix; int step = 0; };
static int trace_salva(ggml_tensor * t, trace_state & s, const std::string & name) {
    static int ordine = 0;
    { FILE * fo = std::fopen((std::string(s.prefix) + "/ordine.txt").c_str(), "a"); if (fo) { std::fprintf(fo, "%d %d %s\n", s.step, ordine++, name.c_str()); std::fclose(fo); } }
    std::vector<float> values(ggml_nelements(t));
    ggml_backend_tensor_get(t, values.data(), 0, values.size()*sizeof(float));
    auto path = std::string(s.prefix) + "/" + std::to_string(s.step) + "-" + name;
    FILE * f = std::fopen((path + ".f32").c_str(), "wb");
    if (!f || std::fwrite(values.data(), sizeof(float), values.size(), f) != values.size()) std::exit(8);
    std::fclose(f);
    f = std::fopen((path + ".txt").c_str(), "w");
    if (!f) std::exit(8);
    std::fprintf(f, "%s %s %lld %lld %lld %lld\n", ggml_op_name(t->op), ggml_type_name(t->type),
        (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
    std::fclose(f);
    return 1;
}
static int trace(ggml_tensor * t, bool ask, void * opaque) {
    auto & s = *(trace_state *)opaque;
    const std::string name(t->name);
    // Z6_LOGITS_TRACE_TUTTI=1: tutti i tensori f32 dei primi due livelli, in ordine di calcolo
    static const bool tutti = std::getenv("Z6_LOGITS_TRACE_TUTTI") != nullptr;
    // Z6_LOGITS_TRACE_LIVELLI=<n>: tutti i tensori f32 dei primi n livelli
    static const int livelli = std::getenv("Z6_LOGITS_TRACE_LIVELLI") ? std::atoi(std::getenv("Z6_LOGITS_TRACE_LIVELLI")) : 0;
    const size_t trattino = name.rfind('-');
    const bool nei_livelli = livelli > 0 && trattino != std::string::npos && trattino + 1 < name.size() &&
        name.find_first_not_of("0123456789", trattino + 1) == std::string::npos && std::atoi(name.c_str() + trattino + 1) < livelli;
    // Z6_LOGITS_TRACE_NOMI=<prefisso>: solo i tensori con quel prefisso; Z6_LOGITS_TRACE_SOLO=<k>: tutti quelli del livello k
    static const char * nomi = std::getenv("Z6_LOGITS_TRACE_NOMI");
    static const char * solo = std::getenv("Z6_LOGITS_TRACE_SOLO");
    if (nomi || solo) {
        static const std::vector<std::string> prefissi = [] {
            std::vector<std::string> v; std::string resto = nomi ? nomi : "";
            while (!resto.empty()) { const size_t c = resto.find(','); v.push_back(resto.substr(0, c)); resto = c == std::string::npos ? "" : resto.substr(c + 1); }
            return v;
        }();
        const bool sua = std::any_of(prefissi.begin(), prefissi.end(), [&](const std::string & p) { return name.rfind(p, 0) == 0; }) ||
            (solo && trattino != std::string::npos && name.substr(trattino + 1) == solo) ||
            name == "result_embd" || name == "result_output";
        if (!(s.step < 2 && t->type == GGML_TYPE_F32 && sua)) return ask ? 0 : 1;
        if (ask) return 1;
        return trace_salva(t, s, name);
    }
    const bool wanted = s.step < 2 && t->type == GGML_TYPE_F32 &&
        (name == "result_embd" || name == "result_output" || nei_livelli ||
         (name.size() > 2 && name.substr(name.size()-2) == "-0") ||
         (tutti && name.size() > 2 && name.substr(name.size()-2) == "-1"));
    if (ask) return wanted ? 1 : 0;
    if (!wanted) return 1;
    return trace_salva(t, s, name);
}

// Fixed teacher-forced tokens: no sampling or divergent generated prefixes.
// Z6_LOGITS_OUT selects the output file; ordinary common CLI loads the model.
int main(int argc, char ** argv) {
    gpt_params params;
    params.prompt = "Il faro illumina il mare.\nUna prova numerica su due schede.";
    if (!gpt_params_parse(argc, argv, params)) return 1;
    const char * path = std::getenv("Z6_LOGITS_OUT");
    if (!path) return 2;
    llama_backend_init();
    llama_numa_init(params.numa);
    auto mp = common_model_params_to_llama(params);
    auto model = llama_model_load_from_file(params.model.c_str(), mp);
    if (!model) return 3;
    auto cp = common_context_params_to_llama(params);
    trace_state state{std::getenv("Z6_LOGITS_TRACE"), 0};
    if (state.prefix) { cp.cb_eval = trace; cp.cb_eval_user_data = &state; }
    // Z6_LOGITS_COME_PERPLEXITY=1: stesso tipo di batch di llama-perplexity
    const bool come_pp = std::getenv("Z6_LOGITS_COME_PERPLEXITY") != nullptr;
    if (come_pp) cp.logits_all = true;
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    // Z6_LOGITS_FILE, Z6_LOGITS_N, Z6_LOGITS_BATCH: testo, token e batch unico spezzato in microbatch, con i primi 1024 logit
    if (const char * fp = std::getenv("Z6_LOGITS_FILE")) {
        FILE * tf = std::fopen(fp, "rb");
        if (!tf) return 9;
        std::string text(1 << 20, 0);
        text.resize(std::fread(&text[0], 1, text.size(), tf));
        std::fclose(tf);
        params.prompt = text;
    }
    // Z6_LOGITS_RISCALDA=1: come llama_init_from_gpt_params, decode vuoto e azzeramento prima di iniziare
    if (std::getenv("Z6_LOGITS_RISCALDA")) {
        std::vector<llama_token> tmp = {llama_token_bos(model) != -1 ? llama_token_bos(model) : llama_token_eos(model)};
        if (llama_decode(ctx, llama_batch_get_one(tmp.data(), (int32_t)tmp.size(), 0, 0)) != 0) return 10;
        llama_kv_cache_clear(ctx);
        llama_synchronize(ctx);
        llama_reset_timings(ctx);
    }
    const char * n_env = std::getenv("Z6_LOGITS_N");
    const char * b_env = std::getenv("Z6_LOGITS_BATCH");
    const size_t n_max = n_env ? std::atoi(n_env) : 16;
    auto tokens = common_tokenize(ctx, params.prompt, true);
    if (tokens.size() > n_max) tokens.resize(n_max);
    const uint32_t passi = std::getenv("Z6_LOGITS_OGNI") ? std::atoi(std::getenv("Z6_LOGITS_OGNI")) : 1;
    const uint32_t shape[2] = {(uint32_t)((tokens.size() + passi - 1) / passi), n_env && !std::getenv("Z6_LOGITS_TUTTI") ? 1024u : (uint32_t)llama_n_vocab(model)};
    FILE * out = std::fopen(path, "wb");
    if (!out || std::fwrite(shape, sizeof(shape), 1, out) != 1) return 5;
    if (b_env && come_pp) {
        const size_t passo = std::atoi(b_env);
        for (size_t i0 = 0; i0 < tokens.size(); i0 += passo) {
            const size_t n = std::min(passo, tokens.size() - i0);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i0, n, i0, 0)) != 0) return 6;
            auto logits = llama_get_logits(ctx);
            for (size_t j = 0; j < n; ++j) {
                if (!logits || std::fwrite(logits + j * llama_n_vocab(model), sizeof(float), shape[1], out) != shape[1]) return 7;
            }
        }
        std::fclose(out);
        std::printf("OK logits %u x %u f32\n", shape[0], shape[1]);
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 0;
    }
    if (b_env) {
        const size_t passo = std::atoi(b_env);
        llama_batch b = llama_batch_init(passo, 0, 1);
        for (size_t i0 = 0; i0 < tokens.size(); i0 += passo) {
            const size_t n = std::min(passo, tokens.size() - i0);
            common_batch_clear(b);
            for (size_t j = 0; j < n; ++j) common_batch_add(b, tokens[i0 + j], i0 + j, { 0 }, true);
            if (llama_decode(ctx, b) != 0) return 6;
            for (size_t j = 0; j < n; ++j) {
                auto logits = llama_get_logits_ith(ctx, j);
                if (!logits || std::fwrite(logits, sizeof(float), shape[1], out) != shape[1]) return 7;
            }
        }
        std::fclose(out);
        std::printf("OK logits %u x %u f32\n", shape[0], shape[1]);
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 0;
    }
    for (size_t i = 0; i < tokens.size(); ++i) {
        state.step = i;
        auto batch = llama_batch_get_one(&tokens[i], 1, i, 0);
        if (llama_decode(ctx, batch) != 0) return 6;
        auto logits = llama_get_logits(ctx);
        // Z6_LOGITS_OGNI=<p>: salva un token ogni p per i contesti lunghi
        static const int ogni = std::getenv("Z6_LOGITS_OGNI") ? std::atoi(std::getenv("Z6_LOGITS_OGNI")) : 1;
        if (i % ogni) continue;
        if (!logits || std::fwrite(logits, sizeof(float), shape[1], out) != shape[1]) return 7;
    }
    std::fclose(out);
    std::printf("OK logits %u x %u f32\n", shape[0], shape[1]);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
