#include "common.h"
#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

bool z6_glm_print_layout(const llama_model * model);

static void require(bool ok, const char * what) { if (!ok) throw std::runtime_error(what); }
static void write(const std::string & path, const void * data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    require(bool(f.write(static_cast<const char *>(data), n)), "write dump");
}
// No eval callback: it would silently disable the pair path. Sequence snapshots
// include KDA+conv, MLA K and DSA kr; emit after synchronization, between decodes.
int main(int argc, char ** argv) {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_batch batch{};
    bool batch_live = false;
    int status = 1;
    try {
        gpt_params params;
        if (!gpt_params_parse(argc, argv, params)) return 1;
        const char * out = std::getenv("Z6_GLM_PAIR_OUT");
        require(out, "set Z6_GLM_PAIR_OUT to an existing directory");
        require(params.n_batch >= 4097 && params.n_ubatch == 1024 && params.n_ctx >= 10306,
                "requires -b 4097 -ub 1024 -c >=10306");
        llama_backend_init(); llama_numa_init(params.numa);
        model = llama_model_load_from_file(params.model.c_str(), common_model_params_to_llama(params));
        require(model, "model load");
        require(z6_glm_print_layout(model), "requires glm5next");
        auto cp = common_context_params_to_llama(params);
        cp.logits_all = true;
        ctx = llama_init_from_model(model, cp);
        require(ctx, "context load");
        auto tokens = common_tokenize(ctx, params.prompt, true);
        require(tokens.size() >= 10306, "provide a corpus with >=10306 model tokens (-f)");
        const uint32_t vocab = llama_n_vocab(model);
        require(vocab == 154880, "wrong GLM vocabulary/model");
        batch = llama_batch_init(4097, 0, 1); batch_live = true;
        const std::string dir(out);
        auto snapshot = [&](const std::string & tag) {
            llama_synchronize(ctx);
            std::vector<uint8_t> bytes(llama_state_seq_get_size(ctx, 0, 0));
            require(!bytes.empty(), "empty snapshot");
            const size_t n = llama_state_seq_get_data(ctx, bytes.data(), bytes.size(), 0, 0);
            require(n == bytes.size(), "snapshot size");
            write(dir+"/"+tag+".state",bytes.data(),bytes.size());
            return bytes;
        };
        auto decode = [&](int start, int n, const std::string & tag) {
            common_batch_clear(batch);
            for (int j = 0; j < n; ++j) common_batch_add(batch,tokens[start+j],start+j,{0},true);
            require(llama_decode(ctx,batch) == 0, "decode");
            std::ofstream f(dir+"/"+tag+".f32",std::ios::binary);
            const uint32_t shape[2] = {uint32_t(n),vocab};
            require(bool(f.write(reinterpret_cast<const char *>(shape),sizeof(shape))),"logit header");
            for (int j = 0; j < n; ++j) {
                const float * logits = llama_get_logits_ith(ctx,j);
                require(logits && bool(f.write(reinterpret_cast<const char *>(logits),4*vocab)),"logit row");
            }
            std::printf("DUMP %s pos=%d tokens=%d vocab=%u\n",tag.c_str(),start,n,vocab);
        };
        decode(0,4096,"a"); snapshot("a");
        decode(4096,4097,"b"); auto saved = snapshot("b");
        decode(8193,2049,"c"); auto forward = snapshot("c");
        require(llama_state_seq_set_data(ctx,saved.data(),saved.size(),0,0) == saved.size(),"restore");
        decode(8193,2049,"r"); auto restored = snapshot("r");
        require(forward == restored,"state mismatch after restore");
        // Exercise ordinary one-token generation after the pair, teacher forced.
        for (int i = 0; i < 64; ++i) decode(10242+i,1,"g"+std::to_string(i));
        snapshot("g");
        std::printf("PASS snapshot restore; compare ALL dumps with serial two-slot and single-slot controls\n");
        status = 0;
    } catch (const std::exception & e) { std::fprintf(stderr,"FAIL %s\n",e.what()); }
    if (batch_live) llama_batch_free(batch);
    if (ctx) llama_free(ctx);
    if (model) llama_free_model(model);
    llama_backend_free();
    return status;
}
