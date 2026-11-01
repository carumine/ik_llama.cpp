#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpp.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

static void check(bool ok, const char * why) {
    if (!ok) { fprintf(stderr, "FAIL %s\n", why); std::exit(1); }
}
struct pin_free { void operator()(void * p) const { if (p) ggml_backend_cuda_z6_host_free(p); } };

static void run(ggml_backend_t gpu, ggml_backend_t cpu, int nt, bool gpu_slow) {
    constexpr int n = 256, nk = 2;
    ggml_context_ptr weights(ggml_init({1024*1024, nullptr, true}));
    auto matrix = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F16, n, n);
    auto input = ggml_new_tensor_3d(weights.get(), GGML_TYPE_F32, n, 1, nt);
    auto ids = ggml_new_tensor_2d(weights.get(), GGML_TYPE_I32, nk, nt);
    auto w = ggml_new_tensor_3d(weights.get(), GGML_TYPE_F32, 1, nk, nt);
    ggml_backend_buffer_ptr wb(ggml_backend_alloc_ctx_tensors(weights.get(), gpu));
    check(bool(wb), "weights allocation");
    std::vector<ggml_fp16_t> md(n*n, ggml_fp32_to_fp16(0));
    for (int i = 0; i < n; ++i) md[i*n+i] = ggml_fp32_to_fp16(1);
    ggml_backend_tensor_set(matrix, md.data(), 0, md.size()*sizeof(md[0]));

    std::unique_ptr<void,pin_free> pinned(ggml_backend_cuda_z6_host_alloc(4*1024*1024));
    check(bool(pinned), "pinned packet allocation");
    ggml_backend_buffer_ptr hb(ggml_backend_cpu_buffer_from_ptr(pinned.get(), 4*1024*1024));
    ggml_context_ptr ctx(ggml_init({16*1024*1024, nullptr, true}));
    std::vector<float> cm(n*n,0);
    for (int i=0;i<n;++i) cm[i*n+i]=1;
    ggml_backend_buffer_ptr cb(ggml_backend_alloc_buffer(cpu,cm.size()*4));
    auto cpu_matrix=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,n,n);
    cpu_matrix->buffer=cb.get();cpu_matrix->data=ggml_backend_buffer_get_base(cb.get());
    ggml_backend_tensor_set(cpu_matrix,cm.data(),0,cm.size()*4);
    ggml_backend_t backends[] = {gpu,cpu};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 4096, false));
    ggml_backend_sched_z6_begin_graph(sched.get());
    auto host = [&](ggml_type type, int64_t a, int64_t b, int64_t c, size_t offset) {
        auto t = ggml_new_tensor_3d(ctx.get(), type, a,b,c);
        t->buffer = hb.get(); t->data = (char *)pinned.get()+offset;
        return t;
    };
    auto hx = host(GGML_TYPE_F32,n,1,nt,0);
    auto hi = host(GGML_TYPE_I32,nk,nt,1,64*1024);
    auto hw = host(GGML_TYPE_F32,1,nk,nt,128*1024);
    auto x = ggml_scale(ctx.get(), input, 1.25f);
    ggml_backend_sched_set_tensor_backend(sched.get(), x, gpu);
    auto g = ggml_new_graph_custom(ctx.get(),4096,false);
    ggml_build_forward_expand(g,x); ggml_build_forward_expand(g,ids); ggml_build_forward_expand(g,w);
    auto hot = ggml_reshape_2d(ctx.get(),x,n,nt);
    ggml_tensor * first = nullptr;
    std::vector<ggml_tensor *> gpu_nodes{x}, cpu_nodes;
    // Many real GPU operations keep the hot tail pending while the CPU starts.
    for (int i=0;i<(gpu_slow?200:1);++i) {
        hot=ggml_mul_mat(ctx.get(),matrix,hot);
        ggml_backend_sched_set_tensor_backend(sched.get(),hot,gpu);
        gpu_nodes.push_back(hot);
        if (!first) { first=hot; ggml_set_name(hot,"z6_caldi_up_gate-test"); }
    }
    ggml_build_forward_expand(g,hot);
    auto cold=ggml_scale(ctx.get(),ggml_reshape_2d(ctx.get(),hx,n,nt),2);
    auto cold_first=cold; cpu_nodes.push_back(cold);
    ggml_backend_sched_set_tensor_backend(sched.get(),cold,cpu);
    if (!gpu_slow) for (int i=0;i<40;++i) {
        cold=ggml_mul_mat(ctx.get(),cpu_matrix,cold); cpu_nodes.push_back(cold);
        ggml_backend_sched_set_tensor_backend(sched.get(),cold,cpu);
    }
    cold->buffer=hb.get(); cold->data=(char *)pinned.get()+256*1024;
    ggml_backend_sched_set_tensor_backend(sched.get(),cold,cpu);
    ggml_z6_caldi_group group={first,hot,cold_first,cold,{x,ids,w},{hx,hi,hw},0};
    ggml_backend_sched_z6_add_group(sched.get(),&group);
    auto result=ggml_add(ctx.get(),hot,cold);
    ggml_backend_sched_set_tensor_backend(sched.get(),result,gpu);
    ggml_build_forward_expand(g,result); gpu_nodes.push_back(result);
    for (int epoch=0;epoch<12;++epoch) {
        for (auto t : gpu_nodes) ggml_backend_sched_set_tensor_backend(sched.get(),t,gpu);
        for (auto t : cpu_nodes) ggml_backend_sched_set_tensor_backend(sched.get(),t,cpu);
        if (epoch==0) check(ggml_backend_sched_alloc_graph(sched.get(),g),"explicit assignments allocated before compute");
        std::vector<float> xd(n*nt),wd(nk*nt),out(n*nt);
        std::vector<int32_t> id(nk*nt);
        for (size_t i=0;i<xd.size();++i) xd[i]=float((i+epoch)%17)/8;
        for (size_t i=0;i<wd.size();++i) { wd[i]=float(epoch+i)/16;id[i]=(epoch+i)%2 ? -1 : 1; }
        ggml_backend_tensor_set(input,xd.data(),0,xd.size()*4);
        ggml_backend_tensor_set(ids,id.data(),0,id.size()*4);
        ggml_backend_tensor_set(w,wd.data(),0,wd.size()*4);
        check(ggml_backend_sched_graph_compute(sched.get(),g)==GGML_STATUS_SUCCESS,"scheduled compute");
        ggml_backend_tensor_get(result,out.data(),0,out.size()*4);
        for (size_t i=0;i<out.size();++i) if (!std::isfinite(out[i]) || std::abs(out[i]-3.75f*xd[i])>=1e-5f) {
            fprintf(stderr,"batch %d slow %d epoch %d i %zu: x %g hx %g cold %g out %g expected %g\n",nt,gpu_slow,epoch,i,xd[i],((float *)hx->data)[i],((float *)cold->data)[i],out[i],3.75f*xd[i]);
            check(false,"fresh activation/output per epoch");
        }
        for (size_t i=0;i<id.size();++i) {
            check(((int32_t *)hi->data)[i]==id[i],"fresh ID packet");
            check(((float *)hw->data)[i]==wd[i],"fresh router weight packet");
        }
    }
    int64_t total=0,overlap=0;
    ggml_backend_sched_z6_stats(sched.get(),&total,&overlap);
    check(total==12 && (!gpu_slow || overlap>0),"CPU must start before the GPU hot tail finishes");
    ggml_backend_sched_reset(sched.get()); // drain before freeing external host storage
    printf("OK batch=%d slow=%s: %lld/%lld CPU starts with hot GPU pending; 12 fresh epochs, CUDA graph reuse\n",nt,gpu_slow?"GPU":"CPU",(long long)overlap,(long long)total);
}
int main() {
    setenv("GGML_Z6_TEMPI","1",1);
    ggml_backend_ptr gpu(ggml_backend_reg_init_backend_from_str("CUDA0"));
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    check(bool(gpu)&&bool(cpu),"backend init");
    ggml_backend_cpu_set_n_threads(cpu.get(),4);
    for (int nt:{1,2,8,32}) for (bool slow:{true,false}) run(gpu.get(),cpu.get(),nt,slow);
}
