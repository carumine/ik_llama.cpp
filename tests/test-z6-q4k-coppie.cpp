#include "../ggml/src/iqk/iqk_gemm_kquants.cpp"
#include <cstdio>
#include <random>
#ifdef HAVE_FANCY_SIMD
template<int N> bool check(std::mt19937 & rng) {
    for (int rows : {1,2,3,16,31,32,64}) for (int nb : {1,10}) for (int trial=0;trial<32;++trial) {
        std::vector<block_q4_K> x(rows*(nb+1));
        std::vector<block_q8_2_x4> y(N*2*nb);
        for (auto & v:x) {
            for (auto & q:v.qs) q=trial<4 ? trial*85 : rng();
            for (auto & q:v.scales) q=rng();
            const uint16_t edges[]={0,0x8000,1,0x3ff,0x400,0x3c00,0x7bff,0xfbff};
            v.d=trial<8?edges[trial]:0x1000+rng()%0x2000;
            v.dmin=trial<8?edges[(trial+3)%8]:0x1000+rng()%0x2000;
        }
        for (auto & v:y) for (int j=0;j<4;++j) {
            v.d[j]=0x3a00+rng()%0x400;
            int sum=0;
            for (int k=0;k<32;++k) {
                int q=trial<4 ? ((k&1)?127:-127) : int(rng()%255)-127;
                v.qs[32*j+k]=q; sum+=q;
            }
            v.d[4+j]=sum;
        }
        std::vector<float> ref((rows+3)*N,12345.f),got=ref;
        DataInfo a{ref.data(),(const char *)y.data(),size_t(rows+3),2*nb*sizeof(block_q8_2_x4),0,1,nullptr,0};
        auto b=a; b.s=got.data();
        mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2,N>(256*nb,x.data(),(nb+1)*sizeof(block_q4_K),a,rows);
        z6_mul_mat_q4k_pairs<N>(256*nb,x.data(),(nb+1)*sizeof(block_q4_K),b,rows);
        if (memcmp(ref.data(),got.data(),ref.size()*4)) {
            fprintf(stderr,"FAIL N=%d rows=%d nb=%d trial=%d\n",N,rows,nb,trial); return false;
        }
    }
    return true;
}
#endif
int main() {
#ifdef HAVE_FANCY_SIMD
    auto ctx=ggml_init({1024*1024,nullptr,true}); if(!ctx) return 2; ggml_free(ctx);
    std::mt19937 rng(1302);
    bool ok=check<1>(rng)&&check<2>(rng)&&check<3>(rng)&&check<4>(rng)&&check<5>(rng)&&check<6>(rng)&&check<7>(rng)&&check<8>(rng);
    puts(ok?"PASS pairs: memcmp, including odd tails":"FAIL pairs"); return ok?0:1;
#else
    return 77;
#endif
}
