#include "../ggml/src/iqk/iqk_gemm_legacy_quants.cpp"
#include "../ggml/src/iqk/iqk_gemm_kquants.h"
#include "../ggml/src/iqk/iqk_z6_q8_panel.h"
#include <random>
#include <cstdio>
#ifdef HAVE_FANCY_SIMD
template<int N> bool check(std::mt19937 & rng) {
    for (int rows:{16,32,64}) for (int nb:{1,10}) for (int trial=0;trial<32;++trial) {
        std::vector<block_q4_K> x(rows*(nb+1));
        for (auto & v:x) {
            for (auto & q:v.qs) q=trial<4 ? trial*85 : rng();
            for (auto & q:v.scales) q=trial<4 ? (trial&1?255:0) : rng();
            v.d=trial==0?0:0x1000+rng()%0x2000; v.dmin=trial==1?0:0x1000+rng()%0x2000;
        }
        std::vector<block_q8_1_r8> old(rows*nb);
        std::vector<z6_q8_panel16> panel(rows*nb/2);
        if (!iqk_convert_kquants_q8X_r8(GGML_TYPE_Q4_K,nb*256,x.data(),(nb+1)*sizeof(block_q4_K),old.data(),rows)) return false;
        z6_q4k_panel16(nb*256,x.data(),(nb+1)*sizeof(block_q4_K),panel.data(),rows);
        // Canonicalize to the old layout: compare all metadata and quant bytes.
        for (int r=0;r<rows;r+=16) for (int j=0;j<nb*8;++j) for (int half=0;half<2;++half) {
            const auto & a=old[(r/8+half)*nb*8+j]; const auto & b=panel[r/16*nb*8+j];
            if (memcmp(a.d,b.d+half*8,16)||memcmp(a.d+8,b.m+half*8,16)) return false;
            for (int l=0;l<8;++l) if (memcmp(a.qs+32*l,b.qs+64*l+32*half,32)) return false;
        }
        std::vector<block_q8_2_x4> y(N*nb*2);
        for (auto & v:y) for (int j=0;j<4;++j) {
            v.d[j]=0x3b00+rng()%0x300; int sum=0;
            for (int l=0;l<32;++l) { int q=trial<4?((l&1)?127:-127):int(rng()%255)-127; v.qs[32*j+l]=q; sum+=q; }
            v.d[j+4]=sum;
        }
        std::vector<float> ref((rows+3)*N,12345.f),got=ref;
        DataInfo a{ref.data(),(const char *)y.data(),size_t(rows+3),nb*2*sizeof(block_q8_2_x4),0,1,nullptr,0};
        auto b=a; b.s=got.data();
        size_t stride=nb*sizeof(block_q8_1_r8);
        mul_mat_q8_1_r8_q8_2<N>(nb*256,old.data(),stride,a,rows);
        z6_panel_kernel<N>(nb*256,panel.data(),stride,b,rows);
        if (memcmp(ref.data(),got.data(),ref.size()*4)) { fprintf(stderr,"FAIL N=%d rows=%d nb=%d trial=%d\n",N,rows,nb,trial); return false; }
    }
    return true;
}
#endif
int main() {
#ifdef HAVE_FANCY_SIMD
    auto ctx=ggml_init({1024*1024,nullptr,true}); if(!ctx) return 2; ggml_free(ctx);
    std::mt19937 rng(1303);
    bool ok=check<2>(rng)&&check<3>(rng)&&check<4>(rng)&&check<5>(rng)&&check<6>(rng)&&check<7>(rng)&&check<8>(rng);
    puts(ok?"PASS panel16: canonical bytes + output memcmp":"FAIL panel16"); return ok?0:1;
#else
    return 77;
#endif
}
