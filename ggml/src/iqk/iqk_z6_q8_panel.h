#pragma once
#include "iqk_common.h"
#ifdef HAVE_FANCY_SIMD
// Private temporary layout, never a tensor/GGUF type. Same bytes as two r8 blocks.
struct z6_q8_panel16 { ggml_half d[16], m[16]; uint8_t qs[512]; };
static_assert(sizeof(z6_q8_panel16)==576, "panel size");
static inline void z6_q4k_panel16(int n,const void * vx,size_t bx,void * vy,int rows) {
    GGML_ASSERT(n%256==0 && rows%16==0);
    auto y=(z6_q8_panel16 *)vy;
    for (int ix=0;ix<rows;ix+=16) for (int i=0;i<n/256;++i) {
        for (int half=0;half<2;++half) {
            ggml_half dh[16]; uint16_t ls[128]; __m256 bits[8][8];
            for (int r=0;r<8;++r) {
                const auto & x=((const block_q4_K *)((const char *)vx+(ix+8*half+r)*bx))[i];
                dh[r]=x.d; dh[r+8]=x.dmin;
                uint32_t tmp[4]; make_q4_scales(x.scales,tmp);
                auto u=(const uint8_t *)tmp;
                for (int j=0;j<8;++j) { ls[8*j+r]=u[j]; ls[8*j+r+64]=u[j+8]; }
                for (int j=0;j<4;++j) {
                    auto v=_mm256_loadu_si256((const __m256i *)x.qs+j);
                    bits[2*j][r]=_mm256_castsi256_ps(_mm256_and_si256(v,_mm256_set1_epi8(15)));
                    bits[2*j+1][r]=_mm256_castsi256_ps(_mm256_and_si256(_mm256_srli_epi16(v,4),_mm256_set1_epi8(15)));
                }
            }
            auto vd=_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)dh));
            auto vm=_mm256_mul_ps(_mm256_set1_ps(-1.f),_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(dh+8))));
            for (int j=0;j<8;++j) {
                auto ds=_mm256_mul_ps(vd,_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)(ls+8*j)))));
                auto ms=_mm256_mul_ps(vm,_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)(ls+64+8*j)))));
                _mm_storeu_si128((__m128i *)(y[j].d+8*half),_mm256_cvtps_ph(ds,_MM_FROUND_TO_NEAREST_INT));
                _mm_storeu_si128((__m128i *)(y[j].m+8*half),_mm256_cvtps_ph(ms,_MM_FROUND_TO_NEAREST_INT));
                iqk_transpose_8x8(bits[j]);
                for (int l=0;l<8;++l) _mm256_storeu_si256((__m256i *)(y[j].qs+64*l+32*half),_mm256_castps_si256(bits[j][l]));
            }
        }
        y+=8;
    }
}
inline __m512i z6_panel_dot(const __m512i * qx, const int8_t * y) {
    auto s0 = _mm512_setzero_si512();
    auto s1 = _mm512_setzero_si512();
    s0 = _mm512_dpbusd_epi32(s0, qx[0], _mm512_set1_epi32(*(const int32_t *)(y +  0)));
    s1 = _mm512_dpbusd_epi32(s1, qx[1], _mm512_set1_epi32(*(const int32_t *)(y +  4)));
    s0 = _mm512_dpbusd_epi32(s0, qx[2], _mm512_set1_epi32(*(const int32_t *)(y +  8)));
    s1 = _mm512_dpbusd_epi32(s1, qx[3], _mm512_set1_epi32(*(const int32_t *)(y + 12)));
    s0 = _mm512_dpbusd_epi32(s0, qx[4], _mm512_set1_epi32(*(const int32_t *)(y + 16)));
    s1 = _mm512_dpbusd_epi32(s1, qx[5], _mm512_set1_epi32(*(const int32_t *)(y + 20)));
    s0 = _mm512_dpbusd_epi32(s0, qx[6], _mm512_set1_epi32(*(const int32_t *)(y + 24)));
    s1 = _mm512_dpbusd_epi32(s1, qx[7], _mm512_set1_epi32(*(const int32_t *)(y + 28)));
    return _mm512_add_epi32(s0, s1);
}
template<int nrc_y>
static void z6_panel_kernel(int n,const void * vx,size_t bx,const DataInfo & info,int nrc_x) {
    static_assert(nrc_y>1, "single-token path retains r8");
    GGML_ASSERT(nrc_x%16==0);
    Q8<nrc_y,block_q8_2_x4> q8(info);
    const int nb=n/32;
        __m512  acc[nrc_y] = {};
        __m512i qx[8];
        float d8[4*nrc_y];
        for (int ix = 0; ix < nrc_x; ix += 16) {
            const z6_q8_panel16 * panel = (const z6_q8_panel16 *)((const char *)vx + ix*bx);
            for (int i4 = 0; i4 < nb/4; ++i4) {
                {
                    __m512 mx[4];
                    for (int ib32 = 0; ib32 < 4; ++ib32) {
                        mx[ib32] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)panel[4*i4+ib32].m));
                    }
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto scales128 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)q8.y[iy][i4].d)), 16));
                        _mm_storeu_ps(d8 + 4*iy, scales128);
                        auto bsums4 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(q8.y[iy][i4].d+4))));
                        bsums4 = _mm_mul_ps(bsums4, scales128);
                        auto bsums256 = _mm256_set_m128(bsums4, bsums4);
                        auto bsums = _mm512_insertf32x8(_mm512_castps256_ps512(bsums256), bsums256, 1);
                        acc[iy] = _mm512_fmadd_ps(mx[0], _mm512_shuffle_ps(bsums, bsums, 0x00), acc[iy]);
                        acc[iy] = _mm512_fmadd_ps(mx[1], _mm512_shuffle_ps(bsums, bsums, 0x55), acc[iy]);
                        acc[iy] = _mm512_fmadd_ps(mx[2], _mm512_shuffle_ps(bsums, bsums, 0xaa), acc[iy]);
                        acc[iy] = _mm512_fmadd_ps(mx[3], _mm512_shuffle_ps(bsums, bsums, 0xff), acc[iy]);
                    }
                }
                for (int ib32 = 0; ib32 < 4; ++ib32) {
                    auto scales = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)panel[4*i4+ib32].d));
                    for (int j=0;j<8;++j) qx[j]=_mm512_loadu_si512((const void *)(panel[4*i4+ib32].qs+64*j));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto sumi = z6_panel_dot(qx, q8.y[iy][i4].qs+32*ib32);
                        auto dy = _mm512_set1_ps(d8[4*iy+ib32]);
                        acc[iy] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[iy]);
                    }
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                info.store(ix, iy, acc[iy]);
                acc[iy] = _mm512_setzero_ps();
            }
        }
}
static inline bool z6_select_panel16(int typeA,int typeB,int Ny,std::array<mul_mat_t,IQK_MAX_NY> & f) {
    static const bool enabled=[] { const char * e=getenv("GGML_Z6_Q8_PANNELLO16"); return e && strcmp(e,"1")==0; }();
    if (!enabled || typeA!=GGML_TYPE_Q4_K || typeB!=GGML_TYPE_Q8_2_X4 || Ny<32) return false;
    // Existing balancer emits 6/7/8 (or 4..8 for a tiled tail), never 1.
    f[0]=nullptr;
    f[1]=z6_panel_kernel<2>; f[2]=z6_panel_kernel<3>; f[3]=z6_panel_kernel<4>;
    f[4]=z6_panel_kernel<5>; f[5]=z6_panel_kernel<6>; f[6]=z6_panel_kernel<7>; f[7]=z6_panel_kernel<8>;
    return true;
}
#else
static inline bool z6_select_panel16(int,int,int,std::array<mul_mat_t,IQK_MAX_NY> &) { return false; }
#endif
