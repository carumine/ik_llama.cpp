// Z6 (GGML_Z6_IQ3S_LUT=1): IQ3_S reordered to 4 bits per weight, GGML_TYPE_IQ3_S_Z6 (block_iq3_s_z6 in ggml-common.h).
//
// Included by ggml-quants.c and by tests/test-z6-iq3s-lut.cpp, after ggml-common.h with the tables (iq3s_grid) and
// after the definition of GGML_FP16_TO_FP32. Plain C that is also valid C++.
//
// The reordering is lossless: each byte of an iq3s_grid entry is the magnitude 1, 3, ..., 15 of one weight and the
// 512 entries are distinct, so (magnitude index, sign) per weight carries the same information as (grid index, sign
// mask) and the original block can be rebuilt byte for byte. The grid is checked once; if a future grid breaks either
// property the conversions return false and the loader keeps IQ3_S.
//
// Layout: byte b of sub-block s (64 weights) holds weight 64*s+b in the low nibble and weight 64*s+32+b in the high
// one; in a nibble, bits 0-2 are the magnitude index m (value 2*m+1) and bit 3 is the sign.
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

static uint16_t z6_iq3s_inverse[4096];   // 4 magnitude indices of 3 bits (bytes in memory order) -> grid index
static uint32_t z6_iq3s_sign4[16];       // 4 sign bits -> 0x08 in the byte of each negative weight (memory order)
static int      z6_iq3s_state = 0;       // 0: not checked, 1: usable, -1: grid not invertible

static inline bool z6_iq3s_prepare(void) {
    if (z6_iq3s_state != 0) return z6_iq3s_state > 0;
    // Concurrent first calls write the same values; the state is published last.
    bool ok = true;
    for (int k = 0; k < 4096; ++k) z6_iq3s_inverse[k] = 0xffff;
    for (int i = 0; i < 512 && ok; ++i) {
        const uint8_t * g = (const uint8_t *)(iq3s_grid + i);
        unsigned k = 0;
        for (int b = 0; b < 4; ++b) {
            if (!(g[b] & 1) || g[b] > 15) ok = false;
            k |= (unsigned)(g[b] >> 1) << (3*b);
        }
        if (ok && z6_iq3s_inverse[k] != 0xffff) ok = false;
        if (ok) z6_iq3s_inverse[k] = (uint16_t)i;
    }
    for (int k = 0; k < 16; ++k) {
        uint8_t bytes[4];
        for (int b = 0; b < 4; ++b) bytes[b] = (k >> b) & 1 ? 0x08 : 0x00;
        memcpy(z6_iq3s_sign4 + k, bytes, 4);
    }
    z6_iq3s_state = ok ? 1 : -1;
    return ok;
}

// IQ3_S -> reordered. x and y must not overlap.
static inline bool z6_iq3s_from_iq3s(const block_iq3_s * x, block_iq3_s_z6 * y, int64_t nb) {
    if (!z6_iq3s_prepare()) return false;
    for (int64_t i = 0; i < nb; ++i) {
        uint8_t w[QK_K];   // one nibble per weight, in weight order
        for (int t = 0; t < QK_K/4; ++t) {
            const uint32_t g = iq3s_grid[x[i].qs[t] | (((x[i].qh[t/8] >> (t%8)) & 1u) << 8)];
            // odd magnitudes: (m - 1)/2 == m >> 1; weight 4*t+b has sign bit 4*(t&1)+b of signs[t/2]
            const uint32_t n4 = ((g >> 1) & 0x07070707u) | z6_iq3s_sign4[(x[i].signs[t/2] >> (4*(t&1))) & 15];
            memcpy(w + 4*t, &n4, 4);
        }
        y[i].d = x[i].d;
        memcpy(y[i].scales, x[i].scales, IQ3S_N_SCALE);
        for (int s = 0; s < QK_K/64; ++s) {
            for (int b = 0; b < 32; ++b) y[i].nib[32*s + b] = (uint8_t)(w[64*s + b] | (w[64*s + 32 + b] << 4));
        }
    }
    return true;
}

// Reordered -> IQ3_S, the bytes of the file. x and y must not overlap.
static inline bool z6_iq3s_to_iq3s(const block_iq3_s_z6 * x, block_iq3_s * y, int64_t nb) {
    if (!z6_iq3s_prepare()) return false;
    for (int64_t i = 0; i < nb; ++i) {
        uint8_t w[QK_K];
        for (int s = 0; s < QK_K/64; ++s) {
            for (int b = 0; b < 32; ++b) {
                w[64*s + b]      = x[i].nib[32*s + b] & 15;
                w[64*s + 32 + b] = x[i].nib[32*s + b] >> 4;
            }
        }
        memset(y + i, 0, sizeof(block_iq3_s));
        y[i].d = x[i].d;
        memcpy(y[i].scales, x[i].scales, IQ3S_N_SCALE);
        for (int t = 0; t < QK_K/4; ++t) {
            unsigned k = 0, sg = 0;
            for (int b = 0; b < 4; ++b) {
                k  |= (unsigned)(w[4*t + b] & 7) << (3*b);
                sg |= (unsigned)(w[4*t + b] >> 3) << b;
            }
            const unsigned idx = z6_iq3s_inverse[k];
            if (idx == 0xffff) return false;   // a combination of magnitudes that is not in the grid
            y[i].qs[t] = (uint8_t)(idx & 0xff);
            y[i].qh[t/8]    = (uint8_t)(y[i].qh[t/8]    | ((idx >> 8) << (t%8)));
            y[i].signs[t/2] = (uint8_t)(y[i].signs[t/2] | (sg << (4*(t&1))));
        }
    }
    return true;
}

static inline unsigned z6_iq3s_nibble(const block_iq3_s_z6 * x, int w) {
    const int s = w/64, r = w%64;
    return r < 32 ? x->nib[32*s + r] & 15u : (unsigned)x->nib[32*s + r - 32] >> 4;
}

// Same products in the same order as dequantize_row_iq3_s: (d * scale) * magnitude * sign.
static inline void z6_iq3s_dequantize_row(const block_iq3_s_z6 * x, float * y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const float db = d * (1 + 2*((x[i].scales[ib32/2] >> 4*(ib32%2)) & 0xf));
            for (int j = 0; j < 32; ++j) {
                const unsigned n = z6_iq3s_nibble(x + i, 32*ib32 + j);
                y[j] = db * (uint8_t)(2*(n & 7) + 1) * (n & 8 ? -1.f : 1.f);
            }
            y += 32;
        }
    }
}
