// Z6 (GGML_Z6_IQ3S_LUT=1): the IQ3_S reordering to 4 bits per weight (GGML_TYPE_IQ3_S_Z6) must not change one bit.
//
// Checked with memcmp, on random blocks, on edge blocks and, with --gguf, on the blocks of a real tensor:
//   1. reordering against a decode written here from the grid, inverse of the reordering against the original bytes,
//      dequantization against the IQ3_S one of the library;
//   2. kernels with 1 to 8 columns: reordered (DequantizerIQ3SLut) against IQ3_S of upstream (DequantizerIQ3S) and
//      against the gather kernel of the preset (DequantizerIQ3SZ6);
//   3. conversion to Q8_K_R16 for the prefill: bytes of the reordered conversion against the IQ3_S one
//      (GGML_Z6_IQ3S_GATHER decides which IQ3_S conversion is the reference: run the test with 1 and with 0);
//   4. the same through the library (type table, iqk_mul_mat, iqk_mul_mat_moe with the conversion threshold,
//      iqk_moe_fused_up_gate): what llama-server runs. Skipped with a notice if the library has no reordered type.
//
//   test-z6-iq3s-lut                                    identity on generated blocks
//   test-z6-iq3s-lut --gguf FILE [--tensor NAME] [--mib N]   plus N MiB (default 64) from the head and N from the tail
//                                                       of an IQ3_S tensor of the file (default: the first *_exps*)
//   test-z6-iq3s-lut --banco MIB THREADS ROUNDS         speed of the single-column kernel and of the conversion,
//                                                       gather against table; first half of the threads on CPUs
//                                                       0.., second half on 20.. (Z6: one socket each), memory
//                                                       touched by the thread that reads it
// Exit: 0 identical, 1 different, 2 usage or file error, 77 no AVX512 kernels in this build.

#include "../ggml/src/iqk/iqk_gemm_iquants.cpp"   // kernels and conversions as the library compiles them
#include "../ggml/src/ggml-z6-iq3s.h"            // the reordering the loader uses
#include "../ggml/src/iqk/iqk_mul_mat.h"
#include "ggml.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) && defined(HAVE_FANCY_SIMD) && defined(__linux__)

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

// Present only in a library built with the reordering: the test also links against an older libggml.
extern "C" {
__attribute__((weak)) bool ggml_z6_iq3s_lut_from_iq3s(const void * src, void * dst, int64_t n_blocks);
__attribute__((weak)) bool ggml_z6_iq3s_lut_to_iq3s(const void * src, void * dst, int64_t n_blocks);
__attribute__((weak)) bool ggml_z6_iq3s_lut_retype(struct ggml_tensor * tensor);
}

namespace {

struct Rng {
    uint64_t s;
    uint64_t operator()() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};

// Q8_K_R16 rows are written with aligned stores: 64-byte alignment, as the work buffer of the library
struct Aligned {
    char * p; size_t n;
    explicit Aligned(size_t bytes) : p((char *)aligned_alloc(64, (bytes + 63)/64*64)), n(bytes) { if (!p) abort(); }
    ~Aligned() { free(p); }
    Aligned(const Aligned &) = delete;
    Aligned & operator=(const Aligned &) = delete;
    char * data() { return p; }
    size_t size() const { return n; }
    char & operator[](size_t i) { return p[i]; }
};

int g_fail = 0;
void fail(const char * what, const char * where, long a = -1, long b = -1) {
    if (++g_fail <= 20) fprintf(stderr, "FAIL %s [%s] %ld %ld\n", what, where, a, b);
}

// mode 0: random bytes, d small and finite; 1: fp16 edge values of d; 2: all magnitudes/signs/scales at the extremes
void fill_blocks(std::vector<block_iq3_s> & x, Rng & rng, int mode) {
    static const uint16_t edges[] = {0, 0x8000, 1, 0x3ff, 0x400, 0x3c00, 0x7bff, 0xfbff};
    for (size_t i = 0; i < x.size(); ++i) {
        auto p = (uint8_t *)&x[i];
        for (size_t k = 0; k < sizeof(block_iq3_s); ++k) p[k] = mode == 2 ? (uint8_t)(((i >> (k % 3)) & 1) ? 0xff : 0x00) : (uint8_t)rng();
        x[i].d = mode == 1 ? edges[i % 8] : (uint16_t)(0x1000 + rng() % 0x2000);
    }
}

// activations as the quantization to Q8_K leaves them: values in [-127, 127], sums of the groups of 16
void fill_q8(std::vector<block_q8_K> & y, Rng & rng, int mode) {
    for (auto & q : y) {
        q.d = 0.002f + (rng() % 1000)*1e-5f;
        int total = 0;
        for (int k = 0; k < QK_K; ++k) {
            const int v = mode == 2 ? ((k & 1) ? 127 : -127) : mode == 3 ? 127 : (int)(rng() % 255) - 127;
            q.qs[k] = (int8_t)v;
            total += v;
        }
        for (int g = 0; g < QK_K/16; ++g) {
            int sum = 0;
            for (int k = 0; k < 16; ++k) sum += q.qs[16*g + k];
            q.bsums[g] = (int16_t)sum;
        }
        q.sum = (float)total;
    }
}

// ---- 1. reordering, inverse, dequantization
bool check_reorder(const block_iq3_s * x, size_t n, const char * where) {
    const int before = g_fail;
    std::vector<block_iq3_s_z6> z(n);
    std::vector<block_iq3_s> back(n);
    if (!z6_iq3s_from_iq3s(x, z.data(), (int64_t)n)) { fail("reordering refused", where); return false; }
    for (size_t i = 0; i < n; ++i) {
        if (z[i].d != x[i].d || memcmp(z[i].scales, x[i].scales, IQ3S_N_SCALE) != 0) fail("d or scales", where, (long)i);
        for (int w = 0; w < QK_K; ++w) {
            // decode written here, from the definition of IQ3_S: grid entry of the group of 4, its byte, the sign bit
            const int t = w/4, b = w%4;
            const unsigned idx = x[i].qs[t] | (((x[i].qh[t/8] >> (t%8)) & 1u) << 8);
            const unsigned mag = ((const uint8_t *)(iq3s_grid + idx))[b];
            const unsigned neg = (x[i].signs[w/8] >> (w%8)) & 1u;
            const unsigned got = z6_iq3s_nibble(&z[i], w);
            if (2*(got & 7) + 1 != mag || (got >> 3) != neg) { fail("nibble", where, (long)i, w); break; }
        }
    }
    if (!z6_iq3s_to_iq3s(z.data(), back.data(), (int64_t)n)) fail("inverse refused", where);
    else if (memcmp(back.data(), x, n*sizeof(block_iq3_s)) != 0) fail("inverse differs from the original bytes", where);

    std::vector<float> fa(n*QK_K), fb(n*QK_K);
    ggml_internal_get_type_traits(GGML_TYPE_IQ3_S).to_float(x, fa.data(), (int64_t)n*QK_K);
    z6_iq3s_dequantize_row(z.data(), fb.data(), (int64_t)n*QK_K);
    if (memcmp(fa.data(), fb.data(), fa.size()*sizeof(float)) != 0) fail("dequantization", where);
    return g_fail == before;
}

// ---- 2. kernels with N columns, rows of nb blocks
template <int N> bool check_kernels(const block_iq3_s * x, int rows, int nb, Rng & rng, int q8_mode, const char * where) {
    const int before = g_fail;
    std::vector<block_iq3_s_z6> z((size_t)rows*nb);
    if (!z6_iq3s_from_iq3s(x, z.data(), (int64_t)rows*nb)) { fail("reordering refused", where); return false; }
    std::vector<block_q8_K> y((size_t)N*nb);
    fill_q8(y, rng, q8_mode);
    const size_t bs = rows + 3;
    std::vector<float> ref(bs*N, 12345.f), gat = ref, lut = ref;
    DataInfo a{ref.data(), (const char *)y.data(), bs, nb*sizeof(block_q8_K), 0, 1, nullptr, 0};
    auto b = a; b.s = gat.data();
    auto c = a; c.s = lut.data();
    mul_mat_qX_K_q8_K_IQ<DequantizerIQ3S,    N>(QK_K*nb, x,        nb*sizeof(block_iq3_s),    a, rows);
    mul_mat_qX_K_q8_K_IQ<DequantizerIQ3SZ6,  N>(QK_K*nb, x,        nb*sizeof(block_iq3_s),    b, rows);
    mul_mat_qX_K_q8_K_IQ<DequantizerIQ3SLut, N>(QK_K*nb, z.data(), nb*sizeof(block_iq3_s_z6), c, rows);
    if (memcmp(ref.data(), lut.data(), ref.size()*sizeof(float)) != 0) fail("kernel: table against IQ3_S of upstream", where, N, rows);
    if (memcmp(gat.data(), lut.data(), gat.size()*sizeof(float)) != 0) fail("kernel: table against gather", where, N, rows);
    return g_fail == before;
}

bool check_kernels_all(const block_iq3_s * x, int rows, int nb, Rng & rng, int q8_mode, const char * where) {
    return check_kernels<1>(x, rows, nb, rng, q8_mode, where) & check_kernels<2>(x, rows, nb, rng, q8_mode, where) &
           check_kernels<3>(x, rows, nb, rng, q8_mode, where) & check_kernels<4>(x, rows, nb, rng, q8_mode, where) &
           check_kernels<5>(x, rows, nb, rng, q8_mode, where) & check_kernels<6>(x, rows, nb, rng, q8_mode, where) &
           check_kernels<7>(x, rows, nb, rng, q8_mode, where) & check_kernels<8>(x, rows, nb, rng, q8_mode, where);
}

// ---- 3. conversion to Q8_K_R16 (prefill): rows must be a multiple of 16
bool check_conversion(const block_iq3_s * x, int rows, int nb, const char * where) {
    const int before = g_fail;
    rows -= rows % 16;
    if (rows <= 0) return true;
    std::vector<block_iq3_s_z6> z((size_t)rows*nb);
    if (!z6_iq3s_from_iq3s(x, z.data(), (int64_t)rows*nb)) { fail("reordering refused", where); return false; }
    const size_t bytes = (size_t)(rows/16)*nb*sizeof(block_q8_k_r16);
    Aligned ya(bytes), yb(bytes);
    memset(ya.data(), 0x5a, bytes); memset(yb.data(), 0x5a, bytes);
    iqk_convert_iq3_s_q8_k_r8    (QK_K*nb, x,        nb*sizeof(block_iq3_s),    ya.data(), rows);
    iqk_convert_iq3_s_z6_q8_k_r16(QK_K*nb, z.data(), nb*sizeof(block_iq3_s_z6), yb.data(), rows);
    if (memcmp(ya.data(), yb.data(), bytes) != 0) fail("conversion to Q8_K_R16", where, rows, nb);
    // and through the dispatcher the matrix products call
    memset(ya.data(), 0x5a, bytes); memset(yb.data(), 0x5a, bytes);
    if (!iqk_convert_iquants_q80_r8(GGML_TYPE_IQ3_S,    QK_K*nb, x,        nb*sizeof(block_iq3_s),    ya.data(), rows) ||
        !iqk_convert_iquants_q80_r8(GGML_TYPE_IQ3_S_Z6, QK_K*nb, z.data(), nb*sizeof(block_iq3_s_z6), yb.data(), rows)) {
        fail("conversion dispatcher refused a type", where);
    } else if (memcmp(ya.data(), yb.data(), bytes) != 0) fail("conversion to Q8_K_R16 via dispatcher", where, rows, nb);
    return g_fail == before;
}

// ---- 4. through the library: what the server runs
bool library_has_type() { return ggml_z6_iq3s_lut_from_iq3s && ggml_z6_iq3s_lut_to_iq3s && ggml_z6_iq3s_lut_retype; }

bool check_library(const block_iq3_s * x, int rows, int nb, Rng & rng, const char * where) {
    const int before = g_fail;
    const size_t n = (size_t)rows*nb;
    const int ne00 = QK_K*nb;
    std::vector<block_iq3_s_z6> z(n), z_api(n);
    std::vector<block_iq3_s> back(n);
    if (!z6_iq3s_from_iq3s(x, z.data(), (int64_t)n)) { fail("reordering refused", where); return false; }

    if (ggml_type_size(GGML_TYPE_IQ3_S_Z6) != sizeof(block_iq3_s_z6) || ggml_blck_size(GGML_TYPE_IQ3_S_Z6) != QK_K ||
        ggml_row_size(GGML_TYPE_IQ3_S_Z6, ne00) != nb*sizeof(block_iq3_s_z6) || !ggml_is_quantized(GGML_TYPE_IQ3_S_Z6)) {
        fail("type table", where);
        return false;
    }
    if (!ggml_z6_iq3s_lut_from_iq3s(x, z_api.data(), (int64_t)n) || memcmp(z_api.data(), z.data(), n*sizeof(block_iq3_s_z6)) != 0)
        fail("library reordering", where);
    if (!ggml_z6_iq3s_lut_to_iq3s(z.data(), back.data(), (int64_t)n) || memcmp(back.data(), x, n*sizeof(block_iq3_s)) != 0)
        fail("library inverse", where);
    {
        std::vector<float> fa(n*QK_K), fb(n*QK_K);
        ggml_internal_get_type_traits(GGML_TYPE_IQ3_S).to_float(x, fa.data(), (int64_t)n*QK_K);
        ggml_internal_get_type_traits(GGML_TYPE_IQ3_S_Z6).to_float(z.data(), fb.data(), (int64_t)n*QK_K);
        if (memcmp(fa.data(), fb.data(), fa.size()*sizeof(float)) != 0) fail("library dequantization", where);
    }

    const size_t sa = nb*sizeof(block_iq3_s), sz = nb*sizeof(block_iq3_s_z6), sy = nb*sizeof(block_q8_K);
    for (int ny : {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 32, 33}) {
        std::vector<block_q8_K> y((size_t)ny*nb);
        fill_q8(y, rng, 0);

        // plain matrix product
        std::vector<float> ca((size_t)rows*ny, 12345.f), cb = ca;
        const bool ra = iqk_mul_mat(rows, ny, ne00, GGML_TYPE_IQ3_S,    x,        sa, GGML_TYPE_Q8_K, y.data(), sy, ca.data(), rows, 0, 1);
        const bool rb = iqk_mul_mat(rows, ny, ne00, GGML_TYPE_IQ3_S_Z6, z.data(), sz, GGML_TYPE_Q8_K, y.data(), sy, cb.data(), rows, 0, 1);
        if (!ra || !rb) fail("iqk_mul_mat refused a type", where, ny, rb);
        else if (memcmp(ca.data(), cb.data(), ca.size()*sizeof(float)) != 0) fail("iqk_mul_mat", where, ny, rows);

        // one expert, ny tokens: the cold experts' down path; from GGML_Z6_IQ3S_CONV_MIN columns it converts to Q8
        std::vector<mmid_row_mapping> map(ny);
        for (int k = 0; k < ny; ++k) map[k] = {0, k};
        std::fill(ca.begin(), ca.end(), 12345.f); cb = ca;
        const bool ma = iqk_mul_mat_moe(rows, ny, ne00, 1, GGML_TYPE_IQ3_S,    x,        sa, GGML_TYPE_Q8_K, y.data(), sy,
                ca.data(), rows*sizeof(float), rows*sizeof(float), map.data(), 0, 1);
        const bool mb = iqk_mul_mat_moe(rows, ny, ne00, 1, GGML_TYPE_IQ3_S_Z6, z.data(), sz, GGML_TYPE_Q8_K, y.data(), sy,
                cb.data(), rows*sizeof(float), rows*sizeof(float), map.data(), 0, 1);
        if (!ma || !mb) fail("iqk_mul_mat_moe refused a type", where, ny, mb);
        else if (memcmp(ca.data(), cb.data(), ca.size()*sizeof(float)) != 0) fail("iqk_mul_mat_moe", where, ny, rows);

        // fused up/gate with SiLU: the cold experts' up path. The second half of the rows plays the gate.
        // From the conversion threshold the library requires whole blocks of 16 rows, as the callers provide.
        const int half = rows/2;
        if (half >= 1 && (ny < atoi(getenv("GGML_Z6_IQ3S_CONV_MIN")) || half % 16 == 0)) {
            std::vector<float> ua((size_t)half*ny, 12345.f), ub = ua;
            const bool fa = iqk_moe_fused_up_gate(half, ny, ne00, 1, GGML_UNARY_OP_SILU, GGML_TYPE_IQ3_S,
                    x, (const char *)x + half*sa, sa, GGML_TYPE_Q8_K, y.data(), sy, nullptr, nullptr,
                    ua.data(), half*sizeof(float), half*sizeof(float), map.data(), 0.f, 0.f, 0.f, 0, 1);
            const bool fb = iqk_moe_fused_up_gate(half, ny, ne00, 1, GGML_UNARY_OP_SILU, GGML_TYPE_IQ3_S_Z6,
                    z.data(), (const char *)z.data() + half*sz, sz, GGML_TYPE_Q8_K, y.data(), sy, nullptr, nullptr,
                    ub.data(), half*sizeof(float), half*sizeof(float), map.data(), 0.f, 0.f, 0.f, 0, 1);
            if (!fa || !fb) fail("iqk_moe_fused_up_gate refused a type", where, ny, fb);
            else if (memcmp(ua.data(), ub.data(), ua.size()*sizeof(float)) != 0) fail("iqk_moe_fused_up_gate", where, ny, half);
        }
    }

    // the loader's type switch: strides and size of an unallocated expert tensor
    {
        ggml_init_params ip = {ggml_tensor_overhead()*4, nullptr, true};
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ3_S, ne00, 32, 3);
        const size_t old = ggml_nbytes(t);
        if (!ggml_z6_iq3s_lut_retype(t) || t->type != GGML_TYPE_IQ3_S_Z6 || t->nb[1] != sz || t->nb[2] != 32*sz ||
            ggml_nbytes(t) != old/sizeof(block_iq3_s)*sizeof(block_iq3_s_z6)) fail("retype", where);
        ggml_tensor * f = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 2);
        if (ggml_z6_iq3s_lut_retype(f) || ggml_z6_iq3s_lut_retype(t)) fail("retype accepted a tensor that is not IQ3_S", where);
        ggml_free(ctx);
    }
    return g_fail == before;
}

bool check_all(const block_iq3_s * x, int rows, int nb, Rng & rng, int q8_mode, const char * where) {
    bool ok = check_reorder(x, (size_t)rows*nb, where);
    ok &= check_kernels_all(x, rows, nb, rng, q8_mode, where);
    ok &= check_conversion(x, rows, nb, where);
    if (library_has_type()) ok &= check_library(x, rows, nb, rng, where);
    return ok;
}

// ---- real tensor
int run_gguf(const std::string & file, std::string name, size_t mib, Rng & rng) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = {true, &meta};
    gguf_context * g = gguf_init_from_file(file.c_str(), gp);
    if (!g) { fprintf(stderr, "cannot read %s\n", file.c_str()); return 2; }
    int it = -1;
    if (!name.empty()) it = gguf_find_tensor(g, name.c_str());
    else for (int i = 0; i < gguf_get_n_tensors(g) && it < 0; ++i) {
        if (gguf_get_tensor_type(g, i) == GGML_TYPE_IQ3_S && strstr(gguf_get_tensor_name(g, i), "_exps")) it = i;
    }
    if (it < 0 || gguf_get_tensor_type(g, it) != GGML_TYPE_IQ3_S) {
        fprintf(stderr, "no IQ3_S tensor%s%s in %s\n", name.empty() ? "" : " named ", name.c_str(), file.c_str());
        return 2;
    }
    name = gguf_get_tensor_name(g, it);
    const ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
    if (!t || t->ne[0] % QK_K != 0) { fprintf(stderr, "tensor %s: unexpected shape\n", name.c_str()); return 2; }
    const int nb = (int)(t->ne[0]/QK_K);
    const size_t row = nb*sizeof(block_iq3_s), total = ggml_nbytes(t), offset = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, it);
    size_t rows = std::min(mib*1024*1024/row, total/row);
    rows -= rows % 32;
    if (rows == 0) { fprintf(stderr, "tensor %s: too small\n", name.c_str()); return 2; }
    const int fd = open(file.c_str(), O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    std::vector<block_iq3_s> x(rows*nb);
    bool ok = true;
    for (int part = 0; part < 2; ++part) {
        const size_t at = part == 0 ? 0 : total - rows*row;
        size_t done = 0;
        while (done < rows*row) {
            const ssize_t r = pread(fd, (char *)x.data() + done, rows*row - done, (off_t)(offset + at + done));
            if (r <= 0) { perror("pread"); return 2; }
            done += (size_t)r;
        }
        char where[256];
        snprintf(where, sizeof(where), "%s %s, %zu rows", name.c_str(), part == 0 ? "head" : "tail", rows);
        ok &= check_reorder(x.data(), rows*nb, where);
        ok &= check_conversion(x.data(), (int)rows, nb, where);
        // kernels and library on slices of the real rows: every row for the single column, a slice for the others
        ok &= check_kernels<1>(x.data(), (int)rows, nb, rng, 0, where);
        const int slice = (int)std::min<size_t>(rows, 2048);
        ok &= check_kernels_all(x.data() + (rows - slice)*nb, slice, nb, rng, 0, where);
        if (library_has_type()) ok &= check_library(x.data(), slice, nb, rng, where);
        printf("%s: %s\n", where, ok ? "identical" : "DIFFERENT");
    }
    close(fd);
    gguf_free(g);
    ggml_free(meta);
    return ok ? 0 : 1;
}

// ---- speed: single-column kernel (generation) and conversion (prefill), gather of the preset against the table
struct Work {
    size_t rows = 0;
    double t[4] = {};   // seconds per round, minimum: kernel gather, kernel table, conversion gather, conversion table
    bool same = true;
};

int run_bench(size_t mib, int nth, int rounds) {
    const int nb = 16;   // up/gate of GLM: rows of 4096 weights
    const size_t rows = (mib*1024*1024/(nb*sizeof(block_iq3_s)))/16*16;
    if (rows == 0 || nth < 1 || rounds < 1) return 2;
    std::vector<Work> W(nth);
    std::atomic<int> ready{0}, go{0};
    auto body = [&](int ith) {
        cpu_set_t set; CPU_ZERO(&set);
        const int cpu = nth == 1 ? (getenv("CPU") ? atoi(getenv("CPU")) : sched_getcpu()) : (ith < (nth + 1)/2 ? ith : 20 + ith - (nth + 1)/2);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        Work & w = W[ith];
        w.rows = rows;
        Rng rng{0x9e3779b97f4a7c15ull*(ith + 1)};
        std::vector<block_iq3_s> x(rows*nb);
        std::vector<block_iq3_s_z6> z(rows*nb);
        std::vector<block_q8_K> y(nb);
        fill_blocks(x, rng, 0);
        fill_q8(y, rng, 0);
        if (!z6_iq3s_from_iq3s(x.data(), z.data(), (int64_t)rows*nb)) { w.same = false; ++ready; return; }
        std::vector<float> ra(rows), rb(rows);
        const int tile = 64;   // rows per conversion call, as iqk_moe_fused_up_gate does
        Aligned qa((size_t)(tile/16)*nb*sizeof(block_q8_k_r16)), qb(qa.size());
        ++ready;
        while (!go.load()) {}
        for (int r = 0; r < rounds; ++r) {
            DataInfo a{ra.data(), (const char *)y.data(), rows, nb*sizeof(block_q8_K), 0, 1, nullptr, 0};
            auto b = a; b.s = rb.data();
            const auto t0 = std::chrono::steady_clock::now();
            mul_mat_qX_K_q8_K_IQ<DequantizerIQ3SZ6, 1>(QK_K*nb, x.data(), nb*sizeof(block_iq3_s), a, (int)rows);
            const auto t1 = std::chrono::steady_clock::now();
            mul_mat_qX_K_q8_K_IQ<DequantizerIQ3SLut, 1>(QK_K*nb, z.data(), nb*sizeof(block_iq3_s_z6), b, (int)rows);
            const auto t2 = std::chrono::steady_clock::now();
            if (memcmp(ra.data(), rb.data(), rows*sizeof(float)) != 0) w.same = false;
            unsigned char mix = 0;
            const auto t3 = std::chrono::steady_clock::now();
            for (size_t ix = 0; ix + tile <= rows; ix += tile) {
                iqk_convert_iq3_s_q8_k_r8(QK_K*nb, x.data() + ix*nb, nb*sizeof(block_iq3_s), qa.data(), tile);
                mix ^= (unsigned char)qa[ix % qa.size()];
            }
            const auto t4 = std::chrono::steady_clock::now();
            for (size_t ix = 0; ix + tile <= rows; ix += tile) {
                iqk_convert_iq3_s_z6_q8_k_r16(QK_K*nb, z.data() + ix*nb, nb*sizeof(block_iq3_s_z6), qb.data(), tile);
                mix ^= (unsigned char)qb[ix % qb.size()];
            }
            const auto t5 = std::chrono::steady_clock::now();
            if (mix != 0 || memcmp(qa.data(), qb.data(), qa.size()) != 0) w.same = false;
            const double d[4] = {std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count(),
                                 std::chrono::duration<double>(t4 - t3).count(), std::chrono::duration<double>(t5 - t4).count()};
            for (int k = 0; k < 4; ++k) if (r == 0 || d[k] < w.t[k]) w.t[k] = d[k];
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < nth; ++i) th.emplace_back(body, i);
    while (ready.load() < nth) std::this_thread::yield();
    go.store(1);
    for (auto & t : th) t.join();
    bool same = true;
    double blocks_s[4] = {};
    for (auto & w : W) {
        same &= w.same;
        for (int k = 0; k < 4; ++k) blocks_s[k] += w.rows*nb/w.t[k];
    }
    const double bytes[4] = {sizeof(block_iq3_s), sizeof(block_iq3_s_z6), sizeof(block_iq3_s), sizeof(block_iq3_s_z6)};
    const char * names[4] = {"kernel 1 column, gather (preset)", "kernel 1 column, table", "conversion Q8, gather (preset)", "conversion Q8, table"};
    printf("%d threads, %zu MiB of IQ3_S per thread (%zu rows of %d weights), minimum of %d rounds; results %s\n",
            nth, mib, rows, QK_K*nb, rounds, same ? "IDENTICAL" : "DIFFERENT");
    for (int k = 0; k < 4; ++k) {
        printf("%-34s %8.2f M blocks/s  %7.2f GB/s read  %6.1f G weights/s  %+6.1f%% weights/s over the gather\n", names[k],
                blocks_s[k]/1e6, blocks_s[k]*bytes[k]/1e9, blocks_s[k]*QK_K/1e9, 100*(blocks_s[k]/blocks_s[k & 2] - 1));
    }
    printf("time ratio table/gather: kernel %.3f, conversion %.3f\n", blocks_s[0]/blocks_s[1], blocks_s[2]/blocks_s[3]);
    return same ? 0 : 1;
}

}   // namespace

int main(int argc, char ** argv) {
    // The reference IQ3_S conversion and the conversion threshold are read once per process: the preset's values
    // unless the caller chose others.
    setenv("GGML_Z6_IQ3S_GATHER", "1", 0);
    setenv("GGML_Z6_IQ3S_CONV_MIN", "8", 0);
    std::string gguf, tensor;
    size_t mib = 64;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--banco" && i + 3 < argc) return run_bench((size_t)atoll(argv[i+1]), atoi(argv[i+2]), atoi(argv[i+3]));
        else if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--tensor" && i + 1 < argc) tensor = argv[++i];
        else if (a == "--mib" && i + 1 < argc) mib = (size_t)atoll(argv[++i]);
        else { fprintf(stderr, "usage: %s [--gguf FILE [--tensor NAME] [--mib N]] | --banco MIB THREADS ROUNDS\n", argv[0]); return 2; }
    }
    if (!z6_iq3s_prepare()) { puts("FAIL: the IQ3_S grid is not invertible, no lossless reordering"); return 1; }
    printf("reference IQ3_S conversion: %s; conversion threshold %s columns; library %s the reordered type\n",
            strcmp(getenv("GGML_Z6_IQ3S_GATHER"), "1") == 0 ? "gather (preset)" : "upstream", getenv("GGML_Z6_IQ3S_CONV_MIN"),
            library_has_type() ? "has" : "DOES NOT HAVE (library checks skipped)");

    Rng rng{1302};
    bool ok = true;
    for (int mode = 0; mode < 3; ++mode) {
        for (int nb : {1, 3, 16}) {
            for (int rows : {1, 2, 5, 16, 33, 64, 256}) {
                std::vector<block_iq3_s> x((size_t)rows*nb);
                fill_blocks(x, rng, mode);
                char where[96];
                snprintf(where, sizeof(where), "generated mode %d, %d rows of %d blocks", mode, rows, nb);
                for (int q8_mode : {0, 2, 3}) ok &= check_all(x.data(), rows, nb, rng, q8_mode, where);
            }
        }
    }
    printf("generated blocks (random, fp16 edges, extremes; rows 1-256, 1/3/16 blocks per row, 1-8 columns): %s\n", ok ? "identical" : "DIFFERENT");
    {   // GGML_Z6_IQ3S_LUT_LIVELLI: list of layers
        struct { const char * list; int il; bool want; } cases[] = {
            {nullptr, 0, true}, {"", 7, true}, {"   ", 7, true},
            {"3-10,13", 3, true}, {"3-10,13", 10, true}, {"3-10,13", 11, false}, {"3-10,13", 13, true}, {"3-10,13", 2, false},
            {" 3 - 10 , 13 ", 5, true}, {"5,5,5", 5, true}, {"10-3", 5, false}, {"40", 40, true}, {"40", -1, false},
            {"3-10,13", -1, false}, {"3,x", 3, true}, {"x,3", 3, false}, {"3-", 3, false}, {"3-,5", 5, false},
            {"0-46", 46, true}, {"0-46", 47, false},
        };
        bool lists = true;
        for (const auto & c : cases) {
            if (ggml_z6_layer_in_list(c.list, c.il) != c.want) {
                fprintf(stderr, "FAIL layer list '%s' layer %d: want %d\n", c.list ? c.list : "(null)", c.il, (int) c.want);
                lists = false;
            }
        }
        printf("layer lists (GGML_Z6_IQ3S_LUT_LIVELLI): %s\n", lists ? "as expected" : "WRONG");
        ok &= lists;
    }
    int rc = ok ? 0 : 1;
    if (!gguf.empty()) {
        const int r = run_gguf(gguf, tensor, mib, rng);
        if (r != 0) rc = r;
    }
    puts(rc == 0 ? "PASS iq3s lut: reordering, inverse, dequantization, kernels 1-8 columns and Q8 conversion are bit-identical"
                 : "FAIL iq3s lut");
    return rc;
}

#else

int main() {
    puts("SKIP iq3s lut: this build has no AVX512 iqk kernels");
    return 77;
}

#endif
