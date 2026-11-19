// Z6: tests without a model for the ngram drafts with a multimodal projector (IK_Z6_SPEC_MTMD, IK_Z6_MTMD_VELOCE).
//  1. positions: z6_media_pos_next / z6_media_size_up_to_pos against the token walk of server_tokens, on random
//     sequences of text and images (n_pos 1 as for M-RoPE images, and other values), every n and every max_pos;
//  2. ngram-mod: add_range without -1 equals add() on every window; with an image the table equals text A and text B
//     added separately; repeating the add does not grow `used`; has() finds a -1 in the lookup window;
//  3. rollback arithmetic: after a draft of 8 with 3 accepted, positions are those of the text-only sequence moved
//     by -(N-1), the image N tokens and one position.
#include "ngram-mod.h"
#include "../examples/server/z6-media-pos.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <utility>
#include <vector>

static int n_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++n_fail; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

using media_map = std::map<size_t, std::pair<size_t, int64_t>>; // start -> (n_tok, n_pos)

static void info(const std::pair<size_t, int64_t> & c, size_t & n_tok, int64_t & n_pos) { n_tok = c.first; n_pos = c.second; }

// the walk of server_tokens::pos_next(n) with has_mtmd
static int64_t ref_pos_next(const media_map & m, int64_t n_tokens) {
    int64_t idx = 0, pos = 0;
    while (idx < n_tokens) {
        auto it = m.find((size_t) idx);
        if (it != m.end()) { pos += it->second.second; idx += (int64_t) it->second.first; }
        else { pos++; idx++; }
    }
    return pos;
}

// the walk of server_tokens::size_up_to_pos(max_pos) with has_mtmd
static size_t ref_size_up_to_pos(const media_map & m, size_t n_total, int64_t max_pos) {
    size_t idx = 0; int64_t pos = 0;
    if (pos >= max_pos) return idx;
    while (idx < n_total) {
        auto it = m.find(idx);
        if (it != m.end()) { pos += it->second.second; idx += it->second.first; }
        else { pos++; idx++; }
        if (pos >= max_pos) break;
    }
    return idx;
}

static void test_positions() {
    std::mt19937 rng(1234);
    for (int caso = 0; caso < 400; ++caso) {
        media_map m;
        size_t n = 0;
        const int parts = 1 + (int)(rng() % 6);
        for (int p = 0; p < parts; ++p) {
            n += rng() % 30;                                   // text, possibly none (images back to back)
            if (rng() % 3 != 0) {
                const size_t n_tok = 1 + rng() % 40;
                const int64_t n_pos = caso % 4 == 0 ? (int64_t)(rng() % 3) : 1;  // M-RoPE: 1; other values too
                m[n] = { n_tok, n_pos };
                n += n_tok;
            }
        }
        n += rng() % 10;
        for (int64_t k = 0; k <= (int64_t) n; ++k) {
            const int64_t a = z6_media_pos_next(m, k, info), b = ref_pos_next(m, k);
            CHECK(a == b, "pos_next caso %d n %zu k %lld: %lld contro %lld", caso, n, (long long) k, (long long) a, (long long) b);
        }
        const int64_t pmax = ref_pos_next(m, (int64_t) n) + 3;
        for (int64_t mp = -2; mp <= pmax; ++mp) {
            const size_t a = z6_media_size_up_to_pos(m, n, mp, info), b = ref_size_up_to_pos(m, n, mp);
            CHECK(a == b, "size_up_to_pos caso %d n %zu max_pos %lld: %zu contro %zu", caso, n, (long long) mp, a, b);
        }
    }
    // the case of the analysis: 20 tokens, image 40x22 (880 tokens, n_pos 1), 5 tokens
    media_map m; m[20] = { 880, 1 };
    CHECK(z6_media_pos_next(m, 905, info) == 26, "pos_next() del caso 20+880+5");
    CHECK(z6_media_pos_next(m, 900, info) == 21, "pos_next(900)");
    CHECK(z6_media_size_up_to_pos(m, 905, 21, info) == 900, "size_up_to_pos(21)");
    CHECK(z6_media_pos_next(m, 30, info) == 21, "n dentro l'immagine conta l'immagine intera");
    media_map vuota;
    CHECK(z6_media_pos_next(vuota, 17, info) == 17 && z6_media_size_up_to_pos(vuota, 17, 9, info) == 9 &&
          z6_media_size_up_to_pos(vuota, 17, 99, info) == 17 && z6_media_size_up_to_pos(vuota, 17, -1, info) == 0, "mappa vuota");
}

static std::vector<int32_t> rand_text(std::mt19937 & rng, size_t n, int vocab) {
    std::vector<int32_t> t(n);
    for (auto & x : t) x = (int32_t)(rng() % vocab);
    return t;
}

static void test_ngram() {
    const uint16_t N = 8;
    const size_t SIZE = 1 << 16;
    std::mt19937 rng(99);
    // without -1: add_range == add on every window
    for (int caso = 0; caso < 50; ++caso) {
        auto t = rand_text(rng, 50 + rng() % 300, 40);         // small vocabulary: repeated n-grams and overwrites
        common_ngram_mod a(N, SIZE), b(N, SIZE);
        const size_t i0 = rng() % 10, i1 = t.size() - N;
        for (size_t i = i0; i < i1; ++i) a.add(t.data() + i);
        b.add_range(t.data(), i0, i1, -1);
        CHECK(a.get_used() == b.get_used(), "used senza -1, caso %d", caso);
        bool uguale = true;
        for (size_t i = 0; i + N <= t.size(); ++i) uguale &= a.get(t.data() + i) == b.get(t.data() + i);
        CHECK(uguale, "tabella senza -1, caso %d", caso);
        // in two pieces, as draft() adds in chunks
        common_ngram_mod c(N, SIZE);
        const size_t mid = i0 + (i1 - i0) / 2;
        c.add_range(t.data(), i0, mid, -1);
        c.add_range(t.data(), mid, i1, -1);
        bool uguale2 = c.get_used() == a.get_used();
        for (size_t i = 0; i + N <= t.size(); ++i) uguale2 &= a.get(t.data() + i) == c.get(t.data() + i);
        CHECK(uguale2, "a pezzi senza -1, caso %d", caso);
    }
    // [A][image][B] == A and B separately
    for (int caso = 0; caso < 50; ++caso) {
        auto A = rand_text(rng, 9 + rng() % 200, 1000), B = rand_text(rng, 9 + rng() % 200, 1000);
        std::vector<int32_t> t = A;
        t.insert(t.end(), 30 + rng() % 900, -1);
        t.insert(t.end(), B.begin(), B.end());
        common_ngram_mod x(N, SIZE), y(N, SIZE);
        x.add_range(t.data(), 0, t.size() - N, -1);
        y.add_range(A.data(), 0, A.size() - N, -1);
        y.add_range(B.data(), 0, B.size() - N, -1);
        CHECK(x.get_used() == y.get_used(), "used con immagine, caso %d: %zu contro %zu", caso, x.get_used(), y.get_used());
        bool uguale = true;
        for (size_t i = 0; i + N <= A.size(); ++i) uguale &= x.get(A.data() + i) == y.get(A.data() + i);
        for (size_t i = 0; i + N <= B.size(); ++i) uguale &= x.get(B.data() + i) == y.get(B.data() + i);
        CHECK(uguale, "tabella con immagine, caso %d", caso);
        const size_t used = x.get_used();
        x.add_range(t.data(), 0, t.size() - N, -1);
        CHECK(x.get_used() == used, "used cresce ripetendo l'aggiunta, caso %d", caso);
        // the plain add() over the image would count used up and map the -1 window to -1
        common_ngram_mod z(N, SIZE);
        for (size_t i = 0; i < t.size() - N; ++i) z.add(t.data() + i);
        const size_t used_z = z.get_used();
        for (size_t i = 0; i < t.size() - N; ++i) z.add(t.data() + i);
        CHECK(z.get_used() > used_z, "controllo del test: add() su -1 dovrebbe far crescere used");
    }
    // has(): lookup window with -1
    std::vector<int32_t> w = { 1, 2, 3, 4, 5, 6, 7, 8 };
    common_ngram_mod h(N, SIZE);
    CHECK(!h.has(w.data(), -1), "has senza -1");
    w[0] = -1; CHECK(h.has(w.data(), -1), "has con -1 in testa");
    w[0] = 1; w[7] = -1; CHECK(h.has(w.data(), -1), "has con -1 in coda");
}

static void test_rollback() {
    // 20 text tokens, an image of N tokens and 1 position, 5 text tokens; the text-only reference has the same
    // number of tokens, one position each
    const size_t n_img = 880;
    media_map m; m[20] = { n_img, 1 };
    const size_t n_tok = 20 + n_img + 5;
    const int64_t shift = (int64_t)(n_img - 1);
    const int64_t n_past = z6_media_pos_next(m, (int64_t) n_tok, info);
    CHECK(n_past == (int64_t) n_tok - shift && n_past == 26, "n_past dopo l'immagine: %lld", (long long) n_past);
    // per-step restore: accepted_pos = n_past + accepted_step, seq_rm from accepted_pos + 1
    const int accepted_step = 3;
    const int64_t accepted_pos = n_past + accepted_step, accepted_pos_testo = (int64_t) n_tok + accepted_step;
    CHECK(accepted_pos == accepted_pos_testo - shift, "accepted_pos spostata di -(N-1)");
    // the checkpoint of the step is taken at pos_next(n_pre_spec_tokens) with n_pre_spec_tokens = n_tok
    CHECK(z6_media_pos_next(m, (int64_t) n_tok, info) == n_past, "posizione del checkpoint");
    // draft of 8 after the sampled token (9 tokens pushed), then keep_first(n - 8): the cut is after the image
    const size_t keep = n_tok + 9 - 8;
    CHECK(keep > 20 + n_img, "il taglio della bozza cade dopo l'immagine");
    CHECK(z6_media_pos_next(m, (int64_t) keep, info) == n_past + 1, "pos_next dopo keep_first");
    // seq_rm from accepted_pos + 1 leaves every image cell (all at position 20)
    CHECK(accepted_pos + 1 > 20, "seq_rm non tocca le celle dell'immagine");
    // the size of the token vector for a position in the text after the image
    CHECK(z6_media_size_up_to_pos(m, n_tok + 9, accepted_pos + 1, info) == (size_t)(accepted_pos + 1 + shift), "size_up_to_pos dopo il ripristino");
}

int main() {
    test_positions();
    test_ngram();
    test_rollback();
    if (n_fail) {
        fprintf(stderr, "test-z6-bozze-mtmd: %d controlli falliti\n", n_fail);
        return 1;
    }
    printf("test-z6-bozze-mtmd: tutto a posto\n");
    return 0;
}
