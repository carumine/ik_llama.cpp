#pragma once
// Z6 (GGML_Z6_DSA_POOL_CACHE=1): host bookkeeping for the GLM5NEXT k-pool key cache (kv_self.kp_l).
//
// A pool key is a function of the kr_l rows of its r member cells. The cache is right for pool b when the key was
// pooled from the cells that are its members now, and none of them was written since. `ombra` records, per pool, the
// member cells its cached key was built from (first entry -1: not built). Before each graph the pools to pool again
// are the complete pools some query of the ubatch can see whose record differs from their members; every pool the
// ubatch writes into loses its record first. Nothing else has to remember to invalidate: shifted positions, moved
// cells, a growing n_pool or positions out of order all show up as a record that no longer matches.
//
// Only a rewrite of the kr_l rows behind unchanged cells is invisible here (state restore): that caller resets the
// record through llama_context::qsa_pooled_stale.

#include <algorithm>
#include <cstdint>
#include <vector>

// pos_of(i): position of cache cell i, negative when empty. membri gets [r*n_pool] member cells (0 for an incomplete
// pool, as inp_kpool_cells), piene [n_pool] whether the pool is complete. Mirrors the fill in llama_set_inputs.
template <typename PosOf>
static inline void llama_z6_kpool_membri(PosOf pos_of, int64_t n_kv, int64_t r,
        std::vector<int32_t> & membri, std::vector<uint8_t> & piene) {
    const int64_t n_pool = n_kv / r;
    std::vector<int32_t> filled(n_pool, 0);
    membri.assign(r * n_pool, 0);
    for (int64_t i = 0; i < n_kv; ++i) {
        const int64_t pos = pos_of(i);
        if (pos < 0) continue;
        const int64_t b = pos / r;
        if (b >= n_pool) continue;
        membri[b * r + (pos % r)] = (int32_t) i;
        filled[b]++;
    }
    piene.assign(n_pool, 0);
    for (int64_t b = 0; b < n_pool; ++b) {
        if (filled[b] < (int32_t) r) {
            std::fill(membri.begin() + b * r, membri.begin() + (b + 1) * r, 0);
        } else {
            piene[b] = 1;
        }
    }
}

// Pools the graph of this ubatch has to pool again, in ascending order. ombra holds r entries per pool of the whole
// cache; invalida drops every record first.
static inline void llama_z6_kpool_da_fare(const std::vector<int32_t> & membri, const std::vector<uint8_t> & piene,
        int64_t r, const int32_t * batch_pos, int64_t n_batch, bool invalida,
        std::vector<int32_t> & ombra, std::vector<int32_t> & da_fare) {
    const int64_t n_pool = (int64_t) piene.size();
    da_fare.clear();
    if (invalida) std::fill(ombra.begin(), ombra.end(), -1);
    int64_t q_max = -1;
    for (int64_t j = 0; j < n_batch; ++j) {
        const int64_t pos = batch_pos[j];
        q_max = std::max(q_max, pos);
        if (pos < 0) continue;
        const int64_t b = pos / r;
        // written now: whatever key is cached was pooled from the old rows, even if the cells are the same
        if ((b + 1) * r <= (int64_t) ombra.size()) ombra[b * r] = -1;
    }
    for (int64_t b = 0; b < n_pool && (b + 1) * r <= (int64_t) ombra.size(); ++b) {
        // an incomplete pool is masked by pool_bias, and so is one that ends after the last query
        if (!piene[b] || (b + 1) * r - 1 > q_max) continue;
        if (!std::equal(membri.begin() + b * r, membri.begin() + (b + 1) * r, ombra.begin() + b * r)) {
            da_fare.push_back((int32_t) b);
        }
    }
}

// The window of a graph that pools only what an ubatch of consecutive positions can complete
static inline int64_t llama_z6_kpool_finestra(int64_t n_pool, int64_t n_tokens, int64_t r) {
    return std::min<int64_t>(n_pool, (n_tokens + r - 1) / r + 1);
}

// Fills the window inputs and records what the graph is about to build. A short list repeats its last entry (the
// scatter writes the same value twice); an empty one pools pool 0 again. Returns how many pools did not fit: they
// keep a record that does not match and come back on the next ubatch.
static inline int64_t llama_z6_kpool_riempi(const std::vector<int32_t> & da_fare, const int32_t * pool_cells,
        const std::vector<uint8_t> & piene, int64_t r, int64_t n_win,
        int32_t * win_cells, int32_t * win_pools, std::vector<int32_t> & ombra) {
    for (int64_t w = 0; w < n_win; ++w) {
        const int32_t b = da_fare.empty() ? 0 : da_fare[std::min<size_t>(w, da_fare.size() - 1)];
        win_pools[w] = b;
        for (int64_t m = 0; m < r; ++m) win_cells[w * r + m] = pool_cells[b * r + m];
        if ((b + 1) * r <= (int64_t) ombra.size()) {
            if (b < (int64_t) piene.size() && piene[b]) {
                std::copy(pool_cells + b * r, pool_cells + (b + 1) * r, ombra.begin() + b * r);
            } else {
                ombra[b * r] = -1;   // pooled from cell 0: masked now, to be pooled again when it completes
            }
        }
    }
    return std::max<int64_t>(0, (int64_t) da_fare.size() - n_win);
}
