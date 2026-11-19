#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>
//
// common_ngram_mod
// ref: https://github.com/ggml-org/llama.cpp/pull/19164
//

// basic n-gram hasher
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    common_ngram_mod(uint16_t n, size_t size);

    size_t  idx(const entry_t * tokens) const;
    void    add(const entry_t * tokens);
    // Z6: add the windows starting at i0..i1-1 (each reads tokens[i..i+n]), skipping every window that contains
    // `skip` among its n keys or its value. With mtmd the image tokens are LLAMA_TOKEN_NULL (-1 == EMPTY): a window
    // over an image would map eight -1 to -1 (counting `used` up every time) and the windows across the image edge
    // are the same for every image. Without `skip` in the range it is add() for every window, in the same order.
    void    add_range(const entry_t * tokens, size_t i0, size_t i1, entry_t skip);
    // Z6: true if tokens[0..n-1] contains `skip` (no lookup over an image)
    bool    has(const entry_t * tokens, entry_t skip) const;
    entry_t get(const entry_t * tokens) const; // return -1 if not found

    void reset();

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const;
    size_t size_bytes() const;

private:
    size_t n; // ngram size to hash

    size_t used;

    std::vector<entry_t> entries;
};
