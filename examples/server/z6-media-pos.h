#pragma once

// Z6 (IK_Z6_MTMD_VELOCE=1): server_tokens::pos_next(n) and size_up_to_pos() with a multimodal projector loaded,
// computed from the media map (O(number of images)) instead of walking every token with a map lookup (O(n log m)),
// which with drafts runs at every step. Same values as the walk, also when n falls inside an image (the walk then
// counts the whole image) and when the map is empty. `info(chunk, n_tok, n_pos)` returns the chunk sizes.
// Kept free of server headers so that tests/test-z6-media-pos.cpp can check it against the walk without a model.

#include <algorithm>
#include <cstddef>
#include <cstdint>

// position after the first n_tokens tokens
template <typename Map, typename Info>
int64_t z6_media_pos_next(const Map & media, int64_t n_tokens, Info info) {
    int64_t pos = n_tokens;
    for (const auto & it : media) {
        const int64_t start = (int64_t) it.first;
        if (start >= n_tokens) {
            break;
        }
        size_t  n_tok = 0;
        int64_t n_pos = 0;
        info(it.second, n_tok, n_pos);
        const int64_t end = start + (int64_t) n_tok;
        pos += n_pos - (std::min(end, n_tokens) - start);
    }
    return pos;
}

// number of tokens up to (and including the chunk that reaches) position max_pos, at most n_total
template <typename Map, typename Info>
size_t z6_media_size_up_to_pos(const Map & media, size_t n_total, int64_t max_pos, Info info) {
    if (max_pos <= 0) {
        return 0;
    }
    size_t  idx = 0;
    int64_t pos = 0;
    for (const auto & it : media) {
        const size_t start = it.first;
        if (start >= n_total) {
            break;
        }
        const int64_t run = (int64_t) (start - idx);
        if (pos + run >= max_pos) {
            return idx + (size_t) (max_pos - pos);
        }
        pos += run;
        idx  = start;
        size_t  n_tok = 0;
        int64_t n_pos = 0;
        info(it.second, n_tok, n_pos);
        pos += n_pos;
        idx += n_tok;
        if (pos >= max_pos) {
            return idx;
        }
    }
    if (idx >= n_total) {
        return idx;
    }
    const int64_t run = (int64_t) (n_total - idx);
    if (pos + run >= max_pos) {
        return idx + (size_t) (max_pos - pos);
    }
    return n_total;
}
