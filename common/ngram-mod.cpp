#include "ngram-mod.h"

//
// common_ngram_mod
//

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0) {
    entries.resize(size);

    reset();
}

size_t common_ngram_mod::idx(const entry_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }

    res = res % entries.size();

    return res;
}

void common_ngram_mod::add(const entry_t * tokens) {
    const size_t i = idx(tokens);

    if (entries[i] == EMPTY) {
        used++;
    }

    entries[i] = tokens[n];
}

void common_ngram_mod::add_range(const entry_t * tokens, size_t i0, size_t i1, entry_t skip) {
    if (i0 >= i1) {
        return;
    }
    // last index holding `skip` among the tokens read so far; window i is clean when it is < i
    ptrdiff_t last = -1;
    for (size_t j = i0; j < i0 + n; ++j) {
        if (tokens[j] == skip) {
            last = (ptrdiff_t) j;
        }
    }
    for (size_t i = i0; i < i1; ++i) {
        if (tokens[i + n] == skip) {
            last = (ptrdiff_t) (i + n);
        }
        if (last >= (ptrdiff_t) i) {
            continue;
        }
        add(tokens + i);
    }
}

bool common_ngram_mod::has(const entry_t * tokens, entry_t skip) const {
    for (size_t i = 0; i < n; ++i) {
        if (tokens[i] == skip) {
            return true;
        }
    }
    return false;
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    const size_t i = idx(tokens);

    return entries[i];
}

void common_ngram_mod::reset() {
    std::fill(entries.begin(), entries.end(), EMPTY);
    used = 0;
}

size_t common_ngram_mod::get_n() const {
    return n;
}

size_t common_ngram_mod::get_used() const {
    return used;
}

size_t common_ngram_mod::size() const {
    return entries.size();
}

size_t common_ngram_mod::size_bytes() const {
    return entries.size() * sizeof(entries[0]);
}
