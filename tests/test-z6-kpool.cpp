// Prova su CPU della logica di llama-z6-kpool.h: una cache KV simulata, la chiave di un pool è una funzione delle
// celle membro e della "versione" (numero di scritture) di ciascuna. Dopo ogni decode ogni pool completo e visibile
// deve avere in cache la chiave che darebbe il ricalcolo completo (il percorso senza cache).
#include "llama-z6-kpool.h"
#include <cstdio>
#include <cstdlib>
#include <random>
struct Sim {
    int64_t size, r, n = 0;
    std::vector<int32_t> pos; std::vector<uint32_t> ver; std::vector<uint64_t> kp;
    std::vector<int32_t> ombra, da_fare; bool stale = false, piena = false;
    int64_t prev_nkv = -1, prev_ntok = -1, prev_win = 0; long pieni = 0, decodes = 0, riusi = 0, max_fuori = 0;
    bool vecchio = false;   // true: logica di ik-claude-iq3s (finestra dai soli token del batch)
    Sim(int64_t s, int64_t r_) : size(s), r(r_), pos(s, -1), ver(s, 0), kp(s / r_, 0), ombra(s / r_ * r_, -1) {}
    uint64_t chiave(const std::vector<int32_t> & m, int64_t b) const {
        uint64_t h = 1469598103934665603ull;
        for (int64_t k = 0; k < r; ++k) { h = (h ^ (uint64_t) m[b*r+k]) * 1099511628211ull; h = (h ^ ver[m[b*r+k]]) * 1099511628211ull; }
        return h;
    }
    // ritorna false se la cache è piena
    bool decode(const std::vector<int32_t> & bp) {
        std::vector<int32_t> cells;
        for (int64_t i = 0; i < size && cells.size() < bp.size(); ++i) if (pos[i] < 0) cells.push_back((int32_t) i);
        if (cells.size() < bp.size()) return false;
        for (size_t j = 0; j < bp.size(); ++j) pos[cells[j]] = bp[j];
        int64_t maxc = 0; for (int64_t i = 0; i < size; ++i) if (pos[i] >= 0) maxc = i + 1;
        n = std::min(size, std::max<int64_t>(256, (maxc + 255) / 256 * 256));
        const int64_t n_pool = n / r, nt = (int64_t) bp.size(), stretta = llama_z6_kpool_finestra(n_pool, nt, r);
        std::vector<int32_t> membri; std::vector<uint8_t> piene;
        llama_z6_kpool_membri([&](int64_t i) { return (int64_t) pos[i]; }, n, r, membri, piene);
        std::vector<int32_t> toccati;
        if (!vecchio) {
            llama_z6_kpool_da_fare(membri, piene, r, bp.data(), nt, stale, ombra, da_fare);
            stale = (int64_t) da_fare.size() > stretta;
        }
        // scelta del grafo, come can_reuse_graph + builder
        const bool riuso = !stale && !(piena && !vecchio) && prev_nkv == n && prev_ntok == nt;
        int64_t n_win;
        if (riuso) { n_win = prev_win; ++riusi; }
        else { n_win = stale ? n_pool : stretta; piena = n_win > stretta; prev_nkv = n; prev_ntok = nt; prev_win = n_win; }
        if (n_win > stretta) ++pieni;
        ++decodes;
        std::vector<int32_t> wc(n_win * r), wp(n_win);
        if (vecchio) {
            if (stale) for (int64_t b = 0; b < n_pool; ++b) toccati.push_back((int32_t) b);
            else for (auto p : bp) { int64_t b = p / r; if (b < 0 || b >= n_pool) continue;
                if (std::find(toccati.begin(), toccati.end(), (int32_t) b) == toccati.end()) toccati.push_back((int32_t) b); }
            if (toccati.empty()) toccati.push_back(0);
            for (int64_t w = 0; w < n_win; ++w) { int32_t b = toccati[std::min<size_t>(w, toccati.size()-1)]; wp[w] = b;
                for (int64_t m = 0; m < r; ++m) wc[w*r+m] = membri[b*r+m]; }
            stale = (int64_t) toccati.size() > n_win;
        } else {
            int64_t fuori = llama_z6_kpool_riempi(da_fare, membri.data(), piene, r, n_win, wc.data(), wp.data(), ombra);
            max_fuori = std::max<long>(max_fuori, fuori);
            stale = false;
        }
        // esecuzione del grafo: scrive le righe kr del batch, poi lo scatter della finestra
        for (auto c : cells) ++ver[c];
        for (int64_t w = 0; w < n_win; ++w) kp[wp[w]] = chiave(wc, w);
        // controllo: per ogni query del batch, ogni pool completo e visibile
        int32_t qmax = -1; for (auto p : bp) qmax = std::max(qmax, p);
        for (int64_t b = 0; b < n_pool; ++b) {
            if (!piene[b] || (b + 1) * r - 1 > qmax) continue;
            if (kp[b] != chiave(membri, b)) return (fprintf(stderr, "    pool %lld non aggiornato (decode %ld, n_kv %lld)\n", (long long) b, decodes, (long long) n), errori++, true);
        }
        return true;
    }
    long errori = 0;
    int32_t maxpos() const { int32_t m = -1; for (auto p : pos) m = std::max(m, p); return m; }
    void taglia(int32_t p0) { for (auto & p : pos) if (p >= p0) p = -1; }
    void sposta(int32_t p0, int32_t delta) { for (auto & p : pos) if (p >= p0) p += delta; if (vecchio) stale = true; else stale = true; }
    void ripristino() { for (int64_t i = 0; i < size; ++i) if (pos[i] >= 0) ++ver[i]; stale = true; }
};
static std::vector<int32_t> fila(int32_t p0, int n) { std::vector<int32_t> v(n); for (int i = 0; i < n; ++i) v[i] = p0 + i; return v; }
int main(int argc, char ** argv) {
    const bool vecchio = argc > 1 && atoi(argv[1]) == 1;
    long tot_err = 0;
    auto scenario = [&](const char * nome, auto f) {
        Sim s(8192, 4); s.vecchio = vecchio; f(s);
        printf("%-52s decode %5ld  grafi a finestra piena %4ld  riusi %5ld  errori %ld%s\n", nome, s.decodes, s.pieni, s.riusi, s.errori,
               s.max_fuori ? "  FUORI FINESTRA" : "");
        tot_err += s.errori + s.max_fuori;
    };
    scenario("1 prompt 1024+1024+37, poi 600 token", [](Sim & s) {
        s.decode(fila(0, 1024)); s.decode(fila(1024, 1024)); s.decode(fila(2048, 37));
        for (int i = 0; i < 600; ++i) s.decode(fila(s.maxpos() + 1, 1)); });
    scenario("2 taglio in coda a meta' pool e rigenerazione", [](Sim & s) {
        s.decode(fila(0, 1024)); for (int i = 0; i < 50; ++i) s.decode(fila(s.maxpos() + 1, 1));
        s.taglia(1030); for (int i = 0; i < 300; ++i) s.decode(fila(s.maxpos() + 1, 1));
        s.taglia(517); s.decode(fila(517, 700)); for (int i = 0; i < 20; ++i) s.decode(fila(s.maxpos() + 1, 1)); });
    scenario("3 ripristino di stato, poi 600 token", [](Sim & s) {
        s.decode(fila(0, 1024)); for (int i = 0; i < 10; ++i) s.decode(fila(s.maxpos() + 1, 1));
        s.ripristino(); for (int i = 0; i < 600; ++i) s.decode(fila(s.maxpos() + 1, 1)); });
    scenario("4 context shift (seq_rm in mezzo + seq_add)", [](Sim & s) {
        s.decode(fila(0, 2048)); for (int i = 0; i < 10; ++i) s.decode(fila(s.maxpos() + 1, 1));
        for (auto & p : s.pos) if (p >= 256 && p < 1027) p = -1;
        s.sposta(1027, -771); for (int i = 0; i < 300; ++i) s.decode(fila(s.maxpos() + 1, 1)); });
    scenario("5 posizioni avanti alle celle (pos/r >= n_pool)", [](Sim & s) {
        s.decode(fila(300, 200)); for (int i = 0; i < 400; ++i) s.decode(fila(s.maxpos() + 1, 1)); });
    scenario("6 batch a posizioni non contigue", [](Sim & s) {
        s.decode(fila(0, 512)); std::vector<int32_t> b, c;
        for (int32_t p = 512; p < 912; ++p) (p % 4 == 3 ? c : b).push_back(p);   // 3 membri su 4, poi il quarto di ognuno
        s.decode(b); s.decode(c); for (int i = 0; i < 100; ++i) s.decode(fila(s.maxpos() + 1, 1)); });
    scenario("7 bozze: 9 token, rifiuto (taglio), ripetuto", [](Sim & s) {
        s.decode(fila(0, 1024)); std::mt19937 g(7);
        for (int i = 0; i < 400; ++i) { int32_t p = s.maxpos() + 1; s.decode(fila(p, 9)); s.taglia(p + 1 + g() % 9); } });
    scenario("8 casuale: append, tagli, shift, ripristini", [](Sim & s) {
        std::mt19937 g(123); s.decode(fila(0, 300));
        for (int i = 0; i < 4000; ++i) {
            int32_t m = s.maxpos(); unsigned k = g() % 100;
            if (m > 6000) { s.taglia(200 + g() % 800); continue; }
            if (k < 80) s.decode(fila(m + 1, 1 + (g() % 10 == 0 ? g() % 64 : 0)));
            else if (k < 90) s.taglia(std::max<int32_t>(1, m - (int32_t)(g() % 40)));
            else if (k < 93) s.ripristino();
            else if (k < 96 && m > 600) { int32_t a = 100 + g() % 200, d = 1 + g() % 150; for (auto & p : s.pos) if (p >= a && p < a + d) p = -1; s.sposta(a + d, -d); }
            else s.decode(fila(m + 1, 1024 > 7000 - m ? 16 : 256));
        } });
    printf("%s: errori totali %ld\n", vecchio ? "logica di ik-claude-iq3s" : "logica corretta", tot_err);
    return tot_err != 0;
}
