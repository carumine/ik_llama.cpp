# Esperti caldi fissi: prova Z6 del 28/10/2026

Ramo da `z6-numa-nodo1` (`269916f9`), tutto spento di default.

```bash
GGML_Z6_CALDI_FILE=/percorso/conteggi.txt
GGML_Z6_CALDI_MIB=2048
GGML_Z6_CALDI_DEVICE=0
```

Il budget comprende banchi quantizzati, padding e mappe. Copia dei pesi solo all'avvio. Ramo attivo per `n_tokens <= 32`; prompt più grandi invariati. GPU calda prima della CPU fredda, riduzioni e unione sulla GPU, scheduler sequenziale.

Mappe complementari con ID `-1`. MMVQ azzera gli esclusi; `ggml_mul_multi_add_id` li salta prima di leggere contributi, pesi o scale. Up/gate e down caldi adiacenti e per singolo token: la fusione MMVQ gestisce dieci esperti, mentre il down separato veloce ne accetta al massimo otto. Nessuna modifica a MMQ.

Richiesti esperti quantizzati contigui in RAM, up/gate compatibili con fusione, senza bias/scale, Hadamard o LoRA. Incompatibile con LRU, `--fit`, `-no-fmoe`.

Build `build-cuda-caldi`: Release, sm_86, native CPU, CURL spento; target `llama-server`, `llama-perplexity`, `llama-bench`, `test-z6-caldi`. Test mirato: 78 casi superati su CPU e 78 su CUDA0, anche con NaN esclusi e tipi/geometria reali di Flash-Next.

Banco: `~/Workspace/z6/bench/scripts/esperti-caldi.py`; build: `esperti-caldi-build.sh`. Dati: `~/Workspace/z6/logs/esperti-caldi-20261028/`. Resoconto completo nella sezione «Esperti caldi: implementazione e misure» di `~/Workspace/z6/docs/ottimizzazioni/cpu-gpu-insieme.md`.

RAM 6+1, 20 core del nodo 0, una 3060. Misura minima da 600 token: riferimento 18,663 t/s, nuova build spenta 18,605, caldi 2 GiB 18,727. Testo identico fra riferimento e build spenta. I 2 GiB allocano 2.147.236.480 byte per 684 esperti scelti su A; copertura su B 9,493%.

**Arresto richiesto dopo il gate**: CPU −12,08% per pezzo, 2,962 ms/token risparmiati nello scheduler, contro 3,271 ms/token GPU aggiunti dalle medie degli eventi. Guadagno di velocità nel rumore ±0,2 t/s. Finestre iniziali conservate: anche escludendo la prima, costo GPU e risparmio CPU quasi coincidono.

Non eseguiti: 1 GiB, prompt 7.200, raccolta in dominio, PPL/KLD e due GPU. Il testo con caldi diverge: qualità ancora da validare.

Per 11 GiB su seconda scheda dedicata: `GGML_Z6_CALDI_MIB=11264 GGML_Z6_CALDI_DEVICE=1`, densi su GPU0 con `--split-mode none --main-gpu 0`. Allocazione e trasferimenti da collaudare; gli eventi statici di `GGML_Z6_TEMPI` richiedono adattamento per dispositivo prima di profilare due schede.

## Aggiornamento C1→C2 e percorso parallelo

Il successivo gate autorizzato usa conteggi della sola generazione C1 e tre prompt C2 (`09-test-aggregazione`, `11-spiega-diff`, `12-bash-dryrun`), due ripetizioni da 600 token. RAM ancora 6+1, 20 core. Riferimento 18,728 t/s, build spenta 18,679, caldi 2 GiB sequenziali 18,962: +1,51% sulla build spenta, +1,25% sul riferimento, sopra la soglia dichiarata di +0,2 t/s. Il +6% stimato non è confermato. Dati in `z6/logs/esperti-caldi-20261028/dominio-c1-c2/`.

La variabile aggiuntiva `GGML_Z6_CALDI_PARALLELO=1`, spenta di default, abilita un pacchetto host pinned da 3.607.040 byte per gli ingressi CPU e il down freddo. Lo scheduler copia X, ID e pesi di routing e registra eventi prima di inviare il lavoro caldo; la CPU attende soltanto questi eventi. Una H2D sulla coda GPU precede l'unione; il suo evento protegge il riuso del pacchetto. Descrittori tipizzati, epoche ed eventi persistenti; `needs_sync` non è azzerato per dati estranei. Nessuna registrazione dei pesi CPU e nessuna nuova fusione.

Limitato a `n_copies=1` e scheduling ordinario per layer. Altrimenti, oppure se lo staging fallisce, ritorno sequenziale. Warm-up con tutti i 512 esperti sequenziale; il contesto MTP non duplica il banco. Due schede ancora da collaudare. La strumentazione CUDA ora ha eventi per backend e campione, con pezzi caldi separati dalla densa/unione.

Test aggiuntivo `test-z6-parallelo`: 96 esecuzioni con batch 1/2/8/32, GPU lenta o CPU lenta, nuovi X/ID/pesi e CUDA graphs riutilizzati; partenze CPU durante il lavoro caldo e uscite corrette. Confermati anche i 78 casi CPU e 78 CUDA degli operatori. Build, test, metadati e misure successive in `z6/logs/esperti-caldi-20261028/`; resoconto principale `cpu-gpu-insieme.md` nella repository Z6.

Confronto completo C1→C2 dopo l'implementazione, 3600 token generati per caso: riferimento 18,737 t/s; spento 18,611; caldi 2 GiB sequenziali 18,917; paralleli 20,393. Spento entro ±0,2 t/s; parallelo +9,58% sullo spento, +7,80% sul sequenziale. CPU: 460,87/387,55/400,72 µs per pezzo (spento/seq/par). GPU: 504,77/568,04/294,70 µs per pezzo, ma 50/50/98 pezzi per token; lavoro GPU 25,238/28,402/28,881 ms/token. I 48 pezzi caldi paralleli durano 68,46 µs ciascuno, densa/unione 511,85 µs. Sovrapposizione osservata 169.982/170.000 gruppi; VRAM massima parallelo 9797 MiB totali scheda.

Temperatura zero: sequenziale/parallelo identici per Python e bash; il diff varia anche nel riferimento ripetuto. Questi testi non bastano a dichiarare la qualità: seguono i controlli PPL/KLD nel resoconto Z6. Nota sul profiler: la voce storica dello scheduler «attesa+copia verso CPU» non comprende l'attesa esplicita dei nuovi ticket. Il tempo totale e il lavoro degli operatori restano misurati.

Qualità (4 blocchi da 2048, microbatch 32): KLD 0,0144, stesso token più probabile 97,6%, ΔPPL 0,008 ± 0,008; sequenziale e parallelo identici. Meno della KV `q8_0` già in uso (0,0214). Con microbatch 32 la build è deterministica, quindi il rumore è zero e non va usato come soglia.

MTP a 32K senza memoria bloccata: solo MTP 24,20 t/s, MTP + caldi 1 GiB 26,62 (+10%). Con i preset veri di `scripts/avvia.sh` (memoria bloccata) il limite è la VRAM dei prompt lunghi: a 128K anche 512 MiB vanno in out of memory in una sessione da agente; a 32K con `-ub 4096` 1024 MiB reggono (+5% in sessione, prompt invariati).

Adottato il 29/10 in `flash-32k` con `CALDI=1024`, build `build-cuda-caldi`, tag `z6-caldi-in-uso-20261029`. Tabelle complete in `z6/docs/ottimizzazioni/cpu-gpu-insieme.md`.
