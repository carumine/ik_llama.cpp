# Z6 NUMA 4+3 e memoria CUDA host

Worktree separato `ik-numa-pinned`, ramo `z6-numa-pinned`, base `z6-numa-nodo1` con merge degli esperti caldi. Il ramo `z6` e le build operative non vengono modificati.

## Attivazione

```bash
GGML_Z6_NUMA_EXPERTS=1 GGML_Z6_NUMA_SPLIT=0.57 GGML_Z6_NUMA_STEAL=0 \
GGML_Z6_NUMA_PINNED=1 GGML_CUDA_HOST_CHUNK_GIB=8 GGML_CUDA_PINNED_CAP_GIB=82 \
numactl --physcpubind=0-39 --localalloc ./build-cuda-numa-pinned/bin/llama-server \
    --no-mmap --defer-ple --cpu-moe --numa numactl --cpu-range 0-39 -t 40 -tb 40 ...
```

Usare lo scope `systemd-run --user --scope -p MemoryMax=88G -p MemorySwapMax=0`, porta 8098 e una GPU libera. Nessun `--membind=0`: 78 GB di esperti non entrano nei 64 GB del nodo 0. Nessun sudo.

Con il pin differito, il caricatore imposta NUMA quando `--defer-ple` crea le copie possedute, poi conferma il posizionamento e registra i pesi soltanto dopo tutte le trasformazioni CPU. Le porzioni dei due nodi sono registrate separatamente con affinità/politica temporanee, mlock limitato alla porzione e verifica globale della residenza dopo il pin. I soli buffer con pagine rilocate sono annullati, riposizionati e registrati di nuovo (massimo otto giri, poi errore). Le politiche del chiamante sono sempre ripristinate.

Su questa macchina `cudaHostRegister` può rilocare alcune pagine: mbind prima del pin senza verifica successiva non basta. `cudaHostRegisterReadOnly` non è supportato dalla 3060/driver in uso. La sonda isolata conferma che mbind su pagine già registrate restituisce EIO.

Le copie H2D/D2H sono suddivise ai confini delle registrazioni adiacenti solo con `GGML_Z6_NUMA_PINNED=1`. Il test mirato verifica byte dell'upload completo, residenza, politiche originali e pulizia delle allocazioni annullate. `GGML_Z6_NUMA_MAPS_PREFIX` salva mappe prima/dopo il pin; il banco aggiunge le mappe del server all'avvio e dopo le richieste. Il tetto host vale anche su Linux e conta le allocazioni pendenti; oltre il tetto c'è fallback CPU con log.

Sono richiesti `--no-mmap`, `GGML_Z6_NUMA_EXPERTS=1`, nessun `--mlock` globale e nessun `GGML_CUDA_NO_PINNED`. Lo scope da 88G e il controllo GPU/porta sono obbligatori nei banchi. Il test deve essere avviato con entrambe le CPU NUMA disponibili, per esempio `numactl --physcpubind=0-39 --localalloc`.

## Esperimenti separati, default spento

- `GGML_Z6_NUMA_AFFINITY=1`: salta il binding NUMA largo se esiste già una lista CPU esplicita.
- `GGML_Z6_NUMA_BARRIER=1`: evita la barriera finale ridondante con l'unione OpenMP, solo senza callback di annullamento.
- `GGML_Z6_NUMA_QUEUE=1`: contatori delle code su linee di cache distinte.
- `GGML_Z6_NUMA_CHUNKS=20`: meno frammenti per esperto, limitati comunque a thread e righe disponibili.

Nessuno di questi esperimenti cambia il formato dei tensori o gli esperti scelti. Le equivalenze numeriche vanno comunque misurate con KLD prima dell'adozione. `GGML_Z6_TEMPI=1` stampa anche la media degli esperti per socket effettivo (non dedotta dal numero del thread).

Banco, build e dati: `~/Workspace/z6/bench/scripts/numa-pinned*`, `~/Workspace/z6/logs/numa-pinned-20261029/`; resoconto in `z6/docs/ottimizzazioni/numa-nodo1.md`, sezione RAM 4+3.
