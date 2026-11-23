# Studio approfondito di NV-Generate-CTMR e MAISI-v2

Documento di studio personale, pensato per essere letto con calma, anche da un altro computer.

Fonti usate:
- Paper: *MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss*, Zhao et al., arXiv:2508.05772v2 (16 nov 2025). Letto dal PDF `MAISI-v2.pdf`.
- Blog NVIDIA: *Synthesize Realistic 3D Medical Images at Scale to Ship Pre-Trained Models*, 22 maggio 2026. Letto come riassunto da una pagina web, quindi alcuni dettagli sono parafrasati.
- Codice: lettura statica dei file della repo. **Il codice non è stato eseguito**: non ho GPU né i pesi. Tutto ciò che è marcato come "non verificato" va provato in esecuzione.

Indice:
1. [Il quadro generale](#1-il-quadro-generale)
2. [Concetti di base](#2-concetti-di-base)
3. [Il paper MAISI-v2 in dettaglio](#3-il-paper-maisi-v2-in-dettaglio)
4. [Il blog NVIDIA e la famiglia di modelli](#4-il-blog-nvidia-e-la-famiglia-di-modelli)
5. [Architettura del codice](#5-architettura-del-codice)
6. [Mappa dei file di scripts/](#6-mappa-dei-file-di-scripts)
7. [Configurazioni](#7-configurazioni)
8. [Dizionari delle etichette](#8-dizionari-delle-etichette)
9. [Flussi end-to-end](#9-flussi-end-to-end)
10. [Come usarlo in pratica](#10-come-usarlo-in-pratica)
11. [Bug, rischi e punti da verificare](#11-bug-rischi-e-punti-da-verificare)
12. [Documentazione e codice non coincidono](#12-documentazione-e-codice-non-coincidono)
13. [Cosa significa tutto questo per il tuo progetto](#13-cosa-significa-tutto-questo-per-il-tuo-progetto)
14. [Glossario](#14-glossario)

---

## 1. Il quadro generale

Il progetto è composto da due livelli:

- **MAISI-v2** (paper): un framework per generare volumi 3D di immagini mediche (CT e, nelle versioni successive, MRI) insieme alla loro segmentazione, cioè una mappa in cui ogni voxel è etichettato con la struttura anatomica a cui appartiene.
- **NV-Generate-CTMR** (repo): l'implementazione open source di MAISI-v2 e delle sue varianti, con pesi pre-addestrati, script di inferenza, di training e di preparazione dati.

L'idea centrale è questa: invece di generare solo immagini, il modello genera **coppie immagine + maschera**. Questo è utile perché la maschera è già la "verità" per addestrare altri modelli di segmentazione, e l'immagine è realistica perché condizionata dalla maschera.

Ci sono due modi principali di usare il sistema:

1. **Da zero, con la maschera generata**: il modello di maschere crea una nuova anatomia (Path A), poi il modello di immagini genera il volume corrispondente.
2. **Da una maschera reale**: si pesca una maschera di training, la si modifica con augmentation, poi il modello di immagini genera l'immagine (Path B).

E c'è un terzo uso: **dato una maschera che hai già** (per esempio la tua segmentazione di un tumore), generare l'immagine corrispondente. Questo è il caso `infer_image_from_mask`, probabilmente quello più vicino al tuo lavoro.

---

## 2. Concetti di base

Questa sezione serve a leggere il resto senza fermarsi ai termini.

### 2.1 Modelli di diffusione

Un modello di diffusione impara a generare dati partendo dal rumore. In addestramento si prende un'immagine reale, le si aggiunge rumore in più passi, e la rete impara a **togliere** il rumore. In inferenza si parte da rumore puro e la rete lo trasforma passo dopo passo in un'immagine.

Il problema è che servono centinaia di passi (tipicamente 1000 con DDPM), e in 3D ogni passo è costoso.

### 2.2 DDPM e rectified flow

- **DDPM** (Denoising Diffusion Probabilistic Models): il processo di rumore è stocastico, e in inferenza si fanno tipicamente 1000 passi. Nel codice è la variante `ddpm-ct`.
- **Rectified flow** (RFlow): invece di un processo stocastico, si impara un campo di velocità che trasporta il rumore verso l'immagine lungo una traiettoria il più possibile **dritta**. Traiettorie dritte si possono percorrere con meno passi. Nel paper si usano 30 passi. Nel codice è la variante `rflow-*`.

Formula chiave del paper (eq. 3): si addestra una rete `v_t(x_t, c)` a predire la velocità `x1 − x0`, dove `x_t = (1−t)·x0 + t·x1`. Qui `x0` è rumore, `x1` è l'immagine reale latente, `c` è la condizione (la maschera).

### 2.3 Spazio latente e VAE

Un volume 3D di 512×512×768 voxel è troppo grande per una diffusione diretta. Quindi si usa un **autoencoder variazionale (VAE)**: comprime il volume in uno spazio latente più piccolo e poi lo ricostruisce.

Nel paper: compressione 4×4×4 per asse (64 volte in volume), con 4 canali latenti invece di 1 canale dell'immagine. Il paper lo chiama tasso di compressione 16. La diffusione avviene nello spazio latente (**LDM**, Latent Diffusion Model), poi il decoder riporta tutto all'immagine.

Nel codice l'autoencoder dell'immagine è `AutoencoderKlMaisi` con canali `[64,128,256]`, e produce 4 canali latenti (vedi [sezione 7.1](#71-architettura-di-rete-config_network_rflowjson)).

### 2.4 ControlNet

ControlNet è una rete ausiliaria che legge la condizione (la maschera) e inietta informazioni nel modello di diffusione principale, che resta **congelato**. In questo modo si addestra solo il ControlNet, con meno memoria e meno dati. L'output del ControlNet sono "residui" che si sommano alle feature intermedie della UNet.

### 2.5 Classifier-free guidance (CFG)

Durante il training, una parte delle volte la condizione viene sostituita con un valore "nullo". In inferenza si calcolano due predizioni, con e senza condizione, e si combinano:

`output = uncond + s · (cond − uncond)`

Con `s` alto l'output segue di più la condizione, ma può diventare meno realistico. Nel codice `cfg_guidance_scale` e `mask_generation_cfg_guidance_scale` controllano `s`.

### 2.6 Maschere, etichette, dizionari

Una maschera di segmentazione è un volume dove ogni voxel contiene un intero: 0 per lo sfondo, 1 per il fegato, 23 per il tumore polmonare, e così via. Il dizionario delle etichette associa nome e numero. Il modello è addestrato con una lista fissa di classi, quindi **il numero di una struttura conta**: un'etichetta diversa significa un'altra struttura per il modello.

### 2.7 FOV (campo di vista)

FOV = `output_size × spacing`: il numero di voxel per la spaziatura in mm dà la dimensione fisica del volume. Il paper e la documentazione dicono che il FOV è il parametro che più incide sulla qualità: un FOV fuori distribuzione dà risultati inutilizzabili, anche se il codice accetta l'input.

### 2.8 FID

*Fréchet Inception Distance*: misura quanto la distribuzione delle immagini generate somiglia a quella di immagini reali. Più basso è meglio. Nel paper si calcola in 2D su tre piani (xy, yz, xz) e si fa la media. È una misura di realismo, non di correttezza clinica.

### 2.9 Dice

Misura la sovrapposizione tra due maschere, da 0 a 1. Nel paper è la metrica di segmentazione usata per valutare se i dati sintetici aiutano un modello a segmentare meglio.

---

## 3. Il paper MAISI-v2 in dettaglio

### 3.1 Il problema

Il paper identifica tre limiti dei modelli di sintesi 3D basati su diffusione:

1. **Generalizzabilità**: molti modelli funzionano solo per un organo, una modalità o una spaziatura di voxel. I dati clinici variano molto.
2. **Velocità**: la diffusione richiede molti passi. Generare 512×512×512 può richiedere più di 10 minuti con i metodi precedenti.
3. **Fedeltà alla condizione**: il volume generato non segue sempre la maschera. Per le immagini mediche è un problema critico, perché un tumore fuori posto rende il dato inutile.

MAISI (la versione precedente, Guo et al. 2025, WACV) risolve il primo problema, ma resta lento (1000 passi DDPM) e ha una fedeltà limitata.

### 3.2 Le tre contribuzioni dichiarate

1. **Primo modello rectified flow per la sintesi 3D medica**, integrato in un LDM. Dichiarano 33× di accelerazione mantenendo la qualità.
2. **Perdita contrastiva specifica per regione** (region-specific contrastive loss): per migliorare la sensibilità a strutture piccole, come i tumori.
3. **Codice, pesi, dettagli di training e una demo GUI**, pubblicati.

### 3.3 Architettura

Il sistema è composto da tre parti:

| Componente | Ruolo | Note |
|---|---|---|
| **VAE** | Comprime il volume in latente 4×4×4 per asse, 4 canali | Riusato da MAISI senza nuovo training |
| **LDM con rectified flow** | Genera il latente, condizionato dalla spaziatura dei voxel | Sostituisce il DDPM di MAISI |
| **ControlNet** | Legge la maschera e guida la generazione | Include la perdita contrastiva |

La spaziatura dei voxel è l'unica condizione "di sistema": viene letta automaticamente dall'header NIfTI, senza annotazione manuale. La maschera è la condizione semantica.

### 3.4 La perdita contrastiva specifica per regione

Questa è l'idea più originale del paper. Il problema: con una sola perdita L1 pesata (il tumore pesa 100 volte di più), il tumore non appare sempre nel risultato.

L'idea: generare **due** volumi dallo **stesso rumore**:
- uno con la maschera originale `c_orig`;
- uno con una maschera perturbata `c_perturb`, dove le etichette della regione d'interesse vengono sostituite con l'etichetta dello sfondo corrispondente. Ad esempio il tumore pancreatico diventa pancreas.

Poi si impongono due condizioni:

- **ROI sensitivity** (`L_roi`): nella regione del tumore i due volumi devono **differire**, perché la condizione è cambiata. Si penalizza la differenza assoluta solo dove la maschera di ROI è attiva, con un limite superiore `δ = 2` per evitare esplosioni di gradiente.
- **Background consistency** (`L_bg`): fuori dalla regione i due volumi devono **coincidere**, perché la condizione lì non è cambiata.

La perdita totale è `L_total = L_flow + λ·(L_roi + L_bg)`, con `λ` il peso della perdita contrastiva.

Il paper nota anche un vincolo pratico: se `λ` è troppo grande, il resto del corpo viene generato male e il controllo di qualità sui valori HU fallisce. Il loro schema finale usa `λ = 0.01` per i primi 30 epoch (per far comparire i tumori) e poi `λ = 0.001` per i successivi 30 (per correggere il corpo).

Per la memoria, quando il volume è grande calcolano la perdita sulle feature intermedie dell'encoder del ControlNet invece che sull'output finale. Le feature sono più grossolane, ma l'allenamento su volumi grandi diventa possibile.

### 3.5 La strategia di training in tre stadi

Il training di MAISI-v2 è descritto così:

| Stadio | Dati | Batch | Learning rate | Durata |
|---|---|---|---|---|
| 1. Pre-training | Tutti i volumi a 128³ (anche downsampled) | 96 | 1e-3 | 1 giorno |
| 2. Main | Volumi a risoluzione piena, raggruppati per forma (bucketed) | 128³→96, 256×256×128→24, 512×512×768→1 | 1e-3, polinomiale | 16000 epoch, circa 10 giorni |
| 3. Fine-tuning | Tutti i dati mescolati, pesi di campionamento per bilanciare i dataset | 1 | 1e-4 | 2000 epoch, circa 10 giorni |

Ragioni dichiarate:
- Lo stadio 1 accelera il training e **previene i NaN**, che compaiono con batch piccoli in mixed precision.
- Lo stadio 2 usa "bucketing": ogni GPU riceve solo volumi della stessa forma, per poter usare batch più grandi.
- Lo stadio 3 corregge lo squilibrio di dati introdotto dallo stadio 2.

Il training completo è su 64 A100 da 80 GB per circa tre settimane.

Il **ControlNet** è addestrato separatamente: 8 A100, AdamW, lr 5e-5 con decadimento polinomiale, 60 epoch, circa 2 giorni. Le maschere di training sono **pseudo-etichette** prodotte da VISTA3D (He et al. 2024) su 127 strutture anatomiche e lesioni; dove esistono etichette manuali, le sostituiscono.

### 3.6 I dati

Il training dell'LDM usa circa **107.000 volumi derivati da 11.000 scansioni CT**. Le fonti sono molte e pubbliche (HNSCC, AbdomenCT-1K, AMOS22, MSD, TotalSegmentator, LIDC, NLST e altre) più due dataset interni: 237 CT di lesioni ossee e 1.386 CT di torace e addome.

Scelte di preprocessing:
- Tutte le immagini ridimensionate a dimensioni multiple di 128 (128, 256, 384, 512, 768 per asse).
- Rimosse le CT con meno di 64 slice.
- Volumi ad alta risoluzione ridotti di risoluzione per arricchire la rappresentanza dei casi a bassa risoluzione.

Il paper è esplicito: **tutto il training è su CT**. Le MRI non sono nel paper; sono state aggiunte con le varianti successive.

### 3.7 Risultati

**Velocità (Tabella 1, valori dal paper).** Generazione di 512×512×512 a 1 mm su H100:

| Modello | Passi | Tempo | FID medio |
|---|---|---|---|
| HA-GAN | 1 | 1 s | 13.595 |
| MedSyn | 50+20 | 100 s | 24.709 |
| GenerateCT | 25×201 | 89 s | 10.757 |
| MAISI (DDPM) | 1000 | 198 s + 15 s VAE | **2.441** |
| MAISI (DDIM) | 6 + 15 | 4.776 |
| **MAISI-v2 (rectified flow)** | **30** | **6 s + 15 s VAE** | **2.322** |

La conclusione: MAISI-v2 ha una qualità paragonabile a MAISI-DDPM (anzi leggermente migliore nel FID medio) con circa 33 volte meno tempo di LDM (198 s contro 6 s). La parte VAE (15 s) resta costante.

**Passi di inferenza (Tabella 3).** Il FID medio scende da 20.3 con 5 passi a 4.4 con 10, 2.6 con 20, 2.3 con 30, 2.06 con 50, 1.97 con 100. Scelgono **30 passi** come default perché oltre c'è un rendimento decrescente.

**Confronto con video (Tabella 2).** Stesso ordine di grandezza di Stable Video Diffusion e Open-Sora 2.0 per voxel, con 34 s contro 100 s e 162 s, ma non è un confronto sullo stesso compito.

**Segmentazione (Tabella 4, quattro tumori più ossa).** Il dato di valutazione più importante per il tuo lavoro: si addestra un segmentatore (Auto3DSeg) con dati reali più dati sintetici, e si misura il Dice su un test set reale.

| Tumore | Solo reali | MAISI-v2 (ControlNet + contrastiva) |
|---|---|---|
| Fegato | 0.662 | 0.695 (+3.3%) |
| Polmone | 0.581 | 0.655 (+7.5%) |
| Pancreas | 0.433 | 0.497 (+6.4%) |
| Colon | 0.449 | 0.491 (+4.2%) |
| Ossa | 0.504 | 0.537 (+3.3%) |

Il testo dice che il miglioramento è significativo in **4 casi su 5** (paired t-test). Nella conversione del PDF in testo alcuni simboli di significatività sono andati persi: controlla la tabella originale prima di citare i valori di significatività.

Nota importante: il confronto con la sola ControlNet senza la perdita contrastiva mostra che sul colon la ControlNet semplice **peggiora** (−4.7%), mentre con la perdita contrastiva migliora (+4.2%). È un esempio concreto del valore della loss.

Il FID è misurato su dati AutoPET2023 (fuori distribuzione, OOD) per valutare la generalizzazione. Il paper calcola il FID su metà centrale delle slice, per evitare slice vuote.

**Controllo di qualità sui valori HU.** Hanno definito intervalli fisiologici per la mediana dei valori HU di fegato, milza, pancreas, reni, polmoni, ossa e cervello. Con `λ` scelto bene, il 100% dei volumi generati li supera. Questo controllo è la base del `quality_check.py` del codice.

### 3.8 Limiti dichiarati dagli autori

1. **Solo CT.** MRI e PET sono nel piano futuro.
2. **Solo segmentazione come compito downstream.** Non valutano traduzione d'immagine, inpainting o rilevamento.
3. **Memoria**: 24 GB bastano per 512×512×256; per 512×512×768 servono 40 GB in inferenza e 80 GB in training.

Aggiungo io un limite che il paper non dice esplicitamente: **il FID non misura la correttezza clinica**. Un volume può avere FID basso e un tumore nella posizione sbagliata. Per questo esistono i test di Dice sul tumore e il controllo HU.

### 3.9 Cosa dice il paper sulla fedeltà del tumore

Nella discussione notano che il ControlNet con rectified flow produce **tumori meno allineati** alla maschera rispetto al ControlNet con DDPM, specialmente per il colon. Le ipotesi:
- il flusso deterministico e dritto riduce la diversità della traiettoria, quindi il modello è meno robusto a piccole perturbazioni della condizione;
- un errore di predizione in una regione piccola o a basso contrasto si accumula durante l'integrazione.

La perdita contrastiva è proposta anche come rimedio a questo. È una nota importante: il tuo tumore è piccolo e a basso contrasto, quindi questo problema ti riguarda direttamente.

---

## 4. Il blog NVIDIA e la famiglia di modelli

### 4.1 Cosa dice il blog

Il blog (22 maggio 2026) presenta NV-Generate-CTMR come **framework open source end-to-end** per generare CT e MRI sintetiche, costruito sulla famiglia MAISI. Afferma che è il primo framework open source a supportare in un singolo modello voxel di dimensioni variabili, volumi di dimensioni variabili e copertura del corpo intero.

Sottolinea il principio "riusare invece di riaddestrare": il fine-tuning è più veloce ed efficiente dal punto di vista energetico rispetto al training da zero. È il principio su cui si basa il tuo piano.

Il blog cita anche un'applicazione: immagini-maschera con tumori generate da NV-Generate-CTMR sono state usate come dati di training aumentati per **NV-Segment-CTMR**.

Affermazioni che sono del vendor e vanno prese come tali: "primo open source", "stato dell'arte".

### 4.2 La tabella dei modelli

| Modello | Data rilascio | Regione | Architettura | Passi | Volume max | Uso | Licenza |
|---|---|---|---|---|---|---|---|
| `ddpm-ct` | ago 2024 | Corpo intero | MAISI-v1 | 1000 | 512×512×768 | Solo immagine; coppia immagine/maschera | Open source, commerciale |
| `rflow-ct` | mar 2025 | Corpo intero | MAISI-v2 | 30 | 512×512×768 | Solo immagine; coppia immagine/maschera | Open source, commerciale |
| `rflow-mr` | ott 2025 | Cervello, prostata, addome, mammella | MAISI-v2 | 30 | 512×512×128 | Solo immagine | **Solo ricerca** |
| `NV-Generate-MR-Brain` | mar 2026 | Cervello intero o skull-stripped | MAISI-v2 | 30 | 512×512×256 | Solo immagine; cross-contrast | Open source, commerciale |

Il blog dice che la maggior parte dei modelli è sotto **NVIDIA Open Model License**. C'è una contraddizione con la tabella: `rflow-mr` è "solo ricerca". Per il tuo progetto la licenza conta: se usi `rflow-mr` per un uso commerciale, non sei coperto.

### 4.3 Il modello per il cervello

`NV-Generate-MR-Brain` è quello più rilevante per te:
- basato su MAISI-v2, addestrato su **MR-RATE** (100.000 studi di MRI cerebrale da più di 83.000 pazienti, circa 700.000 volumi, con referti radiologici e metadati DICOM);
- genera cervello intero o skull-stripped, a scelta;
- sequenze: **T1, T2, FLAIR e SWI**;
- risoluzione fino a 512×512×256;
- include un modulo ControlNet per generare strutture anatomiche specificate e per la **traduzione tra sequenze** (predire una sequenza MRI a partire da un'altra).

Il dataset MR-RATE ha licenza CC-BY-NC per gli istituti di ricerca. Le licenze commerciali passano da Forithmus.

**Attenzione, dal codice**: il download per `rflow-mr-brain` scarica solo `autoencoder_v1.pt` (dal repo CT) e `diff_unet_3d_rflow-mr-brain_v1.pt`. **Non scarica un ControlNet per il cervello.** La funzionalità ControlNet dichiarata nel blog, per il cervello, non è quindi raggiungibile con gli script di download attuali. Da verificare.

### 4.4 Il comando di avvio dal blog

```bash
git clone https://github.com/NVIDIA-Medtech/NV-Generate-CTMR.git
cd NV-Generate-CTMR
export MONAI_DATA_DIRECTORY="./temp_work_dir"
network="rflow"
generate_version="rflow-ct"
python -m scripts.inference -t ./configs/config_network_${network}.json -i ./configs/config_infer.json -e ./configs/environment_${generate_version}.json --random-seed 0 --version ${generate_version}
```

Questo comando, letto nel codice, lancia la pipeline completa Path A/Path B per CT. Non è il comando per il cervello.

---

## 5. Architettura del codice

### 5.1 Inventario

| Cartella | Contenuto | Note |
|---|---|---|
| `scripts/` | 21 file Python, circa 9.000 righe | Librerie e punti di ingresso CLI |
| `configs/` | 40 file JSON | Preset di inferenza, ambienti, rete, dizionari |
| `docs/` | 8 file Markdown | Setup, inferenza, training, dati, valutazione, performance, troubleshooting, applicazioni |
| `skills/` | 9 file Markdown | Guide operative per le stesse pipeline |
| Notebook | 5 nella root, 1 in `data/` | Tutorial |
| `.github/workflows/ci.yml` | 1 | Solo pre-commit con ruff. **Nessun test automatico.** |

### 5.2 Come si lancia il codice

Tutto è pensato per essere eseguito come **moduli Python** dalla radice della repo:

```bash
python -m scripts.<nome_file_senza_py>
```

Il file `scripts/__init__.py` rende `scripts` un package, quindi `python -m scripts.inference` funziona. La repo non è installabile come pacchetto (lo dice `pyproject.toml`).

### 5.3 Le dipendenze esterne

Gran parte dell'architettura **non è nella repo**. Le classi di rete (`AutoencoderKlMaisi`, `DiffusionModelUNetMaisi`, `ControlNetMaisi`, `RFlowScheduler`) arrivano da **MONAI** (`monai.apps.generation.maisi.networks`). Per capire le reti, bisogna leggere il codice di MONAI.

Le librerie principali sono: `torch`, `monai`, `numpy`, `scipy`, `scikit-image`, `nibabel`, `huggingface_hub`, `fire`, `tensorboard`, `matplotlib`.

### 5.4 Grafo delle dipendenze

Il punto di partenza è sempre `inference.py` (CLI principale). Da lì:

- `inference.py` → `download_model_data`, `sample.LDMSampler`, `sample.check_input_*`, `utils.define_instance`.
- `sample.py` (orchestratore) → `augmentation`, `find_masks`, `quality_check`, `infer_image_from_mask`, `sample_mask`, `utils`.
- `sample_mask.py` (generazione maschere, Path A) → `utils`, `utils_infer`.
- `infer_image_from_mask.py` (immagine da maschera) → `augmentation`, `utils`, `utils_infer`.
- `utils_infer.py` (cuore della generazione di immagini) → `utils`.
- `diff_model_*.py` (modello di diffusione senza ControlNet) → `download_model_data`, `sample`, `utils`, `utils_infer`.
- `train_controlnet.py` → `augmentation`, `diff_model_setting`, `utils`.
- `diff_model_create_training_data.py` → `transforms`, `utils`.

Il file più "centrale" è `utils.py` (816 righe), che contiene post-processing, dataloader, istanziazione e perdite.

---

## 6. Mappa dei file di scripts/

Per ogni file: scopo, funzioni principali, uso da riga di comando, cosa legge e cosa scrive.

### 6.1 `inference.py` (261 righe): pipeline completa, coppia immagine+maschera

**Scopo.** È la CLI principale per generare coppie immagine+maschera CT. Carica tutte le reti, costruisce un `LDMSampler` e genera N campioni.

**Parametri**:

| Flag | Default | Significato |
|---|---|---|
| `-e/--environment-file` | `./configs/environment.json` (**non esiste**) | Percorsi, dizionari, database maschere |
| `-t/--config-file` | `./configs/config_maisi.json` (**non esiste**) | Definizione rete; si passa `config_network_*.json` |
| `-i/--inference-file` | `./configs/config_infer.json` | Iperparametri di inferenza |
| `-x/--extra-config-file` | nessuno | Override opzionali, per esempio `config_trt.json` |
| `-s/--random-seed` | nessuno | **Non ha `type=int`**: il valore è una stringa (vedi sezione 11) |
| `--version` | `rflow-ct` | Passato al download; non viene validato |
| `--low-vram` | disattivato | Tiene i modelli inattivi su CPU |

**Flusso**:
1. Imposta `PYTORCH_CUDA_ALLOC_CONF` e fissa il seme se richiesto.
2. Cartella dati da `MONAI_DATA_DIRECTORY` o da una cartella temporanea.
3. Se `controllable_anatomy_size` o `controllable_demographics` è non vuoto, va in **Path A** e non scarica il database delle maschere; altrimenti scarica anche quello (Path B).
4. Legge l'ambiente e costruisce i percorsi.
5. Unisce configurazione di rete, inferenza e override.
6. Valida l'input con `check_input_ct` (modalità 1–7) o `check_input_mr` (modalità 8–20).
7. Costruisce i due scheduler di rumore.
8. Hard-code `torch.device("cuda")`: serve la GPU.
9. Carica i checkpoint: autoencoder dell'immagine, UNet dell'immagine, ControlNet (copiato dalla UNet e poi caricato), autoencoder della maschera, UNet della maschera.
10. Crea `LDMSampler` e chiama `sample_multiple_images`.
11. Stampa la memoria GPU massima.

**Legge**: ambiente, rete, inferenza, eventuale extra; tutti i checkpoint; database e zip delle maschere; dizionari; statistiche mediane.
**Scrive**: `sample_<timestamp>_image.nii.gz` e `sample_<timestamp>_label.nii.gz` per ogni campione.

### 6.2 `infer_image_from_mask.py` (527 righe): maschera utente → immagine

**Scopo.** È il file più importante **per il tuo caso d'uso**: prende una maschera NIfTI che hai già (per esempio la tua segmentazione) e genera l'immagine CT/MR corrispondente. È sia libreria sia CLI.

**Funzioni**:
- `ldm_conditional_sample_one_image_from_mask` (riga 65): wrapper che chiama il cuore della generazione in `utils_infer.py`. Imposta il valore di sfondo: −1000 HU per CT, 0 per MR. Se CFG > 0, crea la maschera non condizionata rimuovendo i tumori (`remove_tumors`).
- `crop_img_body_mask` (riga 173): imposta a sfondo i voxel fuori dal corpo.
- `validate_user_mask` (righe 294–379): carica la maschera con MONAI, la porta in orientamento RAS, avvisa se ci sono etichette fuori da 0..132 o se manca l'etichetta 200 (corpo), e ricampiona se la forma o la spaziatura non sono valide.
- `_suggest_valid_target` (riga 251): propone una forma valida mantenendo circa lo stesso FOV.
- `_USER_MASK_FORMAT_WARNING` (righe 214–243): avviso stampato su stderr sul formato della maschera.

**CLI** (righe 391–415):

| Flag | Obbligatorio | Significato |
|---|---|---|
| `--mask` | sì | Maschera NIfTI dell'utente |
| `-t` | sì | `config_network_*.json` |
| `-e` | sì | `environment_*.json` |
| `-i` | sì | `config_infer*.json` |
| `--random-seed` | no, default 0 | Seme |
| `--low-vram` | no | Offload su CPU |

**Flusso**: valida la maschera; carica i modelli (con fallback su CPU alla riga 433); costruisce i tensori di condizione; campiona; salva con MONAI `SaveImage`.

**Scrive**: solo `<stem_maschera>_<timestamp>_image.nii.gz`. **Non salva la maschera** (il commento nella skill dice diversamente).

### 6.3 `infer_image_from_mask_batch.py` (196 righe): batch su manifest JSON

Esegue solo la fase immagine-da-maschera su un file JSON con un elenco di casi. Usa il **fold di validazione** del dataloader, quindi non elabora tutti i casi. Supporta DDP con `-g` maggiore di 1.

Chiavi attese nel manifest: `label`, `dim`, `spacing`; le regioni del corpo solo per `ddpm-ct`.

### 6.4 `sample.py` (718 righe): orchestratore `LDMSampler`

**Scopo.** È il cervello di `inference.py`. Decide se usare Path A o Path B, gestisce il ciclo di tentativi, impone output_size e spacing, fa il controllo di qualità e salva.

**Classe `LDMSampler`**:
- `__init__` (righe 71–225): memorizza le reti, converte `anatomy_list` in interi con `label_dict`, costruisce le etichette per la conditioning e la mappa per il controllo di qualità.
- `sample_multiple_images(num_img)` (righe 227–369): ciclo centrale, descritto in [sezione 9](#9-flussi-end-to-end).
- `select_mask` (righe 371–388): mescola e cicla le maschere, con augmentation sempre attiva.
- `sample_one_pair`, `prepare_anatomy_size_condition`, `prepare_one_mask_and_meta_info`, `sample_one_mask`.
- `ensure_output_size_and_spacing` (righe 537–580): ricampionamento nearest su `Spacing` e `ResizeWithPadOrCrop`; solleva un errore se le etichette anatomiche vengono perse.
- `read_mask_information`, `find_closest_masks` (righe 607–667), `resample_mask_check_organ_list`, `quality_check_ct` (righe 700–718).

**Funzioni esterne ri-esportate**: `check_input_ct`, `check_input_mr`, `ldm_conditional_sample_one_image`, `crop_img_body_mask`, ecc.

### 6.5 `sample_mask.py` (400 righe): generazione maschera da zero (Path A) e validatori

**Scopo.** Genera una maschera 3D da zero con il modello di maschere v2 (AdaGN), condizionato da un vettore di 19 valori. Contiene anche i validatori di input.

**Costanti importanti**:
- `ANATOMY_SIZE_IDX` (righe 57–72): gli slot 0–13 del vettore di condizione. Fegato 0, milza 1, stomaco 2, pancreas 3, colon 4, rene sinistro 5, rene destro 6, polmone 7, cistifellea 8, tumore polmonare 9, tumore pancreatico 10, tumore epatico 11, tumore del colon 12, lesione ossea 13.
- `_DEMOG_MAX` (riga 53): normalizzazione di età, peso, BMI, altezza.
- `N_ANATOMY=14`, `N_DEMOG=5`, `N_COND=19`.

**Funzioni**:
- `ldm_conditional_sample_one_mask` (righe 78–195): genera la maschera. Passi: vettore non condizionato a −1; spaziatura × 100; rumore; RFlow con 100 passi; CFG con `uncond + s·(cond − uncond)`; decodifica a finestra scorrevole; softmax su 125 canali; argmax; rimappatura con `label_dict_124_to_132.json`; scelta dell'etichetta del tumore target; post-processing.
- `filter_mask_with_organs` (righe 198–215): mantiene solo le etichette richieste.
- `check_input_ct` (righe 218–346): regole su dimensioni, spaziatura, FOV ≥ 256 mm, demografia, **al massimo una voce di anatomia** (riga 298), campi di Path B.
- `check_input_mr` (righe 349–400): regole analoghe per MR.

### 6.6 `utils_infer.py` (491 righe): il cuore della generazione di immagini

Questo file contiene la **logica condivisa** per generare immagini con il ControlNet.

- `ReconModel` (righe 67–92): decodifica con `z / scale_factor` tramite l'autoencoder.
- `initialize_noise_latents` (righe 95–115): rumore gaussiano fp16 di forma `[1, C, X, Y, Z]`.
- `run_controlnet_conditioned_image_dm` (righe 118–341): **il loop centrale**. Per ogni passo: ControlNet, UNet con i residui del ControlNet, passo dello scheduler. Poi decodifica e conversione in HU.
- `load_image_models` (righe 344–390): carica autoencoder, UNet, ControlNet (copiato dalla UNet e poi caricato) e scheduler.
- `load_mask_models` (righe 393–434): carica autoencoder e UNet delle maschere.
- `load_paired_inference_models` (righe 437–460): combina i due.
- `build_conditioning_tensors` (righe 463–491): spaziatura × 100, one-hot delle regioni se il modello le usa, tensore della modalità.

### 6.7 `utils.py` (816 righe): post-processing, dataloader, perdite

Il file più grande. Contiene quattro gruppi di funzioni.

**Maschere e etichette**:
- `remap_labels` (righe 35–56): rimappa etichette con coppie `[origine, destinazione]`.
- `binarize_labels` (righe 173–188): converte l'intero in 8 canali binari. **È l'input del ControlNet.** Per questo le etichette possono arrivare a 255.
- `organ_fill_by_closing`, `organ_fill_by_removed_mask`: riempimento di organi.
- `get_body_region_index_from_mask` (righe 389–425): regioni del corpo dalla maschera. Testa: etichette 22 e 120. Torace: 28–32. Addome: 1, 2, 3, 4, 5, 14. Pelvi: 93, 94.
- `add_body_envelope` (righe 428–565): crea l'involucro del corpo dall'aria nell'immagine (soglia −800 HU), rimuove il lettino e riempie con l'etichetta 200. Usato solo in alcuni casi.
- `general_mask_generation_post_process` (righe 568–695): il post-processing più complesso. Include `tumor_organ_dict = {23:28, 24:4, 26:1, 27:62, 128:200}`, regole per l'intestino, riempimenti per chiusura, regole per tumori 23, 26, 27, 129, vasi epatici (25 → fegato 1), vie aeree (132 mantenuto).

**Dataloader e istanziazione**:
- `define_instance` (righe 208–224): istanzia oggetti MONAI da un JSON con `_target_`.
- `prepare_maisi_controlnet_json_dataloader` (righe 261–340): dataloader per il ControlNet. Usa `CacheDataset` con 8 worker.
- `add_data_dir2path` (righe 227–258): aggiunge il prefisso dei dati e divide per fold.

**Perdite e inferenza dinamica**:
- `KL_loss` (righe 763–782): divergenza KL per il VAE.
- `MapLabelValue` (righe 698–760): rimappatura con MONAI.
- `dynamic_infer` (righe 785–816): inferenza con finestra scorrevole.

### 6.8 `find_masks.py` (157 righe): cercare una maschera nel database (Path B)

`find_masks(body_region, anatomy_list, spacing, output_size, check_spacing_and_output_size, database_filepath, mask_foldername)`.

- Converte la regione in indice: head 0, chest/thorax 1, abdomen 2, pelvis 3.
- Estrae lo zip delle maschere se la cartella non esiste.
- Carica il database JSON. Tiene un elemento solo se **tutte** le etichette di anatomia sono presenti e le regioni coprono quelle richieste.
- Se non c'è un tumore nella lista di anatomia, **scarta** gli elementi con etichette tumorali `[23, 24, 26, 27, 128]`.
- Controlla spaziatura e dimensioni esatte, se richiesto.

### 6.9 `quality_check.py` (148 righe): controllo di qualità HU

- `get_masked_data`: estrae i voxel dalla maschera di un organo (`np.isin` se ci sono almeno 3 etichette, altrimenti OR logico).
- `is_outlier` (righe 69–148): confronta la mediana dell'organo con l'intervallo di riferimento (percentile 0.5/99.5 o sigma 6, il più estremo dei due). Per le ossa il limite superiore è 1000 HU.
- Controlla solo sette gruppi: fegato, milza, pancreas, rene, polmone, osso, cervello. **Non controlla i tumori.**

### 6.10 `augmentation.py` (594 righe): augmentation delle maschere reali e rimozione dei tumori

Contiene due blocchi:

**Augmentation (Path B)**. Per ogni tipo di tumore c'è una funzione che deforma la maschera con trasformazioni affini o elastiche e la riposiziona dentro l'organo:
- `augmentation_tumor_bone` (etichetta 128), `augmentation_tumor_liver` (26), `augmentation_tumor_lung` (23), `augmentation_tumor_pancreas` (24), `augmentation_tumor_colon` (27), `augmentation_tumor_only` (BraTS, 401–403).
- `augmentation_body` (righe 337–346): zoom casuale tra 0.99 e 1.01 per le maschere senza tumore.
- `augmentation` (righe 436–481): sceglie la funzione in base alle etichette presenti.

**Rimozione dei tumori (per il ControlNet non condizionato)**:
- `remove_tumors` (righe 484–520): sostituisce le etichette tumorali con l'organo di origine. Mappa `{26:1, 24:4, 27:62, 116:14, 117:5}`. Per i polmoni usa un voto a maggioranza.
- `remap_labels` (righe 580–594): versione locale, **diversa** da `utils.remap_labels`.

**Attenzione**: i cicli di augmentation per fegato, polmone e pancreas sono `while True` senza limite. Se una maschera non ha abbastanza volume dentro l'organo, il programma può non finire. Solo BraTS e colon hanno un limite. Vedi [sezione 11](#11-bug-rischi-e-punti-da-verificare).

### 6.11 `diff_model_setting.py` (139 righe): configurazione condivisa

- `setup_logging`, `load_config(env, model_cfg, model_def)`: unisce i file JSON con `setattr`, quelli successivi sovrascrivono.
- `initialize_distributed`: inizializzazione DDP.
- `run_torchrun`: lancia sottoprocessi con torchrun. **Nessun file Python lo chiama**: codice morto.

### 6.12 `diff_model_train.py` (541 righe): training del modello di diffusione per le immagini

**Scopo.** Addestra la UNet dell'immagine (senza ControlNet) sugli embedding latenti prodotti da `diff_model_create_training_data.py`.

**Punti chiave**:
- `augment_modality_label` (righe 36–66): con probabilità 10% sostituisce la modalità con 0, la classe "nulla" per la CFG. Per CT i codici 2–7 diventano 1; per MR quelli ≥ 9 diventano 8.
- `prepare_data` (righe 85–139): legge spaziatura da un JSON laterale, regione e modalità. `DataLoader(num_workers=6)` è fisso nel codice.
- `calculate_scale_factor` (righe 174–195): `scale_factor = 1/std` del primo batch. Questo valore viene salvato nel checkpoint.
- `train_one_epoch` (righe 226–366): moltiplica le immagini per lo scale factor, sceglie il timestep, calcola il target `immagini − rumore` per RFlow (o la variante DDPM), L1, AMP.
- `save_checkpoint` (righe 369–400): salva `{epoch, loss, num_train_timesteps, scale_factor, unet_state_dict}` sovrascrivendo ogni epoca.
- Ottimizzatore: Adam. Scheduler: `PolynomialLR` con potenza 2.

**Attenzione**: si aspetta un file JSON laterale `<embedding>.nii.gz.json` per ogni elemento, ma nessun script del repo lo crea. Vedi [sezione 11](#11-bug-rischi-e-punti-da-verificare).

### 6.13 `train_controlnet.py` (560 righe): training del ControlNet

**Scopo.** Addestra il ControlNet mantenendo congelata la UNet dell'immagine.

**Punti chiave**:
- `remove_roi` (righe 37–56): rimuove i tumori per la perdita contrastiva.
- `compute_region_contrasive_loss` (righe 59–124): implementa la perdita ROI e background del paper. Nota: nel nome c'è un refuso (`contrasive`), ma è la funzione giusta.
- `compute_model_output` (righe 127–237): binarizza la maschera, aggiunge rumore, passa per ControlNet e UNet.
- `train_controlnet` (righe 240–537):
  - carica la UNet congelata e lo scale factor (righe 292–302);
  - inizializza il ControlNet copiando i pesi della UNet (riga 307), oppure carica `existing_ckpt_filepath`;
  - AdamW con lr 1e-5; `PolynomialLR` con potenza 2;
  - AMP sempre attiva (riga 394);
  - perdita L1, con pesatura sulle etichette `weighted_loss_label` e perdita contrastiva opzionale;
  - salva `{exp_name}_current.pt` ogni epoca e `{exp_name}_best.pt` quando migliora.

**Attenzione**: il loader di validazione viene **scartato** (riga 337: `train_loader, _ = ...`). Non c'è nessuna metrica di validazione.

### 6.14 `diff_model_create_training_data.py` (287 righe): codifica nello spazio latente

**Scopo.** Prende i volumi reali e li codifica con il VAE, producendo un file `<nome>_emb.nii.gz` per ogni volume. È il primo passo prima del training della UNet.

**Punti chiave**:
- `round_number(number, base=128)`: arrotonda le dimensioni a multipli di 128.
- `process_file` (righe 110–195): codifica con `SlidingWindowInferer` (ROI `[320,320,160]`, overlap 0.4, gaussiano), salva la media della posteriore `z_mu`.
- Distribuzione tra le GPU in round-robin.

**Attenzione**: non scrive il file JSON laterale che `diff_model_train.py` richiede (vedi sopra).

### 6.15 `diff_model_infer.py` (400 righe): generazione di immagini senza ControlNet

**Scopo.** Genera immagini con la sola UNet, senza condizione di maschera. Utile per l'immagine non condizionata o per la modalità MR.

- `run_inference` (righe 108–248): rumore, passi, CFG sulla modalità con classe 0 come non condizionata, decodifica con ROI fisso `[80,80,80]`.
- Intensità: MR in `[0, 1000]` se modalità ≥ 8, altrimenti CT in `[−1000, 1000]`. Output `int16`.
- `save_image`: affine diagonale dalla spaziatura.

Il nome del file di output contiene tutti i parametri: `{prefisso}_seed{seed}_size{X}x{Y}x{Z}_spacing{sx}x{sy}x{sz}_{timestamp}_rank{r}_modality{m}.nii.gz`.

**CLI**: `-e`, `-c`, `-t` obbligatori; `-g`, `--low-vram`, `--version` (se presente scarica i pesi).

### 6.16 `download_model_data.py` (206 righe): download dei pesi

Scarica i file da Hugging Face Hub con `hf_hub_download`.

- `rflow-mr-brain`: `models/autoencoder_v1.pt` (repo `nvidia/NV-Generate-CT`), `models/diff_unet_3d_rflow-mr-brain_v1.pt` (repo `nvidia/NV-Generate-MR-Brain`).
- `ddpm-ct` e `rflow-ct`: autoencoder, modello di maschere v2 e relativo autoencoder, UNet dell'immagine, ControlNet. Senza `--model_only` scarica anche il database delle maschere (zip) e il JSON dei candidati e delle dimensioni anatomiche.
- `rflow-mr`: `autoencoder_v2.pt` e `diff_unet_3d_rflow-mr.pt` (repo `nvidia/NV-Generate-MR`).

**Attenzione**: `--root_dir` sposta **solo** i dataset. I modelli finiscono sempre in `./models/` relativo alla cartella da cui lanci il comando. Non ci sono `--token` o `--overwrite`, anche se alcune skill li descrivono.

### 6.17 `adagn_unet.py` (131 righe): la UNet per le maschere (v2)

Definisce `DiffusionModelUNetMaisiAdaGN`, una sottoclasse della UNet MONAI che usa **AdaGN**: invece di sommare il tempo al blocco residuale, genera una coppia (scala, spostamento) e applica `GroupNorm(h)·(1+scala) + spostamento`. Il vettore di condizione di 19 valori passa per un piccolo MLP (19→256→256). Questo è il modo in cui la maschera di Path A è condizionata dalle dimensioni richieste.

### 6.18 `transforms.py` (310 righe): trasformazioni per il VAE e i dati

- `define_fixed_intensity_transform`: per la CT, intensità da `[−1000, 1000]` a `[0, 1]` con clip. Per MR, percentili 0–99.5 senza clip.
- `define_random_intensity_transform`: solo per MR. Campo di bias (p 0.3), artefatti Gibbs (0.3), contrasto (0.3), spostamento dell'istogramma (0.05).
- `define_vae_transform`: flip, rotazioni, zoom, ritaglio casuale a 64³ per il training del VAE.
- `VAE_Transform`: pipeline per CT e MR, scelta in base a `img["class"]`.

### 6.19 `utils_plot.py` (195 righe): visualizzazione

Funzioni per disegnare slice e volumi 3D. **Nessun file Python lo importa**: lo usano solo i notebook.

### 6.20 `compute_fid_2-5d_ct.py` (726 righe): calcolo del FID

Calcola il FID 2.5D su volumi CT, con la CLI `fire`.

- Usa come estrattore di feature **RadImageNet ResNet-50**, scaricato con `torch.hub` (richiede rete).
- Calcola le feature su slice centrali, con padding e ricampionamento opzionali.
- Salva le feature in `.pt`. I valori FID vengono solo scritti nel log.

### 6.21 `scripts/__init__.py` (10 righe)

Solo l'intestazione di licenza. Serve a rendere `scripts` un package.

---

## 7. Configurazioni

### 7.1 Architettura di rete: `config_network_rflow.json`

**Parametri condivisi**:
- `spatial_dims=3`, `image_channels=1`, `latent_channels=4`.
- `include_body_region=false` (true nella versione DDPM).
- `mask_generation_latent_shape=[4,64,64,64]`: la maschera latente di 64³ corrisponde a 256³ in voxel.

**Autoencoder dell'immagine** (`AutoencoderKlMaisi`): canali `[64,128,256]`, due blocchi residuali per livello, nessuna attenzione, `norm_float16=true`, `num_splits=4` (8 per DDPM).

**UNet dell'immagine** (`DiffusionModelUNetMaisi`): canali `[64,128,256,512]`, attenzione ai livelli 3 e 4 (`[F,F,T,T]`), 32 teste di attenzione ai livelli più profondi, `include_spacing_input=true` (la spaziatura entra come condizione), `num_class_embeds=128` (la modalità). Nel DDPM questa chiave è assente.

**ControlNet** (`ControlNetMaisi`): stessi canali. Il condizionamento della maschera ha 8 canali in ingresso (la maschera binarizzata), con canali intermedi `[8, 32, 64]`.

**Autoencoder delle maschere**: 8 canali in ingresso (maschera binarizzata), **125 canali in uscita** (le classi del modello di maschere), canali `[32,64,128]`, `use_checkpointing=true`.

**UNet delle maschere**: la classe `DiffusionModelUNetMaisiAdaGN` del file `adagn_unet.py`, con `cross_attention_dim=19` e `adagn_condition_dim=19`.

**Scheduler**: RFlow con 1000 timestep di training, campionamento uniforme e `scale=1.4` per l'immagine, `scale=1.0` per le maschere.

**Attenzione**: `mask_generation_scale_factor` è scritto nel file ma **non viene mai letto**: il codice usa il valore salvato nel checkpoint.

### 7.2 Inferenza: `config_infer.json` (pipeline completa)

| Chiave | Valore | Note |
|---|---|---|
| `num_output_samples` | 1 | Numero di coppie generate |
| `body_region` | `["chest"]` | Solo Path B |
| `anatomy_list` | `["lung tumor"]` | Solo Path B |
| `controllable_anatomy_size` | `[]` | Se non vuoto si va in Path A |
| `controllable_demographics` | `null` | |
| `num_inference_steps` | 30 | Passi RFlow per l'immagine (il DDPM ne richiede 1000) |
| `mask_generation_num_inference_steps` | 100 | Passi per la maschera |
| `mask_generation_cfg_guidance_scale` | 2.0 | CFG per la maschera |
| `output_size` | `[256,256,256]` | |
| `spacing` | `[1.5,1.5,2.0]` | mm |
| `autoencoder_sliding_window_infer_size` | `[48,48,48]` | |
| `autoencoder_sliding_window_infer_overlap` | 0.6666 | |
| `modality` | 1 | CT |
| `cfg_guidance_scale` | 0.0 | CFG per l'immagine (con tumore) |

**Preset per GPU** (`config_infer_<GB>g_<X>x<Y>x<Z>.json`). Scelti in base alla memoria:

| File | `output_size` | `spacing` | Finestra | Overlap | Split TP | Passi |
|---|---|---|---|---|---|---|
| `16g_256x256x128` | 256,256,128 | 1.5,1.5,4.0 | 96,96,96 | 0.25 | 2 | 30 |
| `16g_256x256x256` | 256,256,256 | 1.5,1.5,2.0 | 48,48,64 | 0.6666 | 4 | 30 |
| `16g_512x512x128` | 512,512,128 | 0.75,0.75,4.0 | 64,64,32 | 0.5 | 2 | 30 |
| `24g_256x256x256` | 256,256,256 | 1.5,1.5,2.0 | 64,64,64 | 0.25 | 4 | **1000** |
| `24g_512x512x128` | 512,512,128 | 0.75,0.75,4.0 | 80,80,32 | 0.4 | 2 | 30 |
| `24g_512x512x512` | 512,512,512 | 0.75,0.75,1.0 | 64,64,48 | 0.4 | 2 | 30 |
| `32g_512x512x512` | 512,512,512 | 0.75,0.75,1.0 | 80,80,48 | 0.4 | 4 | 30 |
| `80g_512x512x512` | 512,512,512 | 0.75,0.75,1.0 | 80,80,80 | 0.4 | 4 | **1000** |
| `80g_512x512x768` | 512,512,768 | 0.75,0.75,0.66667 | 80,80,96 | 0.4 | 4 | 30 |

Due preset RFlow usano 1000 passi mentre gli altri ne usano 30. Sembra un errore di copia, ma non è verificato.

**`config_trt.json`**: sostituisce ControlNet, UNet, decoder dell'autoencoder e modelli delle maschere con versioni compilate con TensorRT. Richiede CUDA.

### 7.3 Diffusione: `config_maisi_diff_model_<variante>.json`

| Variante | `dim` | `spacing` | Passi | Modalità | CFG | Seme | Batch | lr | Epoch |
|---|---|---|---|---|---|---|---|---|---|
| `rflow-ct` | 256,256,128 | 1.7,1.7,2 | 30 | 1 | 0 | 0 | 2 | 1e-5 | 1000 |
| `ddpm-ct` | 256,256,128 | 1.0,1.0,0.75 | 1000 | 1 | 0 | 0 | 1 | 1e-5 | 1000 |
| `rflow-mr` | 128,256,256 | 1.25,1.0,1.0 | 30 | 9 | **15** | 0 | 1 | 1e-5 | 1000 |
| `rflow-mr-brain` | 256,256,128 | 0.94,0.94,1.36 | 30 | 9 | **2** | 1234 | 1 | 1e-5 | 1000 |

Tutte le varianti hanno gli one-hot delle regioni (`[0,1,0,0]` e `[0,0,1,0]`), ma solo `ddpm-ct` li usa.

### 7.4 Training del ControlNet: `config_maisi_controlnet_train*.json`

| File | Training | Inferenza |
|---|---|---|
| generico | batch 1, fold 0, lr 1e-5, 100 epoch, `weighted_loss_label=[129]`, `weighted_loss=100` | 10 passi, finestra `[80,80,80]`, overlap 0.4, modalità 1 |
| `rflow-ct` | come sopra, `weighted_loss_label=[23]`, perdita contrastiva attiva (δ=2, w=0.01) | 30 passi, `[80,80,80]`, 0.4, modalità 1 |
| `ddpm-ct` | `weighted_loss_label=[129]`, senza contrastiva | 1000 passi, `[80,80,80]`, 0.4, modalità 1 |
| `rflow-mr` | `weighted_loss_label=[129]`, senza contrastiva | 30 passi, `[80,80,32]`, 0.4, modalità 9 |

Il file `rflow-mr` esiste, ma la documentazione dice che non c'è un ControlNet per la MR. Il suo ambiente punta a `temp_work_dir/...` (un esempio con tumore cerebrale), quindi sembra uno scaffold, non una configurazione pronta.

### 7.5 Training del VAE: `config_maisi_vae_train.json`

- Dati: `random_aug=true`, zoom casuale, `select_channel=0`.
- Autoencoder: batch 1, patch `[64,64,64]`, validazione `[96,96,64]`, lr 1e-4, peso perceptuale 0.3, peso KL 1e-7, peso avversariale 0.1, perdita L1, validazione ogni 10 epoch, AMP.
- **`n_epochs=1`**: è un valore dimostrativo. La documentazione dice che il modello rilasciato ha 280 epoch.

### 7.6 File di ambiente

Un file `environment_*.json` dice **quali pesi** usare e **dove** trovarli. Ogni coppia di configurazione deve puntare a pesi compatibili.

| File | Contenuto chiave | Note |
|---|---|---|
| `environment_rflow-ct.json` | AE v1, DM rflow-ct, ControlNet rflow-ct, AE e DM delle maschere v2, zip delle maschere, JSON candidati (4000), dimensioni anatomiche, dizionari | Pipeline CT completa |
| `environment_ddpm-ct.json` | Come sopra con 3000 candidati | Un segmento `configs/` in più in una chiave che non viene mai letta |
| `environment_rflow-mr.json` | AE v2, DM rflow-mr, `trained_controlnet_path=null`, percorsi maschere `null` | **Rompe `inference.py:105`** (vedi sezione 11) |
| `environment_maisi_diff_model_rflow-ct.json` | AE v1, `diff_unet_3d_rflow-ct.pt`, `modality_mapping_path` | `json_data_list=./dataset.json`, file assente |
| `environment_maisi_diff_model_rflow-mr-brain.json` | AE v1 (riusa l'autoencoder CT), `diff_unet_3d_rflow-mr-brain_v1.pt` | Coerente con il download |
| `environment_maisi_diff_model_ddpm-ct.json` | `diff_unet_3d_ddpm-ct.pt`, **senza** `modality_mapping_path` | Coerente: il DDPM non ha embedding di classe |
| `environment_maisi_controlnet_train_rflow-ct.json` | AE v1, DM rflow-ct, `existing_ckpt_filepath`, `exp_name=controlnet_kits_finetune`, sottoinsieme KiTS | Esempio di fine-tuning |
| `environment_maisi_controlnet_train_rflow-mr.json` | Esempio cerebrale sotto `temp_work_dir` | Scaffold |
| `environment_maisi_controlnet_train.json` | Chiavi legacy (`trained_controlnet_path`, `autoencoder_epoch273.pt`) | Alcune chiavi ignorate dal trainer |

### 7.7 Mappatura delle modalità: `modality_mapping.json`

Il numero della modalità va alla UNet e al ControlNet come `class_labels`. Valori principali:

- `unknown 0` (**è anche il token "nullo" per la CFG**)
- CT: `ct 1`, `ct_wo_contrast 2`, `ct_contrast 3`
- MR: `mri 8`, `mri_t1 9`, `mri_t2 10`, `mri_flair 11`, `mri_pd 12`, `mri_dwi 13`, `mri_adc 14`, `mri_ssfp 15`, `mri_mra 16`, `mri_t1c 17`, `mri_swi 20`
- MR skull-stripped: `mri_t1 29`, `mri_t2 30`, `mri_flair 31`, `mri_swi 32`, `mri_mra 33`

Regola: **modalità ≤ 7 è CT**, altrimenti MR. I codici 4–7 non hanno nome ma sono trattati come CT. I codici 18–19 e 21–28 non esistono.

Per il tuo progetto: le sequenze T1, T2 e FLAIR corrispondono ai codici 9, 10 e 11 (o 29, 30, 31 se skull-stripped).

### 7.8 Statistiche di intensità: `image_median_statistics_ct.json`

Sette gruppi (fegato, milza, pancreas, rene, polmone, osso, cervello). Per ogni gruppo: `min_median`, `max_median`, percentili 0.5 e 99.5, sigma 6 e 12 bassi e alti. Usati da `quality_check.is_outlier`.

Il file delle dimensioni anatomiche (`all_anatomy_size_conditions.json`) viene **scaricato ma mai letto**. Il significato della scala [0,1] della dimensione non è definito nel repo.

---

## 8. Dizionari delle etichette

### 8.1 `label_dict.json`: vocabolario di 132 classi

È il dizionario che usa **il codice**. Esempi: fegato 1, milza 3, pancreas 4, rene destro 5, aorta 6, stomaco 12, duodeno 13, rene sinistro 14, cervello 22, tumore polmonare 23, tumore pancreatico 24, vasi epatici 25, tumore epatico 26, tumore del colon 27, lobi polmonari 28–32, vertebre 33–56 e S1 127, trachea 57, colon 62, costole 63–86, cranio 120, lesione ossea 128, vie aeree 132, corpo 200.

Gli slot 2, 16, 18, 20, 21, 129, 130, 131 sono segnaposto `dummy1`–`dummy8`, cioè **classi libere** per il fine-tuning. Le skill indicano gli intervalli liberi come 133–199 e 201–255.

**Attenzione**: `lung` non è una chiave. Ci sono solo i lobi. Il nome `lung` è accettato solo come slot di condizione in Path A.

### 8.2 `label_dict_124_to_132.json`: dall'output dell'autoencoder delle maschere alle etichette MAISI

Il modello di maschere produce 125 canali (indici 0–124). Questo file li mappa alle etichette di `label_dict.json`. Esempi: fegato `[1,1]`, milza `[2,3]`, pancreas `[3,4]`, rene destro `[4,5]`, vie aeree `[123,132]`, corpo `[124,200]`.

Perché è importante: se dai al ControlNet un'etichetta tra 0 e 124 senza rimappatura, l'effetto è sbagliato. La skill di immagine da maschera avverte di questo.

### 8.3 `label_dict_124_to_132_ctmr.json`

Identico al precedente, ma il corpo va a **500** invece di 200. Nessun file Python lo carica.

### 8.4 `label_dict_ctmr.json`: il dizionario "alternativo"

Un altro vocabolario, con nomi diversi: rene 2, polmone 20, osso 21, cervello 22, retto 18, prostata o utero (deprecato) 16, corpo 500, vie aeree 132.

**Nessun file Python lo carica.** Questo risolve parzialmente la domanda aperta del tuo `RIASSUNTO_PROGETTO.md`: il codice usa `label_dict.json` (132 classi) per il modello di maschere. Il tumore cerebrale nel codice non è una classe di `label_dict.json`, perché lì il cervello è 22 e il tumore non c'è. Quindi per il tuo progetto la classe da usare va scelta con attenzione (vedi [sezione 13](#13-cosa-significa-tutto-questo-per-il-tuo-progetto)).

### 8.5 Etichette hard-coded nel codice

- Etichette tumorali `[23, 24, 26, 27, 128]`: in `find_masks.py`, `sample_mask.py`, `utils.py`.
- `tumor_organ_dict` in `utils.py`: 23→28, 24→4, 26→1, 27→62, 128→200.
- Vasi epatici 25 → fegato 1; vie aeree 132 mantenute.
- Regioni del corpo (`utils.py:399-407`): vedi [sezione 6.7](#67-utilspy-816-righe-post-processing-dataloader-perdite).

---

## 9. Flussi end-to-end

### 9.1 Path A: maschera da zero (diffusione)

**Trigger**: `controllable_anatomy_size` o `controllable_demographics` non vuoti.

1. **Condizione** (`sample.py:435-480`): vettore di 19 valori. Gli slot 0–13 valgono −1 di default; l'organo o il tumore richiesto riceve la sua dimensione. Gli slot 14–18 sono età, sesso (M=1, F=0), peso, BMI, altezza, divisi per `_DEMOG_MAX` e limitati a [0,1].
2. **UNet delle maschere** (`sample_mask.py:78-195`): rumore di forma `[1,4,64,64,64]`, RFlow con 100 passi, CFG 2.0 contro il vettore tutto a −1.
3. **Decodifica**: a finestra scorrevole con `ReconModel`. Output a 125 canali.
4. **Discretizzazione**: softmax, argmax, rimappatura con `label_dict_124_to_132.json`.
5. **Post-processing** (`utils.py:568-695`): raffinamento del corpo, pulizia dei tumori, soppressione di componenti di organi, regole intestinali, riempimenti per chiusura, tumore target ridisegnato, vasi epatici e vie aeree.
6. **Ripetizione** (`sample.py:282-302`): fino a 5 nuovi tentativi se manca un'etichetta richiesta; poi procede con un avviso.
7. **Ricampionamento** (`sample.py:487-498`, `:537-580`): spaziatura nearest e `ResizeWithPadOrCrop` su `output_size`. Non verifica che le etichette target siano presenti.
8. **One-hot delle regioni** calcolate dalla maschera (`utils.py:389-425`).
9. **Generazione dell'immagine**: vedi [sezione 9.3](#93-immagine-da-maschera-lldm-condizionato-dal-controlnet).
10. **Salvataggio** della maschera completa a 132 classi (il filtro è solo per Path B).

### 9.2 Path B: maschera reale dal database, con augmentation

**Trigger**: `controllable_anatomy_size` e `controllable_demographics` entrambi vuoti.

1. **Ricerca esatta** con `find_masks(..., True)` (`sample.py:249-257`).
2. **Ricerca approssimata** se ci sono meno maschere di `num_img` (`sample.py:258-264`, `find_closest_masks` a `:607-667`). Ordina i candidati per FOV, dimensioni e spaziatura. Scarta quelli che richiederebbero più di 128 voxel di sovracampionamento. Tiene i primi `max(2·num_img, 5)` dopo un controllo degli organi.
3. **Selezione** (`select_mask`, `:371-388`): mescola e cicla. L'augmentation è sempre attiva.
4. **Lettura** (`read_mask_information`, `:582-605`).
5. **Ricampionamento** se si è usata la ricerca approssimata (`:313-314`).
6. **Augmentation** (`augmentation.py:436-481`): in base all'etichetta presente. 128 → osso; 26 → tumore epatico; 23 → tumore polmonare; 24 → tumore pancreatico; 27 → tumore del colon; 401–403 → BraTS. Altrimenti zoom del corpo tra 0.99 e 1.01. Le funzioni per i tumori usano deformazioni elastiche e ripetono finché abbastanza volume rimane nell'organo.
7. **Generazione dell'immagine** (vedi 9.3).
8. **Salvataggio** dell'etichetta filtrata sulla lista di anatomia (`sample.py:356-357`).

### 9.3 Immagine da maschera (LDM condizionato dal ControlNet)

Cuore: `run_controlnet_conditioned_image_dm` (`utils_infer.py:118-341`). Wrapper: `ldm_conditional_sample_one_image_from_mask` (`infer_image_from_mask.py:65-165`).

1. **Condizione**: la maschera intera viene binarizzata in 8 canali (`utils.py:173-188`), quindi l'input del ControlNet è `[B, 8, X, Y, Z]`.
2. **Non condizionata** (solo se CFG > 0): tumori rimossi con `remove_tumors` (`augmentation.py:484-520`), poi binarizzata.
3. **Rumore** di forma `[1, 4, X/4, Y/4, Z/4]` (`utils_infer.py:211`), timestep RFlow (`:213-217`).
4. **Per ogni passo** (`:239-294`):
   - il ControlNet riceve `(x, t, condizione)` più le etichette di classe se il modello le ha; con CFG il batch raddoppia e la metà non condizionata ha classi e maschera azzerate;
   - la UNet riceve la spaziatura × 100, le eventuali regioni e le classi, più i residui del ControlNet sommati ai blocchi di discesa e al centro;
   - CFG: `uncond + s·(cond − uncond)` (`:288-289`);
   - passo dello scheduler (`:291-294`).
5. **Decodifica** a finestra scorrevole con `ReconModel` (`:310-322`).
6. **Intensità**: CT limitata a [0,1] e poi mappata a [−1000, 1000] HU; MR limitata sotto e mappata a [0, 1000] (`:326-335`).
7. **Sfondo**: `crop_img_body_mask` imposta a sfondo i voxel con maschera 0 (`infer_image_from_mask.py:164`, `:173-185`).
8. **Controllo di qualità** (solo CT, Path A/B): viene registrato, ma **non fa rigenerare** con un solo campione (vedi sezione 11).

### 9.4 Training

**(a) VAE dell'immagine**: non c'è `train_vae.py`. Il training è nel notebook `train_vae_tutorial.ipynb` (letto solo con grep).
- Dati: trasformazioni `VAE_Transform`, ritagli casuali da 64³, validazione a finestra `[96,96,64]`.
- Perdita del generatore: L1 + 1e-7·KL + 0.3·perceptuale + 0.1·avversariale (discriminatore PatchDiscriminator).
- Adam per generatore e discriminatore, lr 1e-4.
- Output: `autoencoder.pt`, `discriminator.pt`, e per epoca `autoencoder_epoch{N}.pt`.

**(b) Embedding** (`diff_model_create_training_data.py`): i volumi vengono portati a multipli di 128, codificati con il VAE a tile di 320×320×160 e overlap 0.4, salvati come `<nome>_emb.nii.gz`.

**(c) UNet dell'immagine** (`diff_model_train.py`): input gli embedding e il file JSON laterale. Le immagini vengono moltiplicate per lo scale factor (1/std del primo batch). Con probabilità 10% la modalità diventa 0 (classe nulla per la CFG). Target `immagini − rumore` per RFlow. L1, Adam, `PolynomialLR`, AMP. Un checkpoint per epoca con lo scale factor incluso.

**(d) ControlNet** (`train_controlnet.py`): UNet congelata (`:317-318`). Il ControlNet parte dai pesi della UNet (`:307`) o da `existing_ckpt_filepath`. Perdita L1 sull'output della UNet, con pesi per l'etichetta `weighted_loss_label` e perdita contrastiva opzionale (`:454-481`). AdamW lr 1e-5, `PolynomialLR`. **Nessuna validazione.** Salva `_current.pt` e `_best.pt`.

### 9.5 Generazione di sole immagini

`diff_model_infer.py`: rumore, denoising della UNet con CFG sulla modalità (classe 0 come non condizionata), decodifica dell'autoencoder con ROI fisso `[80,80,80]`, output `int16`. **Senza ControlNet**, quindi senza condizione di maschera.

---

## 10. Come usarlo in pratica

### 10.1 Setup

```bash
pip install -r requirements.txt
# PyTorch con CUDA: https://pytorch.org/get-started/locally/
```

Requisiti: Python 3.11+, GPU con almeno 16 GB di VRAM (per inferenza con preset piccoli). Per 512×512×768 servono 40 GB in inferenza e 80 GB in training, come dice il paper.

### 10.2 Scaricare i pesi

```bash
# Solo pesi, per inferenza
python -m scripts.download_model_data --version rflow-ct --root_dir "./" --model_only
python -m scripts.download_model_data --version rflow-mr --root_dir "./" --model_only
python -m scripts.download_model_data --version rflow-mr-brain --root_dir "./" --model_only
```

I pesi finiscono in `./models/`. Senza `--model_only` scarica anche il database di maschere (la dimensione non l'ho verificata).

### 10.3 Generare una coppia immagine+maschera (CT)

```bash
python -m scripts.inference \
  -t ./configs/config_network_rflow.json \
  -i ./configs/config_infer.json \
  -e ./configs/environment_rflow-ct.json \
  --random-seed 0 --version rflow-ct
```

### 10.4 Generare un'immagine da una tua maschera (il caso più vicino al tuo)

```bash
python -m scripts.infer_image_from_mask \
  --mask percorso/della/tua_maschera.nii.gz \
  -t ./configs/config_network_rflow.json \
  -e ./configs/environment_rflow-ct.json \
  -i ./configs/config_infer.json
```

Questo carica il modello CT. **Per il cervello il modello giusto è quello MR-Brain, e la pipeline di questo comando non è scritta per la MR.** Vedi [sezione 13](#13-cosa-significa-tutto-questo-per-il-tuo-progetto).

### 10.5 Addestrare la UNet o il ControlNet

I comandi nei documenti usano le configurazioni di diffusione con `config_maisi_diff_model_*` per il ControlNet. La documentazione (`docs/training.md:92`) lo indica, ma la skill riconosce che manca `controlnet_train` e che il comando fallirebbe. Per il ControlNet serve il file `config_maisi_controlnet_train_*.json`.

Esempio di training ControlNet (dalla documentazione, da verificare):

```bash
python -m scripts.train_controlnet \
  -t ./configs/config_network_rflow.json \
  -c ./configs/config_maisi_controlnet_train_rflow-ct.json \
  -e ./configs/environment_maisi_controlnet_train_rflow-ct.json \
  -g 1
```

### 10.6 Lint

```bash
pre-commit run --all-files
```

Non ci sono test automatici.

---

## 11. Bug, rischi e punti da verificare

Ordinati per impatto probabile. "Verificato" significa confermato leggendo il codice; "non verificato" significa che serve eseguirlo.

1. **`inference.py:105` fallisce con valori `null` nell'ambiente** (verificato leggendo). `environment_rflow-mr.json` ha tre chiavi a `null`, quindi `scripts.inference` non può girare con quel file. `diff_model_infer` evita il problema controllando `isinstance(v, str)`.

2. **`--random-seed` senza `type=int`** (`inference.py:55-60`, verificato). Il valore è una stringa e viene passato a `set_determinism` e a `LDMSampler`. Non verificato se numpy o torch lo accettano. È il primo controllo da fare se l'avvio crasha.

3. **`np.long` in `utils.py`** (verificato leggendo). Compare in circa dieci righe, tra cui 361, 382, 413, 590, 615–620, 631, 636, 659–660. `np.long` non è un attributo di NumPy moderno. `requirements.txt` permette `numpy>=1.24.0`. Non verificato se lancia errore. Se lo fa: fallirebbero il Path A (riga 590), la ricerca approssimata e `build_conditioning_tensors` (tramite `get_body_region_index_from_mask`, riga 413).

4. **CUDA hard-coded** (verificato). `augmentation.py` alle righe 104, 152, 208, 260, 314 usa `.cuda()`; `sample_mask.py:133` usa `autocast("cuda")`; `inference.py:172` usa `torch.device("cuda")`. Il Path B richiede una GPU in ogni caso.

5. **Loader di validazione scartato nel training del ControlNet** (verificato, `train_controlnet.py:337`). Non c'è nessuna metrica di validazione, e i dati del fold di validazione non vengono mai usati.

6. **File JSON laterale mai prodotto** (verificato nei file letti). `diff_model_train.py:450` si aspetta `<embedding>.nii.gz.json`, ma `diff_model_create_training_data.py` scrive solo il NIfTI (`:190-192`). Non verificato se il notebook del training lo crea.

7. **Il controllo di qualità non fa rigenerare con un campione** (verificato, `sample.py:335`). Con `num_img=1` la coppia viene sempre salvata, anche se fallisce il controllo. La documentazione dice il contrario.

8. **Finestra dei tentativi per le etichette in Path A** (`sample.py:282-302`): cinque tentativi, poi procede con un avviso. Se l'organo richiesto manca, la coppia viene salvata comunque.

9. **Percorsi di default verso file assenti** (`inference.py:31-42`, `diff_model_train.py:521-536`, `train_controlnet.py:541-556`, `diff_model_create_training_data.py:268-283`, `find_masks.py:61-62`). Funzionano solo se si passano i percorsi espliciti, come fa la documentazione.

10. **Chiave con segmento `configs/` in più** (`environment_ddpm-ct.json:10`, verificato). Non viene mai letta.

11. **Incoerenza nei preset**. `24g_256x256x256` e `80g_512x512x512` impostano 1000 passi RFlow; gli altri 30. Intenzione non verificata.

12. **Codice morto e duplicato**:
    - `run_torchrun` (`diff_model_setting.py:98-139`) non è chiamato da nessuno.
    - `utils_plot.py` è usato solo dai notebook.
    - `oran_list` contiene il valore 10 due volte (`utils.py:609`).
    - `label_int_dict["lung"]` duplica 31 (`sample.py:186`).
    - `mask_generation_scale_factor` nella configurazione non è usato (`config_network_rflow.json:139`).
    - `utils.py:686-689` stampa su ogni chiamata.

13. **Commento obsoleto** (`diff_model_create_training_data.py:253-256`): dice che `modality` può non essere definito; invece è definito a riga 245.

14. **Domanda aperta sul dizionario** (`RIASSUNTO_PROGETTO.md:46`). Il codice chiarisce che si usa `label_dict.json`. Vedi [sezione 8](#8-dizionari-delle-etichette).

15. **Heuristic di `add_body_envelope`** (`utils.py:428-565`): soglia −800 HU, `table_frac_thresh=0.05`, device di default `cuda:0`. Documentate solo nel docstring.

16. **Semantica non chiara**: il significato del valore di `controllable_anatomy_size`; il tumore epatico (slot 11) ha un recall basso secondo le skill.

17. **Mappatura dei canali della maschera** (`config_network_rflow.json:95`): 125 canali e indici di rimappatura 0–124 combaciano, ma la corrispondenza completa canale-indice non è stata confermata oltre il file di rimappatura.

---

## 12. Documentazione e codice non coincidono

Questa sezione elenca dove la documentazione dice una cosa e il codice ne fa un'altra. Non è un giudizio: è un elenco di cose da controllare prima di fidarsi.

| Dove | Affermazione | Realtà nel codice |
|---|---|---|
| `docs/inference.md:200`, `:219`; `skills/infer_image-only.md:72`, `:176-177` | CFG della MR image-only è 10 | `rflow-mr` = 15, `rflow-mr-brain` = 2 |
| `skills/infer_image-from-mask.md:66` | La CLI scrive `sample_<ts>_image` e `sample_<ts>_label` | Scrive solo `<stem_maschera>_<ts>_image.nii.gz` (`infer_image_from_mask.py:503-520`) |
| `skills/infer_mask-only_via_real_aug.md:82` | Le augmentation ripetono fino a 1000 volte | Solo il percorso BraTS ha il limite di 1000 (`augmentation.py:403`); fegato, polmone, pancreas, osso sono `while True` (`:149`, `:205`, `:257`); il colon si ferma a 40 (`:309-319`) |
| `skills/infer_mask-only_via_real_aug.md:31-32` | Il fallback ignora il filtro di anatomia | Passa `anatomy_list` a `find_masks` (`sample.py:621-629`) e scarta i candidati dopo il ricampionamento (`sample.py:573-579`, `:685-691`) |
| `docs/inference.md:210` | Il controllo di qualità fallito rigenera, fino a 2 tentativi | Con `num_img=1` il primo tentativo viene sempre salvato (`sample.py:335-341`); `max_try_time` regola solo il numero di candidati |
| `skills/train_controlnet_image-from-mask.md:68` | `num_class_embeds=128` nelle reti rflow/ddpm distribuite | Assente nella rete DDPM (`config_network_ddpm.json`) |
| `docs/training.md:92` | Comando del ControlNet su GPU singola con configurazioni di diffusione | Manca `controlnet_train`; fallirebbe in `train_controlnet.py:260` (la skill lo riconosce a riga 147) |
| `docs/performance.md:9` | Finestra `>=[64,64,32]` per 16g 256×256×128 | Configurazione: `[96,96,96]` |
| `skills/download-models.md` | Flag `--token` e `--overwrite` | Non presenti in `download_model_data.py:191-206` |
| `scripts/sample.py:19-20`, `scripts/infer_image_from_mask.py:40` | Rimandano a `skills/mask-generation.md` e `skills/image-from-mask.md` | I file reali sono `infer_mask-only*.md` e `infer_image-from-mask.md` |

Nota: tra le skill, quelle che descrivono la pipeline CT (`infer_mask-only_*`, `infer_mask-image-paired`, `download-models`) sono largamente corrette. Le discrepanze sono soprattutto su dettagli operativi.

---

## 13. Cosa significa tutto questo per il tuo progetto

Il tuo obiettivo, dal `RIASSUNTO_PROGETTO.md`: fare fine-tuning per generare maschere con tumore cerebrale, partendo dai tuoi 30 casi, e in seguito un ControlNet per generare MRI dalla maschera.

### 13.1 Cosa il codice fa già per il cervello

- **Immagine da maschera per MRI cerebrale**: il modello `rflow-mr-brain` esiste, con UNet e autoencoder scaricabili. Sequenze: T1, T2, FLAIR, SWI.
- **Dimensioni e spaziatura**: `environment_maisi_diff_model_rflow-mr-brain.json` e `config_maisi_diff_model_rflow-mr-brain.json` sono coerenti con il download.
- **Modalità**: MR skull-stripped usa i codici 29–33 (T1 29, T2 30, FLAIR 31), MR completa 9–17 (T1 9, T2 10, FLAIR 11).

### 13.2 Cosa il codice NON fa per il cervello (per ora)

1. **Nessun ControlNet MRI scaricabile** per il cervello. Il download di `rflow-mr-brain` prende solo autoencoder e UNet. Il blog dice che il modulo ControlNet esiste nel modello brain, ma gli script di download non lo prendono. Da verificare su Hugging Face.
2. **Nessun modello di maschere per il cervello.** Il modello di maschere `rflow-mask` è addestrato su corpo CT. Per generare maschere cerebrali con tumore serve addestrarlo o usare un'altra strada.
3. **Nessuna regola di augmentation per il tumore cerebrale nei percorsi CT.** L'augmentation BraTS (`augmentation_tumor_only`) esiste ed è l'unica pensata per il cervello, ma usa etichette 401–403 (sottoregioni BraTS) che non sono nel `label_dict.json` del modello.
4. **Il tumore cerebrale non è una classe del modello.** Nel `label_dict.json` il cervello è 22 e il tumore non c'è. Nel `label_dict_ctmr.json` esiste `brain tumor` 176, ma **nessun codice lo carica**.

### 13.3 La questione delle etichette, aggiornata

Il `RIASSUNTO_PROGETTO.md` decide di mappare il tumore sulla classe 176 `brain tumor` di `label_dict_ctmr.json`. Il codice dice qualcosa di diverso:

- il modello di maschere usa `label_dict_124_to_132.json`, quindi il vocabolario di riferimento è `label_dict.json` a 132 classi;
- `label_dict_ctmr.json` non viene letto da nessun file Python.

Quindi la classe 176 non è utilizzabile così com'è. Serve una scelta esplicita:
- **Opzione A**: usare uno slot libero (133–199) di `label_dict.json` come nuova classe "brain tumor". Questo non basta: la rete delle maschere produce 125 canali, quindi la nuova classe va inserita anche nella mappatura `label_dict_124_to_132.json` e il modello delle maschere va riaddestrato con quella classe.
- **Opzione B**: riusare la classe esistente più vicina (per esempio la lesione ossea 128, che però è semanticamente sbagliata).

Questa è una decisione tua, da documentare: cambia la struttura dei dati e il fine-tuning.

### 13.4 Licenze

- `NV-Generate-MR-Brain` e `rflow-mr-brain`: NVIDIA Open Model License, commerciale.
- `rflow-mr` (corpo, altre regioni): **solo ricerca**.
- Dataset MR-RATE: CC-BY-NC per la ricerca.

Se il progetto è di ricerca, sei coperto. Se puoi avere un uso commerciale, il modello cerebrale è utilizzabile ma il dataset MR-RATE no, e i dati di training del tuo fine-tuning sono i tuoi, quindi vanno valutati separatamente.

### 13.5 Il percorso più realistico

In base a quanto letto, un percorso ragionevole per il tuo obiettivo:

1. **Verificare l'ambiente**: scaricare `rflow-mr-brain` e controllare che i file siano quelli attesi. Verificare se su Hugging Face esiste un ControlNet per il cervello.
2. **Definire le etichette**: decidere la classe per il tumore cerebrale (opzione A o B della sezione 13.3) e scrivere la tabella di mappatura delle 183 etichette verso le classi del modello. Questo è il rischio più alto del progetto.
3. **Preparare i dati**: convertire le maschere in NIfTI con etichette intere, con la spaziatura e le dimensioni richieste. Creare gli embedding con `diff_model_create_training_data.py`, **e creare a mano il file JSON laterale** che `diff_model_train.py` si aspetta.
4. **Testare la pipeline sulla MR** con un caso solo, prima di fare il fine-tuning: l'immagine da maschera deve funzionare su un tuo caso.
5. **Fine-tuning** della UNet per le maschere (o del ControlNet), con i 24 casi di training e i 6 di validazione. Attenzione: `train_controlnet.py` **non valuta** su validazione, quindi la validazione va scritta.
6. **Valutazione**: come dice il tuo piano, Dice sul tumore sui casi di validazione, distanza tra maschere generate e di training per evitare copie, volume medio per etichetta. Il paper usa FID e Dice; il FID non basta per la clinica.

### 13.6 Cosa tenere a mente dal paper per il tuo caso

- Il paper dimostra che la **perdita contrastiva** migliora la presenza dei tumori piccoli e a basso contrasto. Il tuo tumore cerebrale è un caso simile. Il codice la implementa (`train_controlnet.py`, `compute_region_contrasive_loss`), ma va attivata nella configurazione.
- Il paper mostra che i **tumori con rectified flow sono meno allineati** alla maschera rispetto al DDPM. Vale la pena confrontare anche la versione DDPM (`ddpm-ct`) per il tuo caso, anche se non esiste un DDPM per il cervello.
- Il paper addestra su circa 107.000 volumi CT. Con 30 casi cerebrali la situazione è molto diversa, e il rischio principale è la **memorizzazione**: il modello riproduce i tuoi casi invece di generarne di nuovi. Il controllo di distanza dal training serve proprio a questo.

---

## 14. Glossario

- **Autoencoder variazionale (VAE)**: rete che comprime un volume in un latente piccolo e lo ricostruisce.
- **CFG (classifier-free guidance)**: combinazione di predizione condizionata e non condizionata per rafforzare la condizione.
- **ControlNet**: rete ausiliaria che legge la condizione e guida una rete congelata.
- **Dice**: sovrapposizione tra due maschere, da 0 a 1.
- **DDPM**: modello di diffusione con processo di rumore stocastico, tipicamente 1000 passi.
- **FID**: distanza tra distribuzioni di immagini, più bassa è meglio.
- **FOV**: campo di vista fisico = voxel × spaziatura.
- **HU**: unità Hounsfield, la scala delle intensità CT.
- **LDM**: modello di diffusione nello spazio latente.
- **NIfTI**: formato dei volumi medici con header.
- **One-hot**: rappresentazione di una categoria come vettore con un solo 1.
- **Path A**: maschera generata da zero con diffusione.
- **Path B**: maschera reale dal database con augmentation.
- **RFlow (rectified flow)**: modello con traiettorie quasi dritte, meno passi di inferenza.
- **Scale factor**: fattore che porta i latenti vicino alla varianza unitaria, calcolato come 1/std.
- **Spaziatura (spacing)**: dimensione fisica di un voxel, in mm.
- **Sliding window**: elaborazione di un volume grande a blocchi sovrapposti.
- **VISTA3D**: modello di segmentazione usato per creare le pseudo-etichette del ControlNet.
