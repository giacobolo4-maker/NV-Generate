# CLAUDE.md

Istruzioni di progetto per Claude Code. Rispondi all'utente in italiano.

## Scopo del progetto

Fork di lavoro di NVIDIA NV-Generate-CTMR (codice Apache 2.0). Obiettivo dell'utente:
fare fine-tuning del modello di diffusione per generare maschere di segmentazione cerebrali
con tumore, partendo da 30 volumi MRI con tumore (24 train / 6 validation previsti).
Obiettivo secondario, non ancora definito: ControlNet per generare MRI cerebrali da maschera.

Riferimento completo: [RIASSUNTO_PROGETTO.md](RIASSUNTO_PROGETTO.md). Leggilo quando il task riguarda
dati, etichette o piano di lavoro.

## Struttura del repo

- `scripts/`: entry point Python. Eseguili come moduli dalla root (`python -m scripts.<nome>`).
  - `diff_model_train.py`, `train_controlnet.py`: training.
  - `inference.py`, `diff_model_infer.py`, `infer_image_from_mask*.py`, `sample*.py`: inferenza.
  - `download_model_data.py`: scarica i pesi da HuggingFace in `./models/` e `./datasets/`.
  - `quality_check.py`, `compute_fid_2-5d_ct.py`: valutazione.
- `configs/`: JSON di rete, inferenza, training, dizionari etichette. Le coppie
  `config_*` / `environment_*` vanno usate insieme, con la stessa variante di modello.
- `docs/`: documentazione (setup, inference, training, data, evaluation, troubleshooting).
- `skills/`: documentazione Markdown per operazioni specifiche (download modelli, inferenza
  per tipo, preparazione dati). Sono file di riferimento, non skill di Claude Code.
  Leggi il file pertinente prima di inventare un comando.
- `data/`: solo README e un notebook di esempio. **I dati MRI dell'utente non sono nel repo.**
- `*.ipynb`: tutorial della repo. Non leggerli per intero (gli output contengono immagini
  e sono molto grandi). Usa `Grep` per cercare una cella specifica.

## Comandi principali

Installazione (Python 3.11+, GPU con almeno 16 GB VRAM):

```bash
pip install -r requirements.txt
# PyTorch con CUDA: vedi https://pytorch.org/get-started/locally/
```

Download pesi (variante: `rflow-ct`, `ddpm-ct`, `rflow-mr`, `rflow-mr-brain`):

```bash
python -m scripts.download_model_data --version <VARIANT> --root_dir "./" --model_only
```

Inferenza:

```bash
python -m scripts.inference -t ./configs/config_network_rflow.json -i ./configs/config_infer.json -e ./configs/environment_rflow-ct.json --random-seed 0 --version rflow-ct
```

Training ControlNet (esempio dalla documentazione):

```bash
python -m scripts.train_controlnet -t ./configs/config_network_<network>.json -c ./configs/config_maisi_diff_model_<variant>.json -e ./configs/environment_maisi_diff_model_<variant>.json -g 1
```

Lint e formattazione (ruff, linea massima 150, virgolette doppie):

```bash
pre-commit run --all-files
```

Non esiste una suite di test nella repo. Non inventare comandi `pytest`: se servono test,
proponili come nuovi file e chiedi conferma.

## Punti critici da non sbagliare

- **Etichette.** Il modello pre-addestrato usa 132 strutture (`configs/label_dict.json`). I dati
  dell'utente ne hanno circa 183 con numerazione diversa. Le mappature vanno scritte in un file
  dedicato, mai modificando a mano i dizionari in `configs/`. Il tumore cerebrale è la classe
  176 `brain tumor` in `configs/label_dict_ctmr.json`: è una regione unica, non suddivisa.
  Errori qui invalidano tutto il fine-tuning.
- **FOV = `output_size` × `spacing`.** È il parametro che più incide sulla qualità delle maschere.
  Non cambiarlo senza motivo esplicito.
- **Path A vs Path B** nella generazione di maschere: vedi `skills/infer_mask-only.md`.
  Il modello `rflow-mask` è addestrato su corpo CT, non su cervello, e la repo non include
  uno script di training per maschere.
- **Licenze dei pesi diverse.** `rflow-mr-brain`: NVIDIA Open Model License.
  `rflow-mr`: NVIDIA Non-Commercial License, quindi uso limitato. Segnala quando un task
  dipende da `rflow-mr` o da un uso commerciale.
- **Dati sensibili.** Le MRI sono dati clinici. Non copiare file di dati nel repo, non
  committarli, non includerli in output o log condivisi.

## Convenzioni di lavoro

- Non modificare i file in `configs/` o `scripts/` senza spiegare cosa cambia e perché.
  Preferisci nuovi file di configurazione accanto a quelli esistenti.
- Per modifiche al codice, rispetta lo stile già presente e il linter sopra.
- Prima di eseguire training o inferenza su GPU, chiedi conferma: possono richiedere ore
  e molta VRAM.
- Quando un comando fallisce, cerca prima in `docs/troubleshooting.md`.
- Commit: non fare commit senza richiesta esplicita. I messaggi seguono lo stile dei commit
  esistenti (`git log --oneline`).

## Domande ancora aperte (dal riassunto)

- Formato delle maschere dell'utente (NIfTI con etichette intere?) e sequenze MRI disponibili.
- Elenco delle 183 etichette: non ancora fornito.
- Uso di ricerca o commerciale: determina se `rflow-mr` è utilizzabile.
