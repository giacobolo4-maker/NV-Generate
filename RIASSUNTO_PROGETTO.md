# Riassunto del progetto: generazione di maschere e MRI cerebrali

Documento di lavoro per riprendere il progetto da qualsiasi computer.

## 1. Obiettivo

Fare fine-tuning del modello di diffusione per maschere di segmentazione cerebrali, partendo dal modello pre-addestrato della repo, in modo che generi nuove maschere con tumore a partire dai nostri 30 casi.

Obiettivo secondario, ancora da definire: usare un ControlNet per generare una MRI cerebrale a partire dalla maschera.

## 2. Dati disponibili

- 30 volumi MRI con tumore cerebrale.
- Per ogni volume: la MRI e una maschera con circa 183 etichette anatomiche più il tumore.
- Il tumore è una **regione unica** (non suddivisa in necrosi, edema e parte in contrasto).
- Formato delle maschere (NIfTI con etichette intere): **da verificare**.

## 3. Stato della repo

Repo: NV-Generate-CTMR di NVIDIA (licenza del codice Apache 2.0), copiata in questo repo.

Componenti principali:
- Diffusion model per immagini (`scripts/diff_model_train.py`), con pesi pre-addestrati per CT e MRI, incluso `rflow-mr-brain` per il cervello.
- ControlNet (`scripts/train_controlnet.py`), che guida un diffusion model congelato con una maschera.
- `rflow-mask` (NV-Generate-Body): modello per generare maschere, addestrato su corpo CT e non su cervello. Non esiste uno script di training per maschere nella repo.

Licenze dei pesi:
- `rflow-mr-brain`: NVIDIA Open Model License.
- `rflow-mr`: NVIDIA Non-Commercial License.
- `rflow-ct` e `rflow-mask`: NVIDIA Open Model License.

## 4. Problema delle etichette

- Il modello pre-addestrato usa un dizionario fisso di **132 strutture** (`configs/label_dict.json`).
- I nostri dati ne hanno circa **183**, con numerazione diversa.
- Serve una **tabella di corrispondenza**: strutture rinumerate, fuse o aggiunte come nuove classi.

Tumori nei dizionari della repo:
- `label_dict.json`: tumore polmonare (23), pancreatico (24), epatico (26), colon (27), lesione ossea (128). **Nessun tumore cerebrale.**
- `label_dict_ctmr.json`: `brain tumor` (176), più le sottoregioni `brain tumor NCR/NETC` (401), `brain tumor ED` (402), `brain tumor ET` (403).

Decisione presa: il tumore, essendo una regione unica, va mappato sulla classe **176 `brain tumor`**.

## 5. Piano

1. Verificare quale dizionario usa `rflow-mask` (`label_dict.json` o `label_dict_ctmr.json`).
2. Chiarire la differenza tra 132 e 125 classi citata nel README (sezione mask VAE).
3. Costruire la tabella di mappatura delle 183 etichette verso le classi del modello.
4. Decidere come gestire le classi nuove (aggiunte al modello) e quelle assenti nei dati.
5. Convertire le maschere nel formato richiesto (NIfTI, spaziatura, dimensioni).
6. Preparare il fine-tuning: split 24 train / 6 validation, augmentation sulle maschere.
7. Validare: distanza tra maschere generate e maschere di training (per evitare copie), volume medio per etichetta, Dice sul tumore sui casi di validazione.

## 6. Rischi principali

- **30 casi sono pochi**: rischio di memorizzazione e di scarsa qualità del tumore, che il modello base non conosce.
- **Mappatura delle etichette**: errori qui invalidano tutto il fine-tuning.
- **Licenza**: se si usa `rflow-mr` (non commerciale), l'uso è limitato.

## 7. Domande aperte

- Il formato delle maschere è NIfTI con etichette intere? Quali sequenze MRI sono disponibili (T1, T2, FLAIR, altro)?
- Puoi fornire l'elenco delle 183 etichette?
- Il progetto è per uso di ricerca o potenzialmente commerciale?
- Cosa deve generare la rete esattamente: maschere da zero, o maschere condizionate da parametri?

## 8. Prossimi passi concreti

- Controllare `label_dict.json`, `label_dict_ctmr.json` e lo script di inferenza di `rflow-mask`.
- Preparare il file di mappatura con le 183 etichette, con il tumore già assegnato alla 176.
