# Guida allo studio di NV-Generate-CTMR e MAISI-v2

Questo documento spiega il progetto in modo semplice, ma con tutti i dettagli. Ogni termine tecnico viene spiegato la prima volta che compare, con parole comuni. Per ogni file del codice trovi: a che cosa serve, come si usa, che cosa legge e che cosa scrive.

**Da dove vengono le informazioni**
- Il **paper** MAISI-v2 (arXiv 2508.05772, Zhao et al., NVIDIA, NIH, Oxford, UAMS), letto dal PDF.
- Il **blog** di NVIDIA del 22 maggio 2026, letto come riassunto da una pagina web. Alcuni dettagli sono parafrasati.
- Il **codice**, letto riga per riga da un agente e controllato, ma **non eseguito**: non ci sono GPU né pesi dei modelli nel mio ambiente. Quello che è marcato "da verificare" va provato sul tuo computer.

**Come leggere le sigle**
- **TC**: tomografia computerizzata, in inglese CT. Immagini a raggi X in 3D.
- **MRI** o **risonanza**: immagini magnetiche in 3D.
- **GPU**: la scheda grafica, che fa i calcoli pesanti.
- **File JSON**: file di testo con impostazioni, scritti in un formato ordinato con chiavi e valori.
- **NIfTI** (`.nii.gz`): il formato standard dei volumi medici.

---

## Indice

1. [Di che cosa parla il progetto](#1-di-che-cosa-parla-il-progetto)
2. [Parole chiave](#2-parole-chiave)
3. [Il paper MAISI-v2, spiegato passo per passo](#3-il-paper-maisi-v2-spiegato-passo-per-passo)
4. [Il blog NVIDIA e i modelli disponibili](#4-il-blog-nvidia-e-i-modelli-disponibili)
5. [Come è organizzato il codice](#5-come-è-organizzato-il-codice)
6. [A cosa serve ogni file](#6-a-cosa-serve-ogni-file)
7. [I file di configurazione](#7-i-file-di-configurazione)
8. [Le etichette delle strutture](#8-le-etichette-delle-strutture)
9. [Il percorso di un'immagine, passo per passo](#9-il-percorso-di-unimmagine-passo-per-passo)
10. [Come si usa in pratica](#10-come-si-usa-in-pratica)
11. [Problemi noti](#11-problemi-noti)
12. [Dove la documentazione non corrisponde al codice](#12-dove-la-documentazione-non-corrisponde-al-codice)
13. [Che cosa significa per il tuo progetto](#13-che-cosa-significa-per-il-tuo-progetto)
14. [Glossario](#14-glossario)

---

## 1. Di che cosa parla il progetto

Il progetto crea **immagini mediche finte ma realistiche**. Le crea in 3D, cioè come un blocco di volume e non come una foto piatta.

Ogni immagine viene creata insieme a una **mappa delle strutture** (in inglese *mask* o *label map*). La mappa dice, per ogni piccolo punto del volume, che cosa c'è: fegato, osso, tumore, aria, e così via.

Perché serve? Per addestrare altri programmi che riconoscono le strutture (chiamati modelli di **segmentazione**). Con dati reali servono molte autorizzazioni e molto tempo di lavoro di medici. Con dati finti ma realistici si può aumentare la quantità di dati disponibili.

Ci sono due modi principali di usarlo:

1. **Creare tutto da zero**: il programma inventa una mappa delle strutture e poi crea l'immagine che corrisponde.
2. **Partire da una mappa che hai già**: per esempio la tua segmentazione di un tumore. Il programma crea l'immagine che corrisponde a quella mappa. È il modo più vicino al tuo lavoro.

Un terzo modo è creare solo l'immagine, senza mappa. Ma in questo caso non controlli dove si trovano le strutture.

---

## 2. Parole chiave

Queste sono le parole che compaiono più spesso nel resto del documento.

- **Modello di diffusione**: un programma che impara a creare un'immagine partendo dal rumore, cioè da una confusione casuale. Funziona come una foto che si mette a fuoco passo dopo passo. Ogni passo è un calcolo. Più passi significa più tempo.

- **Rectified flow**: una variante più veloce della diffusione. Invece di fare un percorso tortuoso dal rumore all'immagine, lo fa in linea più dritta. Una linea dritta si percorre con meno passi. Nel paper bastano 30 passi.

- **DDPM**: la variante classica della diffusione, con un processo casuale e tipicamente 1000 passi. È più lenta ma più "varia" nei risultati.

- **Spazio latente**: un modo compresso di rappresentare l'immagine. Invece di lavorare sui milioni di punti del volume, si lavora su una versione più piccola. Poi si decomprime per tornare all'immagine.

- **VAE (autoencoder variazionale)**: la parte che comprime e decomprime. Fa da "traduttore" tra l'immagine e la sua versione compressa. Il nome *encoder* indica la compressione, *decoder* la decompressione.

- **ControlNet**: un componente aggiuntivo che legge la mappa delle strutture e suggerisce al modello principale dove mettere ciascuna struttura. Il modello principale resta fermo, mentre si addestra solo il ControlNet.

- **Maschera (mask)**: la mappa delle strutture. Ogni punto ha un numero intero. 0 è lo sfondo, 1 è il fegato, 23 è il tumore del polmone, e così via.

- **Etichetta (label)**: il numero di una struttura. Il modello conosce solo una lista fissa di etichette. Se cambi il numero di una struttura, il modello la tratta come un'altra struttura.

- **Spaziatura (spacing)**: quanti millimetri misura un punto (un *voxel*) del volume. Per esempio 1.5 × 1.5 × 2.0 mm significa che ogni punto è largo 1.5 mm, alto 1.5 mm e profondo 2 mm.

- **Voxel**: un "pixel 3D", cioè un piccolo cubetto del volume.

- **FOV (campo di vista)**: la dimensione reale del volume, cioè numero di voxel per spaziatura. Se il FOV è fuori dal normale, il risultato peggiora molto, anche se il programma accetta l'input.

- **Classifier-free guidance (CFG)**: un trucco per far seguire di più la condizione al modello. Il modello fa due predizioni, una con la condizione e una senza, e poi amplifica la differenza. Un valore alto segue di più la condizione, ma può rendere l'immagine meno realistica.

- **FID**: un punteggio che misura quanto le immagini sembrano reali, confrontando le distribuzioni statistiche. Più basso è meglio. Non dice nulla sulla correttezza medica.

- **Dice**: un punteggio da 0 a 1 che misura quanto due maschere coincidono. 1 significa coincidenza perfetta. Più alto è meglio.

- **Hounsfield (HU)**: la scala dei valori della TC. L'aria è circa −1000 HU, l'acqua 0, l'osso molto più alto.

- **Modello di segmentazione**: un programma che, dato un'immagine, disegna da solo le strutture (per esempio il tumore). È quello che vuoi addestrare con i dati finti.

---

## 3. Il paper MAISI-v2, spiegato passo per passo

### 3.1 Il problema che risolve

Il paper parte da tre limiti dei modelli di sintesi 3D basati sulla diffusione:

1. **Generalizzabilità**: molti modelli funzionano solo per un organo, una modalità o una spaziatura precisa. I dati reali invece variano molto: scanner diversi, protocolli diversi, risoluzioni diverse.
2. **Velocità**: la diffusione richiede molti passi. In 3D, creare un volume 512×512×512 con i metodi precedenti può richiedere più di dieci minuti.
3. **Fedeltà alla condizione**: il volume generato non sempre segue la maschera. Per le immagini mediche è un problema critico, perché un tumore nel posto sbagliato rende il dato inutile.

Il predecessore, **MAISI** (Guo et al., 2025), risolve il primo problema, ma resta lento e ha poca fedeltà alla maschera.

### 3.2 Le tre idee principali

1. **Primo uso del rectified flow in 3D medico**. Il rectified flow sostituisce la diffusione stocastica con un percorso quasi dritto, che richiede meno passi. Gli autori dichiarano una **accelerazione di 33 volte**.

2. **Perdita contrastiva specifica per regione** (region-specific contrastive loss). Una "perdita" (loss) è il numero che il programma cerca di far diminuire durante l'addestramento. Questa è una perdita pensata per i tumori piccoli. L'idea è spiegata in [sezione 3.4](#34-la-perdita-contrastiva-per-regione-spiegata-semplice).

3. **Codice, pesi e dettagli di addestramento pubblicati**, insieme a una demo grafica.

### 3.3 Come funziona il modello, pezzo per pezzo

Il sistema ha tre parti, che lavorano in cascata:

| Parte | Che cosa fa, in parole semplici | Note |
|---|---|---|
| **VAE** | Comprime il volume 3D in una versione piccola (spazio latente) e poi lo ricostruisce | Gli autori riusano il VAE di MAISI senza riaddestrarlo. Comprime 4×4×4 per asse con 4 canali |
| **LDM con rectified flow** | Crea la versione compressa partendo dal rumore, guidato dalla spaziatura dei voxel | Sostituisce il DDPM di MAISI |
| **ControlNet** | Legge la maschera e guida il modello principale | Qui si applica la perdita contrastiva |

La **spaziatura dei voxel** è una condizione "di sistema": il programma la legge automaticamente dall'intestazione del file NIfTI, senza che nessuno la scriva a mano. La **maschera** è la condizione semantica: dice che cosa va messo dove.

Il paper scrive la formula del rectified flow. In parole semplici: il modello impara una "velocità" che porta un punto dal rumore (`x0`) all'immagine reale (`x1`) lungo una linea dritta. Durante l'addestramento si confronta la velocità predetta con quella della linea dritta, e si corregge l'errore.

### 3.4 La perdita contrastiva per regione, spiegata semplice

**Il problema.** Anche con una penalità più alta sul tumore (il paper usa peso 100), il tumore a volte non compare nel risultato.

**L'idea.** Il programma crea **due** immagini a partire dallo **stesso rumore**:
- una con la maschera originale (con il tumore);
- una con una maschera "truccata", dove il tumore è sostituito dal tessuto normale. Per esempio, un tumore del pancreas diventa pancreas.

Poi il programma impone due regole:
- **Dentro la regione del tumore** (sensibilità alla regione, `L_roi`): le due immagini devono essere **diverse**, perché la condizione è cambiata. Si penalizza la somiglianza solo dove c'è il tumore.
- **Fuori dalla regione** (coerenza di sfondo, `L_bg`): le due immagini devono essere **uguali**, perché fuori la condizione non è cambiata.

Il termine di ogni pezzo è misurato come differenza assoluta tra le due immagini, con un limite superiore (δ = 2) per evitare che il valore esploda. Senza questo limite, nel paper, l'addestramento diventa instabile e compaiono valori NaN (numeri non validi).

La perdita totale è la somma della perdita di flusso più il termine contrastivo moltiplicato per un peso `λ`.

**Il problema del peso.** Se `λ` è troppo grande, il corpo intorno viene generato male, e il controllo di qualità sui valori HU non passa. Il paper usa:
- `λ = 0.01` per i primi 30 epoch (per far comparire i tumori);
- `λ = 0.001` per i 30 successivi (per correggere il resto del corpo).

Un **epoch** è un giro completo sui dati di addestramento.

Per le immagini molto grandi, la perdita viene calcolata su feature intermedie dell'encoder del ControlNet, invece che sull'output finale. Le feature sono più grossolane, ma così la memoria basta.

### 3.5 L'addestramento in tre fasi

Il modello non viene addestrato in un colpo solo. Il paper descrive tre fasi:

| Fase | Quali dati | Dimensione del gruppo (batch) | Learning rate | Durata |
|---|---|---|---|---|
| 1. Pre-training | Tutti i volumi piccoli, 128×128×128 (anche ridotti) | 96 | 1e-3 | 1 giorno |
| 2. Main | Volumi a risoluzione piena, raggruppati per forma | 128³: 96; 256×256×128: 24; 512×512×768: 1 | 1e-3, con decadimento | 16000 epoch, circa 10 giorni |
| 3. Fine-tuning | Tutti i dati mescolati, con pesi per bilanciare i dataset | 1 | 1e-4 | 2000 epoch, circa 10 giorni |

Il **learning rate** è la velocità con cui il modello cambia i suoi valori a ogni correzione: se è troppo alta il modello impazzisce, se è troppo bassa impara lentamente.

Il **batch** è quanti esempi vengono elaborati insieme. Con immagini grandi il batch deve essere piccolo, perché la memoria non basta.

Perché tre fasi:
- La **fase 1** accelera tutto e previene i NaN, che compaiono con batch piccoli in precisione ridotta (*mixed precision*).
- La **fase 2** usa il *bucketing*: ogni scheda grafica riceve solo volumi della stessa forma, così può usare batch più grandi.
- La **fase 3** corregge lo squilibrio che la fase 2 ha lasciato: alcuni dataset erano più numerosi di altri.

Il training completo è stato fatto su **64 schede A100 da 80 GB**, per circa tre settimane.

Il **ControlNet** è addestrato a parte: 8 schede A100, ottimizzatore AdamW, learning rate 5e-5 con decadimento polinomiale, 60 epoch, circa 2 giorni. Le maschere di addestramento sono **pseudo-etichette**: non sono state disegnate a mano, ma create dal modello VISTA3D (He et al., 2024) su 127 strutture e lesioni. Dove esistono etichette manuali, le sostituiscono.

### 3.6 I dati di addestramento

- **Circa 11.000 scansioni TC** di partenza, diventate circa **107.000 volumi** con la preparazione.
- Fonti pubbliche: HNSCC (testa e collo), AbdomenCT-1K, AeroPath, AMOS22, COVID-19, CRLM-CT, CT-ORG, CTPelvic1K, LIDC (polmoni), NLST, MSD (task 3, 6, 7, 8, 9, 10), Multi-organ, StonyBrook, TCIA Colon, TotalSegmentator v2.
- Due dataset interni: 237 TC con lesioni ossee e 1.386 TC di torace e addome.

Le scelte di preparazione:
- Tutte le immagini ridimensionate a dimensioni multiple di 128 (128, 256, 384, 512, 768 per asse).
- Rimosse le TC con meno di 64 fette (*slice*), per qualità.
- I volumi ad alta risoluzione vengono anche ridotti di risoluzione, per avere più esempi a bassa risoluzione.

**Attenzione**: tutto il training è su **TC**. Le risonanze non sono nel paper; sono arrivate con le versioni successive.

### 3.7 I risultati, con i numeri

**Velocità e qualità (Tabella 1).** Creazione di un volume 512×512×512 con spaziatura di 1 mm, su una scheda H100. Il FID è calcolato su un dataset di test diverso da quello di addestramento (AutoPET2023). Più basso è meglio.

| Modello | Passi | Tempo | FID medio |
|---|---|---|---|
| HA-GAN | 1 | 1 s | 13.595 |
| MedSyn | 50 + 20 | 100 s | 24.709 |
| GenerateCT | 25 × 201 | 89 s | 10.757 |
| MAISI (DDPM) | 1000 | 198 s + 15 s di decodifica | 2.441 |
| MAISI (DDIM) | 6 + 15 | 4.776 |
| **MAISI-v2 (rectified flow)** | **30** | **6 s + 15 s di decodifica** | **2.322** |

In parole semplici: MAISI-v2 ha una qualità simile a MAISI-DDPM (anzi leggermente migliore nel punteggio medio), ma è circa **33 volte più veloce** nella parte di creazione. I 15 secondi della decodifica restano uguali.

**Numero di passi (Tabella 3).** Più passi = immagini migliori, ma con rendimenti decrescenti:

| Passi | 5 | 10 | 20 | 30 | 40 | 50 | 70 | 100 |
|---|---|---|---|---|---|---|---|---|
| FID medio | 20.3 | 4.4 | 2.6 | 2.3 | 2.2 | 2.1 | 2.0 | 2.0 |

Gli autori scelgono **30 passi** come valore predefinito, perché dopo 30 il miglioramento è piccolo.

**Confronto con i video (Tabella 2).** Il paper confronta anche con modelli di video, come Stable Video Diffusion e Open-Sora 2.0, per dare un'idea di scala. Attenzione: non è un confronto sullo stesso compito. Ma dà un'idea: MAISI-v2 con la maschera impiega circa 34 secondi per un volume da 512×512×512, contro 100 e 162 secondi dei modelli video.

**Utilità per addestrare altri modelli (Tabella 4).** Questo è il test più importante per il tuo lavoro. Si addestra un modello di segmentazione (Auto3DSeg) con dati reali, e poi con dati reali più dati sintetici. Si misura il Dice su un test di dati reali.

| Tumore | Solo dati reali | Con MAISI-v2 | Miglioramento |
|---|---|---|---|
| Fegato | 0.662 | 0.695 | +3.3% |
| Polmone | 0.581 | 0.655 | +7.5% |
| Pancreas | 0.433 | 0.497 | +6.4% |
| Colon | 0.449 | 0.491 | +4.2% |
| Ossa | 0.504 | 0.537 | +3.3% |

Il paper dice che il miglioramento è statisticamente significativo in **4 casi su 5** (test t su campioni appaiati). Attenzione: nella copia PDF che ho letto alcuni simboli di significatività si sono persi. Prima di citare i valori di significatività, controlla la tabella originale.

**Il risultato più interessante.** Se si usa il ControlNet semplice, senza la perdita contrastiva, il colon **peggiora** (−4.7%). Con la perdita contrastiva diventa **migliore** (+4.2%). Quindi la perdita contrastiva fa la differenza.

**Controllo di qualità sui valori HU.** Gli autori hanno calcolato gli intervalli normali della mediana dei valori HU per fegato, milza, pancreas, reni, polmoni, ossa e cervello. Un volume passa il controllo se la mediana di ogni organo cade nell'intervallo. Con un `λ` scelto bene, il 100% dei volumi passa. Questo controllo è la base del file `quality_check.py` del codice.

### 3.8 Limiti dichiarati dagli autori

1. **Solo TC.** Le risonanze e la PET sono nel piano futuro.
2. **Solo la segmentazione come test.** Non hanno valutato altri compiti, come la traduzione da un'immagine all'altra, il riempimento di zone mancanti (*inpainting*) o il rilevamento.
3. **Memoria**: 24 GB bastano per volumi 512×512×256. Per 512×512×768 servono 40 GB per creare e 80 GB per addestrare.

Un limite che il paper non dichiara: un FID basso non garantisce che il tumore sia nel posto giusto dal punto di vista medico. Per questo serve anche il Dice sul tumore.

### 3.9 Il punto debole del rectified flow sui tumori

Il paper ammette che con il rectified flow il tumore a volte **segue meno bene la maschera** rispetto al DDPM, soprattutto per il colon. Gli autori danno due spiegazioni:
- Il percorso deterministico e dritto riduce la varietà del percorso, quindi il modello è meno robusto a piccole variazioni della condizione.
- Un errore in una regione piccola o poco contrastata si accumula durante il percorso.

La perdita contrastiva è la risposta degli autori a questo problema. Per il tuo caso è importante: anche il tuo tumore è piccolo e poco contrastato.

---

## 4. Il blog NVIDIA e i modelli disponibili

### 4.1 Che cosa dice il blog

Il blog, del 22 maggio 2026, presenta NV-Generate-CTMR come un **sistema completo e aperto** per creare TC e risonanze sintetiche. Sostiene che è il primo progetto aperto che, con un solo modello, gestisce voxel di dimensioni diverse, volumi di dimensioni diverse e tutto il corpo.

Il blog insiste su un principio: **riusare un modello è meglio che addestrarlo da zero**. Il fine-tuning, cioè il riaddestramento leggero su dati nuovi, è più veloce e consuma meno energia.

Cita un uso concreto: immagini-maschera con tumori generate da questo progetto sono servite come dati aggiuntivi per **NV-Segment-CTMR**, un modello di segmentazione NVIDIA.

Attenzione: affermazioni come "primo progetto aperto" e "stato dell'arte" sono dell'azienda, non verificate indipendentemente.

### 4.2 La tabella dei modelli

| Modello | Parte del corpo | Quando | Architettura | Passi | Volume massimo | Uso | Licenza |
|---|---|---|---|---|---|---|---|
| `ddpm-ct` | Tutto il corpo, TC | ago 2024 | MAISI-v1 | 1000 | 512×512×768 | Solo immagine, o coppia immagine/maschera | Aperta, commerciale |
| `rflow-ct` | Tutto il corpo, TC | mar 2025 | MAISI-v2 | 30 | 512×512×768 | Solo immagine, o coppia immagine/maschera | Aperta, commerciale |
| `rflow-mr` | Cervello, prostata, addome, seno | ott 2025 | MAISI-v2 | 30 | 512×512×128 | Solo immagine | **Solo ricerca** |
| `NV-Generate-MR-Brain` | Cervello intero o senza cranio | mar 2026 | MAISI-v2 | 30 | 512×512×256 | Solo immagine; passaggio tra sequenze | Aperta, commerciale |

Il blog dice che la maggior parte dei modelli è sotto la **NVIDIA Open Model License**. C'è però una contraddizione: nella tabella `rflow-mr` risulta "solo ricerca". Per il tuo lavoro conta: se usi `rflow-mr` per scopi commerciali, non sei coperto.

### 4.3 Il modello per il cervello

`NV-Generate-MR-Brain` è quello che ti interessa di più:
- basato su MAISI-v2, addestrato su **MR-RATE**: 100.000 studi di risonanza cerebrale di più di 83.000 pazienti, circa 700.000 volumi, con referti radiologici e metadati DICOM (il formato standard dei dati delle macchine medicali);
- crea il cervello intero o senza cranio, a scelta;
- sequenze: **T1, T2, FLAIR e SWI** (la SWI è sensibile al sangue e al ferro);
- risoluzione fino a 512×512×256;
- include un componente ControlNet per le strutture e per passare da una sequenza all'altra (per esempio prevedere una T2 partendo da una T1).

Il dataset MR-RATE è con licenza CC-BY-NC: uso libero solo per la ricerca. Per uso commerciale bisogna rivolgersi a Forithmus, l'azienda che lo distribuisce.

**Attenzione dal codice**: per `rflow-mr-brain` lo script di download scarica solo l'autoencoder (dal repo CT) e il modello principale (dal repo cervello). **Non scarica un ControlNet per il cervello.** Quindi il ControlNet cerebrale citato dal blog non risulta scaricabile con gli script attuali. Da verificare su Hugging Face.

### 4.4 Il comando d'avvio del blog

```bash
git clone https://github.com/NVIDIA-Medtech/NV-Generate-CTMR.git
cd NV-Generate-CTMR
export MONAI_DATA_DIRECTORY="./temp_work_dir"
network="rflow"
generate_version="rflow-ct"
python -m scripts.inference -t ./configs/config_network_${network}.json -i ./configs/config_infer.json -e ./configs/environment_${generate_version}.json --random-seed 0 --version ${generate_version}
```

Che cosa fa, in parole semplici: scarica il progetto, imposta una cartella per i dati temporanei, e lancia il programma principale per creare coppie immagine-maschera di TC.

Attenzione: questo comando è per la TC, non per il cervello.

---

## 5. Come è organizzato il codice

### 5.1 Le cartelle principali

| Cartella | Che cosa contiene | Note |
|---|---|---|
| `scripts/` | I programmi veri e propri, in Python | 21 file, circa 9.000 righe |
| `configs/` | File JSON con le impostazioni | 40 file: come lavorare, quali modelli, quali nomi |
| `docs/` | Documenti di spiegazione | 8 file Markdown |
| `skills/` | Guide operative per le stesse procedure | 9 file Markdown |
| notebook (`.ipynb`) | Tutorial interattivi | 5 nella radice, 1 in `data/`. Pesanti perché contengono anche le immagini |
| `.github/workflows/ci.yml` | Un controllo automatico | Solo lo stile del codice (ruff). **Nessun test** |

Un **notebook** è un documento che mescola testo e codice eseguibile, cella per cella. Si apre con Jupyter.

### 5.2 Come si lancia un programma

Dalla cartella principale del progetto:

```bash
python -m scripts.nome_del_file
```

Esempio: `python -m scripts.inference` lancia `scripts/inference.py` (senza `.py`). Il file `scripts/__init__.py`, vuoto a parte la licenza, serve a Python per riconoscere la cartella come un pacchetto. Il progetto non si installa come pacchetto: si lancia direttamente.

### 5.3 Le librerie esterne

Gran parte dei mattoni dei modelli (le reti neurali, cioè i "grafi di calcolo" che imparano) **non è nel progetto**. Arrivano dalla libreria **MONAI**, una libreria per l'imaging medico costruita su PyTorch. Quando nel codice leggi `AutoencoderKlMaisi`, `DiffusionModelUNetMaisi`, `ControlNetMaisi` o `RFlowScheduler`, la definizione è in MONAI.

Le altre librerie principali:
- `torch` (PyTorch): il motore di calcolo per le reti neurali.
- `numpy` e `scipy`: calcoli matematici sugli array.
- `scikit-image`: elaborazione di immagini.
- `nibabel`: lettura e scrittura dei file NIfTI.
- `huggingface_hub`: scaricamento dei pesi dal sito Hugging Face.
- `fire`: trasforma una funzione Python in un comando da terminale.
- `tensorboard`: grafici dell'addestramento.

### 5.4 Come si collegano i file

Il programma principale è `inference.py`. Lui chiama gli altri:

- `inference.py` chiama `download_model_data`, `sample.LDMSampler`, i controlli `check_input_*` di `sample.py` e `define_instance` di `utils.py`.
- `sample.py` (il direttore d'orchestra) chiama `augmentation`, `find_masks`, `quality_check`, `infer_image_from_mask`, `sample_mask` e alcune funzioni di `utils`.
- `sample_mask.py` (creazione della mappa da zero) chiama `utils` e `utils_infer`.
- `infer_image_from_mask.py` (immagine dalla mappa) chiama `augmentation`, `utils` e `utils_infer`.
- `utils_infer.py` (il motore dell'immagine) chiama `utils`.
- `diff_model_*.py` (modello senza mappa) chiama `download_model_data`, `sample`, `utils`, `utils_infer`.
- `train_controlnet.py` chiama `augmentation`, `diff_model_setting` e `utils`.
- `diff_model_create_training_data.py` chiama `transforms` e `utils`.
- `infer_image_from_mask_batch.py` chiama `infer_image_from_mask`, `utils` e `utils_infer`.

Il file più centrale è `utils.py`, con quasi 800 righe: tutti gli altri ne usano le funzioni.

---

## 6. A cosa serve ogni file

Per ogni file trovi: scopo, funzioni principali, comandi da terminale, che cosa legge e che cosa scrive. Le righe tra parentesi indicano dove trovare la cosa nel file.

### 6.1 `inference.py` (261 righe): la pipeline completa, coppia immagine + maschera

**Scopo.** È la porta d'ingresso per creare coppie immagine-maschera di TC. Carica tutti i modelli, crea il "direttore d'orchestra" (`LDMSampler`) e genera le coppie.

**Funzione principale**: `main()` (riga 29).

**Parametri da terminale**:

| Flag | Default | Significato |
|---|---|---|
| `-e` / `--environment-file` | `./configs/environment.json` (**non esiste**) | Percorsi, dizionari, archivio delle maschere |
| `-t` / `--config-file` | `./configs/config_maisi.json` (**non esiste**) | Struttura della rete; si passa `config_network_*.json` |
| `-i` / `--inference-file` | `./configs/config_infer.json` | Impostazioni di creazione |
| `-x` / `--extra-config-file` | nessuno | Impostazioni aggiuntive, per esempio `config_trt.json` |
| `-s` / `--random-seed` | nessuno | Il seme casuale (vedi [sezione 11](#11-problemi-noti)) |
| `--version` | `rflow-ct` | Passato al download; non viene verificato |
| `--low-vram` | disattivo | Tiene i modelli inattivi sulla CPU, per schede con poca memoria |

Il **seme casuale** (seed) è un numero che fissa la sequenza del caso: con lo stesso seme, il risultato è sempre uguale. Serve per riprodurre un esperimento.

**Che cosa fa, in ordine**:
1. Imposta la memoria della scheda grafica e il seme casuale.
2. Stabilisce la cartella dei dati: quella di `MONAI_DATA_DIRECTORY`, oppure una cartella temporanea.
3. Se hai indicato dimensioni d'organo o dati del paziente, va in **Percorso A** (mappa da zero) e non scarica l'archivio. Altrimenti va in **Percorso B** e scarica anche l'archivio.
4. Legge l'ambiente e costruisce i percorsi dei file.
5. Unisce le impostazioni della rete, della creazione e quelle extra. Se non indicate, usa `cfg_guidance_scale = 0` e `mask_generation_cfg_guidance_scale = 2.0`.
6. Controlla l'input con `check_input_ct` (modalità da 1 a 7, cioè TC) oppure `check_input_mr` (da 8 a 20, cioè risonanza).
7. Crea i due "scheduler", cioè i programmi che decidono il rumore passo per passo: uno per l'immagine, uno per la maschera.
8. Imposta la GPU in modo fisso: `torch.device("cuda")`. Senza GPU NVIDIA non va.
9. Carica i pesi dei modelli: autoencoder dell'immagine, modello principale dell'immagine, ControlNet (copiato dal modello principale e poi caricato), autoencoder della maschera, modello della maschera.
10. Crea il `LDMSampler` e lancia la generazione di N campioni.
11. Stampa la memoria massima usata.

**Che cosa legge**: i file di ambiente, rete, creazione e quelli extra; tutti i checkpoint (i file dei pesi); l'archivio e i dizionari; le statistiche dei tessuti.

**Che cosa scrive**: per ogni campione, `sample_<data-ora>_image.nii.gz` e `sample_<data-ora>_label.nii.gz`.

**Dipende da**: `download_model_data.py`, `sample.py`, `utils.define_instance`.

### 6.2 `infer_image_from_mask.py` (527 righe): dalla tua maschera all'immagine

**Scopo.** È il file più utile per il tuo lavoro. Gli dai una maschera NIfTI che hai già (per esempio la tua segmentazione) e lui crea l'immagine TC o MR che corrisponde. È sia una libreria da importare sia un programma da terminale.

**Funzioni principali**:
- `ldm_conditional_sample_one_image_from_mask` (riga 65): la funzione principale. Per la TC lo sfondo vale −1000 HU; per la risonanza vale 0. Se la CFG è maggiore di zero, crea la versione "senza tumore" della maschera (per la guida). Poi chiama il motore `run_controlnet_conditioned_image_dm` e ritaglia il corpo.
- `crop_img_body_mask` (riga 173): i punti fuori dal corpo vengono riportati allo sfondo.
- `validate_user_mask` (righe 294–379): controlla la tua maschera. Carica il file, la porta nell'orientamento standard RAS, avvisa se ci sono numeri di struttura fuori dall'intervallo 0–132 o se manca il numero 200 (corpo), e la ricampiona se dimensione o spaziatura non vanno bene.
- `_suggest_valid_target` (riga 251): propone una dimensione valida che mantiene più o meno lo stesso FOV.
- `_USER_MASK_FORMAT_WARNING` (righe 214–243): avviso stampato su stderr sul formato della maschera.

**Parametri da terminale** (righe 391–415):

| Flag | Obbligatorio | Significato |
|---|---|---|
| `--mask` | sì | La tua maschera NIfTI |
| `-t` | sì | `config_network_*.json` |
| `-e` | sì | `environment_*.json` |
| `-i` | sì | `config_infer*.json` |
| `--random-seed` | no, default 0 | Seme |
| `--low-vram` | no | Usa la CPU per alcuni modelli |

**Flusso**: valida la maschera; carica i modelli (con ripiego sulla CPU alla riga 433); costruisce i tensori della condizione, cioè i dati che il modello legge come istruzioni, con forma `[C, X/4, Y/4, Z/4]`; genera; salva con `SaveImage` di MONAI.

**Che cosa scrive**: solo `<nome_maschera>_<data-ora>_image.nii.gz`. **Non salva la maschera** (alcune skill dicono il contrario).

**Dipende da**: `augmentation.remove_tumors`, `utils.binarize_labels`, `utils_infer`, `diff_model_setting.load_config`.

### 6.3 `infer_image_from_mask_batch.py` (196 righe): la stessa cosa su tanti casi

Legge un file JSON (chiamato *manifest*), che elenca più casi, e crea le immagini di tutti. Usa solo il "fold" di validazione del dataset, quindi non tutti i casi. Con più schede grafiche (`-g`) lavora in parallelo.

Chiavi attese nel manifest: `label` (il file della maschera), `dim` (la dimensione), `spacing` (la spaziatura). La regione del corpo serve solo per il DDPM.

**CLI**: `-t`, `-e`, `-i` obbligatori; `-g` (default 1); `--low-vram`.

**Che cosa scrive**: `sample_<data-ora>_image.nii.gz` e `_label.nii.gz` nella cartella di uscita.

### 6.4 `sample.py` (718 righe): il direttore d'orchestra

**Scopo.** Coordina tutto il lavoro di `inference.py`. Decide se usare il Percorso A (mappa da zero) o il Percorso B (mappa dall'archivio), gestisce i tentativi, impone la dimensione e la spaziatura, fa il controllo di qualità e salva.

**Classe `LDMSampler`**:
- `__init__` (righe 71–225): conserva i modelli, trasforma i nomi delle strutture richieste in numeri usando il dizionario, prepara le liste per il controllo di qualità. Richiede che la finestra di calcolo dell'autoencoder sia multipla di 16.
- `sample_multiple_images(num_img)` (righe 227–369): il ciclo principale. Dettagli nella [sezione 9](#9-il-percorso-di-unimmagine-passo-per-passo).
- `select_mask` (righe 371–388): mescola e cicla le maschere dell'archivio. L'augmentation è sempre attiva.
- `sample_one_pair`, `prepare_anatomy_size_condition`, `prepare_one_mask_and_meta_info`, `sample_one_mask`: passaggi intermedi.
- `ensure_output_size_and_spacing` (righe 537–580): ricampiona con il metodo "nearest" (prende il valore del punto più vicino, senza mescolare le etichette) e poi aggiunge o taglia il bordo per arrivare alla dimensione finale. Se le strutture richieste vanno perse, solleva un errore.
- `read_mask_information` (righe 582–605): legge le informazioni di una maschera.
- `find_closest_masks` (righe 607–667): cerca le maschere più simili.
- `resample_mask_check_organ_list` (righe 669–698): scarta le maschere che hanno perso gli organi richiesti.
- `quality_check_ct` (righe 700–718): il controllo di qualità.

**Che cosa scrive**: le coppie immagine-etichetta. Nel Percorso B l'etichetta viene filtrata e tiene solo le strutture richieste; nel Percorso A tiene tutte le 132 classi.

### 6.5 `sample_mask.py` (400 righe): la mappa da zero, e i controllori degli input

**Scopo.** Crea una mappa delle strutture da zero, usando il modello delle maschere (versione 2, con AdaGN). Il modello è guidato da un vettore di **19 numeri** che descrive cosa vuoi. Contiene anche i due controllori degli input (`check_input_ct` e `check_input_mr`).

**Costanti importanti**:
- `ANATOMY_SIZE_IDX` (righe 57–72): i primi 14 numeri del vettore sono le dimensioni degli organi e dei tumori richiesti. Per esempio: fegato (posizione 0), milza (1), stomaco (2), pancreas (3), colon (4), rene sinistro (5), rene destro (6), polmone (7), cistifellea (8), tumore del polmone (9), tumore del pancreas (10), tumore del fegato (11), tumore del colon (12), lesione ossea (13).
- `_DEMOG_MAX` (riga 53): i valori massimi per normalizzare età, peso, BMI e altezza. Servono a riportare tutto tra 0 e 1.
- `N_ANATOMY = 14`, `N_DEMOG = 5`, `N_COND = 19` (19 = 14 + 5).

**Funzioni principali**:
- `ldm_conditional_sample_one_mask` (righe 78–195): genera la mappa. I passi: il vettore "vuoto" è tutto −1; la spaziatura viene moltiplicata per 100 (per averla come numero intero leggibile); parte il rumore; il RFlow fa 100 passi con CFG 2.0; il modello dà 125 canali; si sceglie la classe più probabile per ogni punto (argmax); si rimappano le classi nelle 132 finali; si sceglie il tumore target; si applicano le regole di pulizia.
- `filter_mask_with_organs` (righe 198–215): tiene solo le etichette richieste.
- `check_input_ct` (righe 218–346): regole per la TC: dimensioni (righe 242–245), spaziatura (246–249), campo di vista di almeno 256 mm (250–254), controlli sui dati del paziente (256–291), **una sola voce di anatomia** (298–302), controlli per il Percorso B (336–345).
- `check_input_mr` (righe 349–400): regole simili per la risonanza. Controlla che le strutture richieste esistano nel dizionario.

### 6.6 `utils_infer.py` (491 righe): il motore che crea le immagini

**Scopo.** È il cuore della creazione dell'immagine dalla mappa. È condiviso da più programmi.

**Funzioni principali**:
- `move_models` (riga 59): sposta i modelli sulla GPU o sulla CPU.
- `ReconModel` (righe 67–92): il decompressore. Riporta l'immagine compressa all'immagine piena, dividendo per il fattore di scala.
- `initialize_noise_latents` (righe 95–115): crea il rumore di partenza, come una immagine di neve casuale nello spazio compresso.
- `run_controlnet_conditioned_image_dm` (righe 118–341): **il ciclo centrale**. Per ogni passo: il ControlNet guarda la maschera e produce indicazioni; il modello principale riceve queste indicazioni e corregge l'immagine; lo scheduler fa un piccolo passo verso l'immagine finale. Alla fine decomprime e converte in HU.
- `load_image_models` (righe 344–390): carica autoencoder, modello principale, ControlNet e scheduler.
- `load_mask_models` (righe 393–434): carica autoencoder, modello e scheduler delle maschere.
- `load_paired_inference_models` (righe 437–460): carica entrambi.
- `build_conditioning_tensors` (righe 463–491): prepara le condizioni: la spaziatura (×100), gli indicatori delle regioni del corpo (solo se il modello li usa), la modalità.

Il **tensore** è semplicemente un array multidimensionale, cioè una griglia di numeri con più assi.

### 6.7 `utils.py` (816 righe): gli attrezzi generali

Il file più grande. Ha quattro gruppi di funzioni.

**Gruppo 1: sistemare le maschere e le etichette**
- `remap_labels` (righe 35–56): cambia numeri di etichetta secondo una tabella di coppie `[vecchio, nuovo]`.
- `binarize_labels` (righe 173–188): trasforma il numero di ogni punto in 8 "strati" binari. Questo è l'input che il ControlNet riceve. Per questo le etichette possono arrivare a 255.
- `organ_fill_by_closing` (righe 343–366) e `organ_fill_by_removed_mask` (righe 369–386): riempiono i buchi negli organi.
- `get_body_region_index_from_mask` (righe 389–425): riconosce la regione del corpo dalla maschera. Testa: etichette 22 e 120. Torace: 28–32. Addome: 1, 2, 3, 4, 5, 14. Bacino: 93 e 94.
- `add_body_envelope` (righe 428–565): crea il contorno del corpo partendo dall'aria nell'immagine (sotto −800 HU). Rimuove il lettino e riempie il corpo con l'etichetta 200.
- `general_mask_generation_post_process` (righe 568–695): la pulizia più complessa, dopo la generazione. Contiene la tabella `tumor_organ_dict = {23: 28, 24: 4, 26: 1, 27: 62, 128: 200}`, regole per l'intestino, riempimenti per chiusura, regole per i tumori 23, 26, 27 e 129, e le regole per i vasi epatici (25 diventa fegato 1) e per le vie aeree (132 resta).

**Gruppo 2: dati e oggetti**
- `define_instance` (righe 208–224): crea un oggetto MONAI leggendo un file JSON con la voce `_target_`.
- `add_data_dir2path` (righe 227–258): aggiunge il percorso della cartella ai nomi dei file e divide tra training e validazione.
- `prepare_maisi_controlnet_json_dataloader` (righe 261–340): prepara i dati per il ControlNet. Usa `CacheDataset` con 8 processi in parallelo, che tiene i dati in memoria per non rileggerli.

**Gruppo 3: perdite**
- `KL_loss` (righe 763–782): misura quanto la distribuzione compressa si discosta dalla normale. Serve al VAE.
- `MapLabelValue` (righe 698–760): rimappatura con MONAI.

**Gruppo 4: creazione a blocchi**
- `dynamic_infer` (righe 785–816): elabora un volume grande a blocchi sovrapposti (finestra scorrevole), perché non sta in memoria in un colpo solo.

### 6.8 `find_masks.py` (157 righe): cercare una maschera nell'archivio (Percorso B)

**Funzione**: `find_masks(body_region, anatomy_list, spacing, output_size, check_spacing_and_output_size, database_filepath, mask_foldername)` (righe 55–157).

Che cosa fa, passo per passo:
- Trasforma il nome della regione in un numero: testa 0, torace 1, addome 2, bacino 3 (`convert_body_region`, righe 21–52).
- Se la cartella delle maschere non esiste, estrae lo zip dell'archivio (righe 89–97).
- Carica il file del database (righe 99–102).
- Tiene un elemento solo se **tutte** le strutture richieste sono presenti (riga 107) e se le regioni del corpo coprono quelle richieste (righe 114–124).
- Se nell'elenco non c'è un tumore, **scarta** gli elementi che hanno etichette tumorali `[23, 24, 26, 27, 128]` (righe 126–129).
- Se richiesto, controlla che spaziatura e dimensione siano esattamente quelle (righe 131–135).

Restituisce una lista di dizionari con nome del file della maschera, spaziatura, dimensione, eventuali regioni.

### 6.9 `quality_check.py` (148 righe): il controllo di qualità

- `get_masked_data` (righe 15–66): estrae i punti di un organo dalla maschera. Con almeno tre etichette usa `np.isin` (controlla se il valore è in un elenco); altrimenti usa il "OR logico" (una condizione semplice).
- `is_outlier` (righe 69–148): confronta la mediana dell'organo con l'intervallo normale. Il limite inferiore è il più estremo tra il percentile 0.5 e la soglia 6-sigma; il limite superiore è il più estremo tra il percentile 99.5 e la soglia 6-sigma. Per le ossa il limite superiore è 1000 HU (righe 114–115).

Sigma è la deviazione standard, cioè quanto i valori si allontanano in media dal centro. "6-sigma" è una soglia molto ampia, quindi scatta solo per valori davvero strani.

Controlla solo sette gruppi: fegato, milza, pancreas, rene, polmone, osso e cervello. **Non controlla i tumori.**

### 6.10 `augmentation.py` (594 righe): variazioni delle maschere reali

**Scopo.** Prende una maschera reale dall'archivio e la varia: la ruota, la allunga, la sposta, la deforma (con una deformazione elastica, come una gomma che si piega). Così il modello vede molte varianti invece di una sola.

**Due blocchi.**

*Blocco 1: l'augmentation vera e propria (Percorso B)*
- `augmentation_tumor_bone` (righe 68–118): lesione ossea, etichetta 128. Usa trasformazioni affini (rotazioni, spostamenti) e controlla che la lesione stia dentro l'osso. Ripete finché la sovrapposizione è sufficiente (soglia 0.8 per i primi 40 tentativi, poi 0.75).
- `augmentation_tumor_liver` (righe 121–168): tumore del fegato, etichetta 26, con deformazione elastica.
- `augmentation_tumor_lung` (righe 171–226): tumore del polmone, etichetta 23, spostato dentro un lobo.
- `augmentation_tumor_pancreas` (righe 229–276): tumore del pancreas, etichetta 24, dentro il pancreas (4).
- `augmentation_tumor_colon` (righe 279–334): tumore del colon, etichetta 27, dentro il colon (62). Si ferma dopo 40 tentativi.
- `augmentation_body` (righe 337–346): zoom casuale tra 0.99 e 1.01 quando non ci sono tumori.
- `augmentation_tumor_only` (righe 349–408): versione generica per i tumori cerebrali di BraTS (etichette 401–403).
- `augmentation` (righe 436–481): sceglie quale funzione usare in base alle etichette presenti.

*Blocco 2: rimuovere i tumori (serve per la versione "senza tumore")*
- `remove_tumors` (righe 484–520): sostituisce le etichette dei tumori con quelle dell'organo di origine. La tabella è `{26: 1, 24: 4, 27: 62, 116: 14, 117: 5}`. Per il polmone usa un voto a maggioranza tra i vicini (`remove_tumors_majority_vote`, righe 523–577).
- `remap_labels` (righe 580–594): una versione locale della rimappatura. **Attenzione**: è diversa da quella in `utils.py`.

**Problema importante.** I cicli di augmentation per fegato, polmone e pancreas sono `while True` (righe 149, 205, 257): girano finché non trovano una variante valida. Se la maschera non ha abbastanza spazio dentro l'organo, il programma può non finire mai. Solo BraTS e colon hanno un limite. Vedi [sezione 11](#11-problemi-noti).

### 6.11 `diff_model_setting.py` (139 righe): le impostazioni comuni

- `setup_logging` (righe 26–44): imposta i messaggi di log, cioè quello che il programma scrive mentre lavora.
- `load_config(env, model_cfg, model_def)` (righe 47–76): unisce i file JSON. Se due file hanno la stessa chiave, vince l'ultimo.
- `initialize_distributed` (righe 79–95): prepara il calcolo su più schede (DDP, *distributed data parallel*).
- `run_torchrun` (righe 98–139): lancia sottoprocessi con torchrun. **Nessun file Python lo chiama**: è codice inutilizzato.

### 6.12 `diff_model_train.py` (541 righe): addestrare il modello principale dell'immagine

**Scopo.** Insegna al modello principale a creare immagini, partendo dalle versioni compresse di immagini reali (gli "embedding", prodotti da `diff_model_create_training_data.py`).

**Funzioni principali**:
- `augment_modality_label` (righe 36–66): ogni tanto (10% delle volte) sostituisce la modalità con 0. Così il modello impara anche a lavorare senza indicazioni di tipo. Lo zero è la "classe nulla" della CFG. Per la TC i codici 2–7 diventano 1; per la risonanza quelli da 9 in su diventano 8.
- `load_filenames` (righe 69–82): trasforma i nomi dei file per trovare gli embedding.
- `prepare_data` (righe 85–139): legge la spaziatura da un file accompagnatore, la regione del corpo e la modalità. Usa 6 processi per caricare i dati.
- `load_unet` (righe 142–171): carica il modello principale, con DDP se ci sono più schede.
- `calculate_scale_factor` (righe 174–195): calcola il fattore di scala come 1 diviso la deviazione standard del primo batch. Il fattore viene salvato nel file dei pesi.
- `create_optimizer` (righe 198–209): ottimizzatore Adam.
- `create_lr_scheduler` (righe 212–223): riduce il learning rate con una curva polinomiale di potenza 2.
- `train_one_epoch` (righe 226–366): moltiplica le immagini per il fattore di scala, sceglie il passo di tempo casuale, calcola il bersaglio (per RFlow: immagine meno rumore), calcola l'errore L1 (la differenza assoluta media), aggiorna i pesi. Usa l'AMP (precisione mista, più veloce).
- `save_checkpoint` (righe 369–400): salva `{epoch, loss, num_train_timesteps, scale_factor, unet_state_dict}`. Il file viene sovrascritto a ogni epoch.
- `diff_model_train` (righe 403–517): cerca tutti i file da addestrare. Se un file manca, lo salta in silenzio (righe 446–448).

**Parametri da terminale** (righe 520–541): `-e` (ambiente), `-c` (configurazione), `-t` (rete), `-g` (schede, default 1), `--no_amp` (disattiva la precisione mista).

**Che cosa legge**: gli embedding, i file accompagnatori, la lista dei dati, `modality_mapping.json`, il file dei pesi se si riparte da un punto già fatto.

**Che cosa scrive**: il file dei pesi in `model_dir/model_filename` a ogni epoch.

**Attenzione**: si aspetta un file accompagnatore `<embedding>.nii.gz.json` per ogni embedding (riga 450), ma nessun programma del progetto lo crea. Vedi [sezione 11](#11-problemi-noti).

### 6.13 `train_controlnet.py` (560 righe): addestrare il ControlNet

**Scopo.** Addestra il ControlNet, cioè il componente che legge la maschera e dice al modello dove mettere le strutture. Il modello principale resta fermo.

**Funzioni principali**:
- `remove_roi` (righe 37–56): rimuove i tumori dalla maschera. Serve solo per la perdita contrastiva.
- `compute_region_contrasive_loss` (righe 59–124): implementa la perdita per regione del paper. Nel nome c'è un refuso (`contrasive` invece di `contrastive`), ma la funzione è quella giusta.
- `compute_model_output` (righe 127–237): binarizza la maschera (riga 190), aggiunge il rumore (riga 193), passa per il ControlNet (righe 197–208) e per la UNet (righe 212–233).
- `train_controlnet` (righe 240–537): la funzione principale. Controlla la configurazione della perdita contrastiva (righe 259–277), carica il modello principale congelato e il fattore di scala (righe 292–302), blocca i pesi della UNet (righe 317–318), inizializza il ControlNet (riga 307), carica eventuali pesi precedenti (righe 309–315), prepara i dati (righe 337–346), crea l'ottimizzatore AdamW con learning rate 1e-5 (riga 351), applica il decadimento polinomiale (riga 355), attiva l'AMP (riga 394), calcola la perdita L1 con eventuale peso sulle etichette (righe 454–462) e la perdita contrastiva (righe 466–481), salva i pesi (righe 508–533).

**Parametri da terminale**: quelli di `diff_model_train.py`, senza `--no_amp`.

**Che cosa legge**: il modello principale (obbligatorio), eventuali pesi precedenti, la lista dei dati, `modality_mapping.json`, i file NIfTI.

**Che cosa scrive**: `{nome_esperimento}_current.pt` (l'ultimo stato), `{nome_esperimento}_best.pt` (il migliore) e i log per TensorBoard.

**Attenzione**: il caricatore di validazione viene **buttato via** (riga 337: `train_loader, _ = ...`). Quindi non c'è nessuna misura di come va su dati nuovi.

### 6.14 `diff_model_create_training_data.py` (287 righe): preparare gli embedding

**Scopo.** Prende le immagini reali e le comprime con il VAE, ottenendo un file `<nome>_emb.nii.gz` per ciascuna. È il primo passo prima di `diff_model_train.py`.

**Funzioni principali**:
- `create_transforms` (righe 33–74): carica l'immagine, la orienta in modo standard (RAS), applica la normalizzazione dell'intensità, eventualmente ridimensiona.
- `round_number` (righe 77–90): arrotonda le dimensioni al multiplo di 128 più vicino.
- `process_file` (righe 110–195): comprime con una finestra scorrevole di 320×320×160 e una sovrapposizione del 40%. Salva la media della distribuzione compressa (`z_mu`), che è la versione "centrale" dell'embedding. Se un file dà errore, lo registra e passa al successivo.
- `diff_model_create_training_data` (righe 198–263): carica il VAE e distribuisce i lavori tra le schede in modo alternato (round robin).

**Che cosa scrive**: `<percorso>_emb.nii.gz`. **Non scrive il file accompagnatore JSON** richiesto da `diff_model_train.py`.

**Parametri**: `-e`, `-c`, `-t` (default inesistenti), `-g`.

### 6.15 `diff_model_infer.py` (400 righe): creare solo immagini, senza mappa

**Scopo.** Crea immagini TC o risonanza usando solo il modello principale, senza maschera. Utile per la risonanza, per esempio.

**Funzioni principali**:
- `set_random_seed` (righe 37–49): fissa il seme.
- `load_models` (righe 52–82): carica l'autoencoder e il modello, con il fattore di scala dal file dei pesi.
- `run_inference` (righe 108–248): crea il rumore (righe 146–155), calcola i passi (righe 159–166), applica la CFG sulla modalità con la classe 0 come non condizionata (righe 205–215), decomprime con finestra fissa di 80³ e sovrapposizione 0.4 (righe 222–230). Per la risonanza l'intensità va da 0 a 1000; per la TC da −1000 a 1000 HU (righe 238–247). L'uscita è in formato intero a 16 bit (`int16`).
- `save_image` (righe 251–275): salva l'immagine con una matrice di posizione costruita dalla spaziatura.

Il nome del file contiene tutti i parametri usati: `{prefisso}_seed{seme}_size{X}x{Y}x{Z}_spacing{sx}x{sy}x{sz}_{data-ora}_rank{r}_modality{m}.nii.gz`. Così sai sempre come è stato creato.

**Parametri**: `-e`, `-c`, `-t` obbligatori; `-g`, `--low-vram`, `--version` (se presente scarica i pesi).

### 6.16 `download_model_data.py` (206 righe): scaricare i pesi

**Scopo.** Scarica i file dei modelli da Hugging Face, il sito dove sono pubblicati.

**Che cosa scarica, per versione**:
- `rflow-mr-brain`: l'autoencoder (dal repo `nvidia/NV-Generate-CT`) e il modello principale (`diff_unet_3d_rflow-mr-brain_v1.pt`, dal repo `nvidia/NV-Generate-MR-Brain`).
- `ddpm-ct` e `rflow-ct`: autoencoder, modello delle maschere e relativo autoencoder, modello principale dell'immagine, ControlNet. Senza `--model_only` scarica anche l'archivio delle maschere (zip) e i file con le dimensioni anatomiche e l'indice dei candidati.
- `rflow-mr`: `autoencoder_v2.pt` e `diff_unet_3d_rflow-mr.pt` (dal repo `nvidia/NV-Generate-MR`).

Funzioni principali: `ensure_hf_download_tracked` (righe 9–23, avvisa Hugging Face che il file è stato usato), `fetch_to_hf_path_cmd` (righe 26–76, scarica e copia, salta i file già presenti), `download_model_data` (righe 79–188).

**Parametri**: `--version` (default `rflow-ct`), `--root_dir` (default `./`), `--model_only`. Non ci sono `--token` o `--overwrite`, anche se alcune skill li descrivono.

**Attenzione**: `--root_dir` sposta **solo** i dataset. I modelli finiscono sempre in `./models/`, relativo alla cartella da cui lanci il comando.

### 6.17 `adagn_unet.py` (131 righe): il modello per le maschere

Definisce la classe `DiffusionModelUNetMaisiAdaGN`, una versione del modello principale che lavora con la **AdaGN** (*Adaptive Group Normalization*).

Che cosa significa, in parole semplici: a ogni blocco il modello normalmente aggiunge il "tempo" del processo. Qui invece calcola due numeri (una scala e uno spostamento) a partire dal vettore di 19 numeri, e li usa per regolare l'intero blocco. Il risultato è che la richiesta di dimensione entra in ogni parte del modello.

Funzioni:
- `_adagn_resnet_forward` (righe 12–37): applica scala e spostamento.
- `__init__` (righe 50–72): crea il piccolo MLP (19 → 256 → 256) che calcola scala e spostamento, e allarga i blocchi.
- `forward` (righe 107–131): concatena la condizione e la passa al modello. Se il contesto è a tre dimensioni, ne calcola la media.

### 6.18 `transforms.py` (310 righe): le trasformazioni dei dati

- `define_fixed_intensity_transform` (righe 45–71): per la TC, porta i valori da −1000…1000 a 0…1 con taglio. Per la risonanza usa i percentili 0–99.5, senza taglio.
- `define_random_intensity_transform` (righe 74–101): solo per la risonanza. Aggiunge variazioni casuali: campo di polarizzazione (*bias field*) con probabilità 0.3, artefatti di Gibbs (0.3), contrasto (0.3), spostamento dell'istogramma (0.05).
- `define_vae_transform` (righe 104–225): trasformazioni per il training del VAE: capovolgimenti, rotazioni di 90 gradi, zoom e rotazioni casuali, ritaglio casuale.
- `VAE_Transform` (righe 228–310): sceglie la pipeline per TC o risonanza in base al campo `class` del dato.

Il **campo di polarizzazione** è una sfumatura di luminosità che la macchina può produrre. Il **fenomeno di Gibbs** è un tipo di artefatto da ricostruzione.

### 6.19 `utils_plot.py` (195 righe): disegnare i risultati

Funzioni per disegnare fette e volumi 3D: `find_label_center_loc`, `normalize_label_to_uint8`, `visualize_one_slice_in_3d`, `show_image`, `to_shape`, `get_xyz_plot`. **Nessun file Python le importa**: le usano solo i notebook.

### 6.20 `compute_fid_2-5d_ct.py` (726 righe): misurare il FID

Calcola il FID su volumi TC, con una CLI costruita con `fire`.

- Usa come estrattore di caratteristiche la rete **RadImageNet ResNet-50**, scaricata con `torch.hub` (richiede internet la prima volta).
- Calcola il FID su tre piani (XY, YZ, ZX) e fa la media.
- Usa solo il 50% centrale delle fette, per non contare quelle vuote.
- Salva le caratteristiche in file `.pt`. I valori FID vengono solo scritti nel log.

Funzione principale: `main(...)` (riga 368), con dozzine di parametri, tra cui il numero di immagini (`num_images=100`) e la forma target (`512x512x512`).

### 6.21 `scripts/__init__.py` (10 righe)

Solo l'intestazione di licenza. Serve a Python per riconoscere la cartella come pacchetto.

---

## 7. I file di configurazione

I file in `configs/` sono come le impostazioni di un programma: non cambiano il codice, ma dicono come si comporta. Il programma è lo stesso, cambia il comportamento.

### 7.1 La rete: `config_network_rflow.json` e `config_network_ddpm.json`

**Parametri comuni (rflow)**:
- `spatial_dims = 3`: il modello lavora in 3D.
- `image_channels = 1`: l'immagine ha un solo canale (come una scala di grigi).
- `latent_channels = 4`: nello spazio compresso ci sono 4 canali.
- `include_body_region = false`: non usa le regioni del corpo (lo fa il DDPM).
- `mask_generation_latent_shape = [4, 64, 64, 64]`: la maschera compressa di 64³ corrisponde a una maschera di 256³ in voxel.

**Autoencoder dell'immagine** (`AutoencoderKlMaisi`, righe 12–38): canali `[64, 128, 256]`, due blocchi residuali per livello, nessuna attenzione, precisione a 16 bit per la normalizzazione, `num_splits = 4` (8 per il DDPM). `num_splits` è in quante parti si spezza il calcolo per risparmiare memoria.

**Modello principale dell'immagine** (`DiffusionModelUNetMaisi`, righe 39–65): canali `[64, 128, 256, 512]`. L'attenzione (il meccanismo che confronta le parti dell'immagine tra loro) è attiva solo ai due livelli più profondi (`[F, F, T, T]`), con 32 teste. Usa `include_spacing_input = true` (la spaziatura entra come condizione), `num_class_embeds = 128` (la modalità). Nel DDPM queste ultime due chiavi non ci sono.

**ControlNet** (`ControlNetMaisi`, righe 66–90): stessi canali. Riceve 8 canali di maschera binarizzata, con canali intermedi `[8, 32, 64]`.

**Autoencoder delle maschere** (righe 91–118): riceve 8 canali di maschera e produce **125 canali**, cioè le 125 classi del modello di maschere. Canali `[32, 64, 128]`, `use_checkpointing = true` (risparmia memoria ricalcolando alcuni passaggi invece di memorizzarli).

**Modello delle maschere** (righe 119–138): la classe `DiffusionModelUNetMaisiAdaGN`, con `cross_attention_dim = 19` e `adagn_condition_dim = 19`.

**Scheduler** (righe 140–155): il RFlow ha 1000 passi di riferimento durante l'addestramento, campionamento uniforme, e `scale = 1.4` per l'immagine, `scale = 1.0` per la maschera.

**Attenzione**: `mask_generation_scale_factor` è scritto nel file, ma **non viene mai letto**. Il codice usa il valore salvato nel file dei pesi.

### 7.2 La creazione: `config_infer.json` e le varianti per GPU

Il file di base `config_infer.json`:

| Chiave | Valore | Significato |
|---|---|---|
| `num_output_samples` | 1 | Quante coppie creare |
| `body_region` | `["chest"]` | Solo Percorso B |
| `anatomy_list` | `["lung tumor"]` | Solo Percorso B: le strutture richieste |
| `controllable_anatomy_size` | `[]` | Se non è vuoto, si va in Percorso A |
| `controllable_demographics` | `null` | Dati del paziente (età, sesso…) |
| `num_inference_steps` | 30 | Passi del RFlow per l'immagine (il DDPM ne vuole 1000) |
| `mask_generation_num_inference_steps` | 100 | Passi per la maschera |
| `mask_generation_cfg_guidance_scale` | 2.0 | CFG per la maschera |
| `output_size` | `[256, 256, 256]` | Dimensione finale in voxel |
| `spacing` | `[1.5, 1.5, 2.0]` | Spaziatura in mm |
| `autoencoder_sliding_window_infer_size` | `[48, 48, 48]` | Dimensione dei blocchi per la decompressione |
| `autoencoder_sliding_window_infer_overlap` | 0.6666 | Quanto si sovrappongono i blocchi |
| `modality` | 1 | TC |
| `cfg_guidance_scale` | 0.0 | CFG per l'immagine con tumore |

**Varianti per GPU** (`config_infer_<GB>g_<X>x<Y>x<Z>.json`). Ogni variante è pensata per una quantità di memoria della scheda grafica:

| File | Dimensione | Spaziatura | Finestra | Sovrapposizione | Parti | Passi |
|---|---|---|---|---|---|---|
| `16g_256x256x128` | 256, 256, 128 | 1.5, 1.5, 4.0 | 96, 96, 96 | 0.25 | 2 | 30 |
| `16g_256x256x256` | 256, 256, 256 | 1.5, 1.5, 2.0 | 48, 48, 64 | 0.6666 | 4 | 30 |
| `16g_512x512x128` | 512, 512, 128 | 0.75, 0.75, 4.0 | 64, 64, 32 | 0.5 | 2 | 30 |
| `24g_256x256x256` | 256, 256, 256 | 1.5, 1.5, 2.0 | 64, 64, 64 | 0.25 | 4 | **1000** |
| `24g_512x512x128` | 512, 512, 128 | 0.75, 0.75, 4.0 | 80, 80, 32 | 0.4 | 2 | 30 |
| `24g_512x512x512` | 512, 512, 512 | 0.75, 0.75, 1.0 | 64, 64, 48 | 0.4 | 2 | 30 |
| `32g_512x512x512` | 512, 512, 512 | 0.75, 0.75, 1.0 | 80, 80, 48 | 0.4 | 4 | 30 |
| `80g_512x512x512` | 512, 512, 512 | 0.75, 0.75, 1.0 | 80, 80, 80 | 0.4 | 4 | **1000** |
| `80g_512x512x768` | 512, 512, 768 | 0.75, 0.75, 0.66667 | 80, 80, 96 | 0.4 | 4 | 30 |

Due preset RFlow usano 1000 passi invece di 30. Sembra un errore di copia, ma non è verificato.

Il file **`config_trt.json`** sostituisce ControlNet, UNet, decoder e modelli delle maschere con versioni compilate con TensorRT, un motore di NVIDIA che accelera i calcoli. Richiede una GPU CUDA.

### 7.3 L'addestramento della diffusione: `config_maisi_diff_model_<variante>.json`

| Variante | Dimensione (`dim`) | Spaziatura | Passi | Modalità | CFG | Seme | Batch | Learning rate | Epoch |
|---|---|---|---|---|---|---|---|---|---|
| `rflow-ct` | 256, 256, 128 | 1.7, 1.7, 2 | 30 | 1 | 0 | 0 | 2 | 1e-5 | 1000 |
| `ddpm-ct` | 256, 256, 128 | 1.0, 1.0, 0.75 | 1000 | 1 | 0 | 0 | 1 | 1e-5 | 1000 |
| `rflow-mr` | 128, 256, 256 | 1.25, 1.0, 1.0 | 30 | 9 | **15** | 0 | 1 | 1e-5 | 1000 |
| `rflow-mr-brain` | 256, 256, 128 | 0.94, 0.94, 1.36 | 30 | 9 | **2** | 1234 | 1 | 1e-5 | 1000 |

Tutte le varianti hanno gli indicatori delle regioni (`[0,1,0,0]` e `[0,0,1,0]`), ma solo `ddpm-ct` li usa.

### 7.4 L'addestramento del ControlNet: `config_maisi_controlnet_train*.json`

| File | Addestramento | Creazione |
|---|---|---|
| generico | batch 1, fold 0, lr 1e-5, 100 epoch, `weighted_loss_label = [129]`, `weighted_loss = 100` | 10 passi, finestra `[80,80,80]`, sovrapposizione 0.4, modalità 1 |
| `rflow-ct` | come sopra, ma `weighted_loss_label = [23]` e perdita contrastiva attiva (δ = 2, peso 0.01) | 30 passi, `[80,80,80]`, 0.4, modalità 1 |
| `ddpm-ct` | `weighted_loss_label = [129]`, senza perdita contrastiva | 1000 passi, `[80,80,80]`, 0.4, modalità 1 |
| `rflow-mr` | `weighted_loss_label = [129]`, senza perdita contrastiva | 30 passi, `[80,80,32]`, 0.4, modalità 9 |

Il `weighted_loss_label` è l'etichetta a cui dare più peso nella perdita: il valore 129 corrisponde alla lesione (un numero della lista). Il `weighted_loss` è il peso.

Il file `rflow-mr` esiste, ma la documentazione dice che non c'è un ControlNet per la risonanza. Il suo ambiente punta a `temp_work_dir/...` (un esempio cerebrale), quindi sembra uno scheletro, non una configurazione pronta.

### 7.5 L'addestramento del VAE: `config_maisi_vae_train.json`

- **Dati**: `random_aug = true` (variazioni casuali), `spacing_type = "rand_zoom"` (zoom casuale), `select_channel = 0`.
- **Autoencoder**: batch 1, ritagli di `[64, 64, 64]`, validazione con finestra `[96, 96, 64]`, learning rate 1e-4, peso perceptuale 0.3, peso KL 1e-7, peso avversariale 0.1, perdita L1, validazione ogni 10 epoch, AMP attivo.
- **`n_epochs = 1`**: è un valore dimostrativo. La documentazione dice che il modello rilasciato ha avuto 280 epoch.

Il **peso perceptuale** confronta le immagini secondo caratteristiche "visive" e non solo punto per punto. Il **peso avversariale** usa un discriminatore, cioè una rete che cerca di distinguere immagini vere da finte.

### 7.6 I file di ambiente: `environment_*.json`

Un file di ambiente dice **quali pesi** usare e **dove** trovarli. La coppia "rete + ambiente" deve essere della stessa famiglia, altrimenti il programma non trova i file giusti.

| File | Contenuto principale | Note |
|---|---|---|
| `environment_rflow-ct.json` | AE v1, DM rflow-ct, ControlNet rflow-ct, AE e DM delle maschere v2, zip delle maschere, JSON dei candidati (4000), dimensioni anatomiche, dizionari | Pipeline TC completa |
| `environment_ddpm-ct.json` | Come sopra, con 3000 candidati | Ha un segmento `configs/` in più in una chiave che non viene mai letta |
| `environment_rflow-mr.json` | AE v2, DM rflow-mr, `trained_controlnet_path = null`, percorsi delle maschere `null` | **Rompe `inference.py`, riga 105** (vedi [sezione 11](#11-problemi-noti)) |
| `environment_maisi_diff_model_rflow-ct.json` | AE v1, `diff_unet_3d_rflow-ct.pt`, `modality_mapping_path` | `json_data_list = ./dataset.json`, file assente |
| `environment_maisi_diff_model_rflow-mr.json` | AE v2, `diff_unet_3d_rflow-mr.pt` | |
| `environment_maisi_diff_model_rflow-mr-brain.json` | AE v1 (riusa l'autoencoder TC), `diff_unet_3d_rflow-mr-brain_v1.pt` | Coerente con il download |
| `environment_maisi_diff_model_ddpm-ct.json` | `diff_unet_3d_ddpm-ct.pt`, **senza** `modality_mapping_path` | Coerente: il DDPM non ha la modalità |
| `environment_maisi_controlnet_train_rflow-ct.json` | AE v1, DM rflow-ct, `existing_ckpt_filepath`, `exp_name = controlnet_kits_finetune`, sottoinsieme KiTS | Esempio di fine-tuning |
| `environment_maisi_controlnet_train_ddpm-ct.json` | Simile, per il DDPM | |
| `environment_maisi_controlnet_train_rflow-mr.json` | Esempio cerebrale in `temp_work_dir` | Scheletro |
| `environment_maisi_controlnet_train.json` | Chiavi vecchie (`trained_controlnet_path`, `autoencoder_epoch273.pt`) | Il trainer ignora alcune chiavi |

### 7.7 La modalità: `modality_mapping.json`

Ogni tipo di scansione ha un numero, che viene dato al modello come istruzione:

- `unknown` = 0 (**è anche la "classe nulla" per la CFG**)
- TC: `ct` = 1, `ct_wo_contrast` = 2, `ct_contrast` = 3
- Risonanza: `mri` = 8, `mri_t1` = 9, `mri_t2` = 10, `mri_flair` = 11, `mri_pd` = 12, `mri_dwi` = 13, `mri_adc` = 14, `mri_ssfp` = 15, `mri_mra` = 16, `mri_t1c` = 17, `mri_swi` = 20
- Risonanza senza cranio (*skull-stripped*): `mri_t1` = 29, `mri_t2` = 30, `mri_flair` = 31, `mri_swi` = 32, `mri_mra` = 33

**Regole**:
- Numeri da 1 a 7 = TC; da 8 in su = risonanza.
- I codici 4–7 non hanno un nome ma sono trattati come TC.
- I codici 18–19 e 21–28 non esistono.
- Il numero viene usato in tre posti: nei dataloader (traduce il testo in numero), nella UNet e nel ControlNet come `class_labels` (la classe della modalità), e per decidere se l'immagine è TC o risonanza.

Per il tuo progetto: T1 è 9 (o 29 senza cranio), T2 è 10 (o 30), FLAIR è 11 (o 31).

### 7.8 Le statistiche dei tessuti: `image_median_statistics_ct.json`

Sette gruppi (fegato, milza, pancreas, rene, polmone, osso, cervello). Per ogni gruppo: valore minimo e massimo della mediana, percentili 0.5 e 99.5, sigma 6 e 12 sopra e sotto. Sono i valori di riferimento per `quality_check.py`.

Esempio: per il fegato `sigma_6_low = −21.6`, `sigma_6_high = 156.3`, `percentile_0_5 = 9.5`, `percentile_99_5 = 162.0`.

Il file con le dimensioni anatomiche (`all_anatomy_size_conditions.json`) viene **scaricato ma mai letto**. Non è definito nel progetto che cosa significhi la scala tra 0 e 1 delle dimensioni.

---

## 8. Le etichette delle strutture

### 8.1 `label_dict.json`: il vocabolario di 132 strutture

È il dizionario che il codice usa davvero. Esempi:

| Numero | Struttura | Numero | Struttura |
|---|---|---|---|
| 1 | fegato | 23 | tumore del polmone |
| 3 | milza | 24 | tumore del pancreas |
| 4 | pancreas | 26 | tumore del fegato |
| 5 | rene destro | 27 | tumore del colon |
| 6 | aorta | 57 | trachea |
| 12 | stomaco | 62 | colon |
| 13 | duodeno | 63–86 | costole |
| 14 | rene sinistro | 120 | cranio |
| 22 | cervello | 128 | lesione ossea |
| 25 | vasi epatici | 132 | vie aeree |
| 28–32 | lobi polmonari | 200 | corpo |
| 33–56 | vertebre (e S1 = 127) | | |

I numeri **2, 16, 18, 20, 21, 129, 130, 131** sono segnaposto (`dummy1`–`dummy8`): il modello li ha lasciati liberi. Anche 133–199 e 201–255 sono liberi, e servono per aggiungere strutture nuove durante il fine-tuning.

**Attenzione**: la parola `lung` non è una chiave del dizionario. Ci sono solo i nomi dei singoli lobi. Il nome `lung` è accettato solo come condizione in Percorso A.

Il dizionario viene usato in: `LDMSampler` (`sample.py`, riga 126), i controllori (`sample_mask.py`, righe 341–345 e 395–399), `find_masks.py` (riga 107), e le etichette di tentativo in Percorso A (`sample.py`, righe 159–161).

### 8.2 `label_dict_124_to_132.json`: dal modello delle maschere alle 132 strutture

Il modello delle maschere produce numeri da 0 a 124 (125 canali). Questo file li traduce nelle etichette del dizionario principale. Il formato è `{nome: [origine, destinazione]}`:

- fegato `[1, 1]`
- milza `[2, 3]`
- pancreas `[3, 4]`
- rene destro `[4, 5]`
- vie aeree `[123, 132]`
- corpo `[124, 200]`
- sfondo `[0, 0]`

Viene applicato da `remap_labels` (`utils.py`, righe 51–56) in `sample_mask.py` (riga 175). Per questo la skill di immagine dalla maschera avverte di non dare al ControlNet etichette tra 0 e 124 senza traduzione.

### 8.3 `label_dict_124_to_132_ctmr.json`

Identico al precedente, ma il corpo va a **500** invece di 200 (riga 500). **Nessun file Python lo carica.**

### 8.4 `label_dict_ctmr.json`: un vocabolario alternativo

Un altro elenco, con nomi diversi: rene 2, polmone 20, osso 21, cervello 22, retto 18, prostata o utero (deprecato) 16, corpo 500, vie aeree 132, e segnaposto 6, 7, 8 a 129–131. Ha le chiavi `lung` e `bone`.

**Nessun file Python lo carica.** Quindi il "brain tumor" 176 presente qui non serve al codice attuale. Questo chiarisce parzialmente la domanda aperta del tuo `RIASSUNTO_PROGETTO.md`: il codice usa `label_dict.json` (132 classi) per il modello delle maschere.

### 8.5 Le etichette scritte nel codice

- **Etichette tumorali** `[23, 24, 26, 27, 128]`: in `find_masks.py` (riga 126), `sample_mask.py` (riga 181), `utils.py` (riga 594).
- **`tumor_organ_dict`** in `utils.py` (riga 594): 23→28, 24→4, 26→1, 27→62, 128→200. Dice quale organo "ospita" ciascun tumore.
- **Vasi epatici 25** → fegato 1 (`utils.py`, riga 690). **Vie aeree 132** restano (riga 691).
- **Regioni del corpo** (`utils.py`, righe 399–407): testa, torace, addome, bacino, come visto nella sezione 6.7.
- **Dimensioni per il Percorso A** (`sample_mask.py`, righe 57–72): come visto nella sezione 6.5.

---

## 9. Il percorso di un'immagine, passo per passo

### 9.1 Percorso A: mappa creata da zero (diffusione)

**Quando si usa**: se `controllable_anatomy_size` o `controllable_demographics` non sono vuoti (`sample.py`, riga 236).

1. **Preparare la condizione** (`sample.py`, righe 435–480): si costruisce un vettore di 19 numeri. Gli slot 0–13 valgono −1 di default; l'organo o il tumore richiesto riceve la sua dimensione (righe 460–465). Gli slot 14–18 sono età, sesso (maschio = 1, femmina = 0), peso, BMI e altezza, divisi per i valori massimi e limitati tra 0 e 1 (righe 467–476).
2. **Generare la maschera con il modello delle maschere** (`sample_mask.py`, righe 78–195): parte da un rumore di forma `[1, 4, 64, 64, 64]`. Il RFlow fa 100 passi. La CFG confronta la predizione con la condizione e quella con il vettore tutto a −1, con peso 2.0.
3. **Decodificare**: con la finestra scorrevole, usando `ReconModel`. Risultato: 125 canali.
4. **Scegliere la classe per ogni punto**: softmax (trasforma i valori in probabilità che sommano a 1), poi argmax (sceglie la più probabile), poi la rimappatura con `label_dict_124_to_132.json`.
5. **Pulire la maschera** (`utils.py`, righe 568–695): sistema il corpo, pulisce i tumori, elimina componenti di organi troppo piccole o isolate, applica le regole dell'intestino, riempie i buchi per chiusura, ridisegna il tumore target, sistema vasi e vie aeree.
6. **Riprovare se serve** (`sample.py`, righe 282–302): se manca un'etichetta richiesta, rigenera fino a cinque volte; poi va avanti con un avviso.
7. **Adattare la dimensione** (`sample.py`, righe 487–498 e 537–580): ricampiona con il metodo "nearest" alla spaziatura richiesta, poi aggiunge o taglia il bordo fino alla dimensione finale. Non controlla che le etichette target siano presenti (riga 499).
8. **Calcolare le regioni del corpo** dalla maschera (`sample.py`, riga 501; `utils.py`, righe 389–425).
9. **Creare l'immagine** (vedi [9.3](#93-creare-limmagine-dalla-mappa)).
10. **Salvare la maschera completa** con le 132 classi. Il filtro si applica solo al Percorso B.

### 9.2 Percorso B: mappa presa dall'archivio, con augmentation

**Quando si usa**: se `controllable_anatomy_size` e `controllable_demographics` sono entrambi vuoti.

1. **Cercare una corrispondenza esatta** con `find_masks(..., True)` (`sample.py`, righe 249–257).
2. **Se non bastano, cercare quelle più simili** (`sample.py`, righe 258–264; `find_closest_masks` alle righe 607–667). Ordina i candidati per FOV, dimensione e spaziatura. Scarta quelli che richiederebbero più di 128 voxel di ricampionamento. Tiene i primi `max(2·num_img, 5)` dopo il controllo degli organi.
3. **Scegliere una maschera** (`select_mask`, righe 371–388): mescola e cicla. L'augmentation è sempre attiva.
4. **Leggere le informazioni** della maschera (`read_mask_information`, righe 582–605).
5. **Ricampionare** se la maschera è stata presa dalla ricerca approssimata (righe 313–314).
6. **Applicare l'augmentation** (`augmentation.py`, righe 436–481): in base all'etichetta presente. 128 → osso; 26 → fegato; 23 → polmone; 24 → pancreas; 27 → colon; 401–403 → cervello (BraTS). Se non c'è nessun tumore, zoom del corpo tra 0.99 e 1.01.
7. **Creare l'immagine** (vedi 9.3).
8. **Salvare l'etichetta filtrata** sulle strutture richieste (`sample.py`, righe 356–357; `sample_mask.py`, righe 198–215).

### 9.3 Creare l'immagine dalla mappa

Il cuore è `run_controlnet_conditioned_image_dm` (`utils_infer.py`, righe 118–341). Il contenitore che lo chiama è `ldm_conditional_sample_one_image_from_mask` (`infer_image_from_mask.py`, righe 65–165).

1. **Preparare la condizione**: la maschera intera viene trasformata in 8 strati binari (`utils.py`, righe 173–188). Quindi il ControlNet riceve un tensore di forma `[B, 8, X, Y, Z]`. Per questo le etichette arrivano fino a 255.
2. **Preparare la versione "senza tumore"** (solo se la CFG è maggiore di zero): si rimuovono i tumori (`augmentation.py`, righe 484–520) e si binarizza (`infer_image_from_mask.py`, righe 127–136).
3. **Creare il rumore** di forma `[1, 4, X/4, Y/4, Z/4]` (`utils_infer.py`, riga 211) e i passi di tempo del RFlow (righe 213–217).
4. **Per ogni passo** (righe 239–294):
   - il ControlNet riceve `(x, t, condizione)` più le classi della modalità se il modello le ha; con la CFG il batch raddoppia, e la metà senza condizione ha le classi e la maschera azzerate;
   - la UNet riceve la spaziatura moltiplicata per 100, le regioni se presenti, le classi, e i residui del ControlNet sommati ai blocchi di discesa e al centro;
   - la CFG combina le due predizioni: `non_condizionata + s · (condizionata − non_condizionata)` (righe 288–289);
   - lo scheduler fa il passo successivo (righe 291–294).
5. **Decomprimere** con la finestra scorrevole e `ReconModel` (righe 310–322).
6. **Riportare alla scala dei tessuti** (righe 326–335): per la TC il valore viene limitato tra 0 e 1, poi portato tra −1000 e 1000 HU. Per la risonanza viene limitato in basso e portato tra 0 e 1000.
7. **Azzerare lo sfondo**: `crop_img_body_mask` porta a sfondo i punti con maschera 0 (`infer_image_from_mask.py`, riga 164; righe 173–185).
8. **Controllo di qualità** (solo TC, Percorsi A e B): viene registrato, ma con un solo campione non fa rigenerare (vedi [sezione 11](#11-problemi-noti)).

### 9.4 Come si addestra il modello

Il training ha quattro parti, da fare in quest'ordine.

**(a) Il VAE dell'immagine.** Non esiste un file `train_vae.py`. Il training è nel notebook `train_vae_tutorial.ipynb`, che ho solo cercato con grep, non letto per intero.
- Dati: trasformazioni `VAE_Transform` (`transforms.py`, righe 228–310); ritagli casuali di 64³; validazione con finestra `[96, 96, 64]`.
- Perdita del generatore: L1 + 1e-7 · KL + 0.3 · perdita perceptuale + 0.1 · perdita avversariale. Il discriminatore è un `PatchDiscriminator`, cioè guarda piccole porzioni dell'immagine.
- Ottimizzatore Adam sia per il generatore sia per il discriminatore, learning rate 1e-4, con riscaldamento iniziale (*warmup*).
- Output: `model_dir/autoencoder.pt`, `discriminator.pt` e, per epoch, `autoencoder_epoch{N}.pt`.

**(b) Gli embedding** (`diff_model_create_training_data.py`). I volumi vengono portati a multipli di 128 (righe 77–90), codificati con il VAE a blocchi di 320×320×160 con sovrapposizione 0.4 (righe 173–186), e salvati come `<nome>_emb.nii.gz` (righe 190–192). La versione compressa è 4 volte più piccola per asse.

**(c) Il modello principale** (`diff_model_train.py`). Riceve gli embedding e il file accompagnatore. Le immagini vengono moltiplicate per il fattore di scala (1 / deviazione standard del primo batch, righe 186–188). Con probabilità 10% la modalità viene azzerata (righe 36–66). Il bersaglio per RFlow è `immagine − rumore`. Perdita L1, Adam, decadimento polinomiale, AMP. Un checkpoint per epoch, con il fattore di scala salvato dentro.

**(d) Il ControlNet** (`train_controlnet.py`). Il modello principale è congelato (righe 317–318). Il ControlNet parte dai pesi del modello principale (riga 307) oppure da un file precedente (`existing_ckpt_filepath`). Perdita L1 sull'output del modello principale, con eventuale peso sulle etichette e perdita contrastiva (righe 454–481). AdamW con learning rate 1e-5 e decadimento polinomiale. **Nessuna validazione.** Salva `_current.pt` e `_best.pt`.

### 9.5 Creare solo immagini, senza maschera

`diff_model_infer.py` crea immagini senza ControlNet. Si parte dal rumore, il modello principale fa il denoising con la CFG sulla modalità (classe 0 come non condizionata), poi l'autoencoder decomprime con finestra fissa `[80, 80, 80]` e sovrapposizione 0.4. L'uscita è in formato intero a 16 bit.

Il risultato **non ha una mappa**: non puoi controllare dove stanno le strutture.

---

## 10. Come si usa in pratica

### 10.1 Preparazione

```bash
pip install -r requirements.txt
# PyTorch con CUDA: https://pytorch.org/get-started/locally/
```

Serve Python 3.11 o superiore, e una scheda grafica NVIDIA con almeno 16 GB di memoria per le configurazioni piccole. Per i volumi 512×512×768 servono 40 GB per creare e 80 GB per addestrare, come dice il paper.

### 10.2 Scaricare i pesi

```bash
python -m scripts.download_model_data --version rflow-ct --root_dir "./" --model_only
python -m scripts.download_model_data --version rflow-mr --root_dir "./" --model_only
python -m scripts.download_model_data --version rflow-mr-brain --root_dir "./" --model_only
```

`--model_only` scarica solo i pesi. I pesi finiscono nella cartella `models/`. Senza `--model_only` scarica anche l'archivio delle maschere: la dimensione non l'ho verificata.

### 10.3 Creare una coppia immagine + maschera (TC)

```bash
python -m scripts.inference -t ./configs/config_network_rflow.json -i ./configs/config_infer.json -e ./configs/environment_rflow-ct.json --random-seed 0 --version rflow-ct
```

### 10.4 Creare un'immagine dalla tua maschera

```bash
python -m scripts.infer_image_from_mask --mask percorso/della/tua_maschera.nii.gz -t ./configs/config_network_rflow.json -e ./configs/environment_rflow-ct.json -i ./configs/config_infer.json
```

Questo comando usa il modello TC. **Per il cervello serve un altro modello, e questo comando non è pensato per la risonanza.**

### 10.5 Addestrare il ControlNet

Esempio tratto dalla documentazione (da verificare):

```bash
python -m scripts.train_controlnet -t ./configs/config_network_rflow.json -c ./configs/config_maisi_controlnet_train_rflow-ct.json -e ./configs/environment_maisi_controlnet_train_rflow-ct.json -g 1
```

**Attenzione**: la documentazione in alcuni punti indica il file di configurazione sbagliato (quello della diffusione invece di quello del ControlNet). Controlla sempre che sia il file del ControlNet.

### 10.6 Controllare lo stile del codice

```bash
pre-commit run --all-files
```

Controlla lo stile con `ruff`. Non ci sono test che verificano che il programma funzioni.

---

## 11. Problemi noti

Ordinati per importanza. **"Verificato"** significa che l'ho letto nel codice. **"Da verificare"** significa che va provato sul tuo computer.

1. **Un file di ambiente fa crashare il programma.** `environment_rflow-mr.json` ha tre chiavi vuote (`null`), e `inference.py` alla riga 105 non le gestisce. (Verificato leggendo il codice.)

2. **Il seme casuale è una stringa.** `--random-seed` non è dichiarato come numero (`inference.py`, righe 55–60): il valore arriva come testo e viene passato a `set_determinism` e a `LDMSampler`. Non so se numpy o torch lo accettano. Se il programma non parte, questo è il primo sospetto. (Da verificare.)

3. **Un nome non più valido in NumPy.** `utils.py` usa `np.long` alle righe 361, 382, 413, 416, 418, 421, 590, 615–620, 631, 636, 659–660. Nelle versioni moderne di NumPy non esiste più. `requirements.txt` permette `numpy>=1.24.0`. Se viene davvero usato, il programma si blocca: fallirebbero il Percorso A (riga 590), la ricerca approssimata e `build_conditioning_tensors`. (Da verificare.)

4. **La scheda NVIDIA è obbligatoria.** Il nome `cuda` è scritto direttamente in più punti: `augmentation.py` righe 104, 152, 208, 260, 314; `sample_mask.py` riga 133; `inference.py` riga 172. Il Percorso B richiede una GPU in ogni caso. (Verificato.)

5. **Il ControlNet non valuta su dati nuovi.** Il caricatore di validazione viene buttato via (`train_controlnet.py`, riga 337). Non sai se il modello migliora davvero su casi che non ha mai visto. (Verificato.)

6. **Manca un file accompagnatore.** `diff_model_train.py` riga 450 cerca `<embedding>.nii.gz.json`, ma `diff_model_create_training_data.py` scrive solo il NIfTI (righe 190–192). Non so se il notebook di addestramento lo crea. (Verificato nei file letti; da verificare nel notebook.)

7. **Il controllo di qualità non fa rigenerare con un solo campione.** `sample.py`, riga 335: la condizione per rigenerare non scatta quando `num_img = 1`, quindi la coppia viene salvata anche se non passa il controllo. La documentazione dice il contrario. (Verificato.)

8. **Il tentativo di trovare le strutture in Percorso A ha un limite.** Dopo cinque tentativi (`sample.py`, righe 282–302), il programma va avanti con un avviso. La coppia viene salvata anche se manca un organo richiesto. (Verificato.)

9. **Percorsi di default verso file che non esistono.** `inference.py` (righe 31–42), `diff_model_train.py` (righe 521–536), `train_controlnet.py` (righe 541–556), `diff_model_create_training_data.py` (righe 268–283), `find_masks.py` (righe 61–62). Funzionano solo se passi i percorsi giusti, come fa la documentazione. (Verificato.)

10. **Alcune funzioni possono non finire mai.** L'augmentation per fegato, polmone, pancreas non ha un limite di tentativi (`augmentation.py`, righe 149, 205, 257). Se la maschera è difficile, il programma può sembrare bloccato. Il colon si ferma a 40 e BraTS a 1000. (Verificato.)

11. **Preset con un numero di passi diverso.** `24g_256x256x256` e `80g_512x512x512` usano 1000 passi, gli altri 30. Potrebbe essere un errore di copia. (Da verificare.)

12. **Codice inutilizzato e doppioni.**
    - `run_torchrun` (`diff_model_setting.py`, righe 98–139) non è mai chiamato.
    - `utils_plot.py` è usato solo dai notebook.
    - In `utils.py` riga 609, il valore 10 è ripetuto nella lista `oran_list`.
    - In `sample.py` riga 186, `label_int_dict["lung"]` duplica il 31.
    - `mask_generation_scale_factor` nella configurazione non è letto (`config_network_rflow.json`, riga 139).
    - `utils.py` righe 686–689 stampa un messaggio a ogni chiamata.

13. **Un commento sbagliato.** In `diff_model_create_training_data.py` (righe 253–256) un commento dice che `modality` può non essere definita. In realtà è definita a riga 245.

14. **Il dizionario per il tuo tumore.** Non è chiaro quale dizionario usare per il tumore cerebrale. Vedi [sezione 13](#13-che-cosa-significa-per-il-tuo-progetto).

15. **Valori fissi nella funzione `add_body_envelope`** (`utils.py`, righe 428–565): soglia −800 HU per l'aria, `table_frac_thresh = 0.05` per il lettino, device di default `cuda:0`. Sono spiegati solo nel commento della funzione.

16. **Il significato della dimensione non è chiaro.** Non è definito nel progetto che cosa rappresenti il numero di `controllable_anatomy_size`. Le skill avvertono che il tumore del fegato (slot 11) è riconosciuto male, cioè con poca sensibilità.

17. **La corrispondenza canali-etichette della maschera.** I 125 canali (`config_network_rflow.json`, riga 95) e le rimappature 0–124 combaciano, ma la corrispondenza completa canale per canale non l'ho confermata oltre il file di rimappatura.

---

## 12. Dove la documentazione non corrisponde al codice

Non è un giudizio: è un elenco di cose da controllare prima di fidarsi.

| Dove | Cosa dice | Cosa fa il codice |
|---|---|---|
| `docs/inference.md` (righe 200, 219) e `skills/infer_image-only.md` (righe 72, 176–177) | Il CFG della risonanza è 10 | È 15 per `rflow-mr` e 2 per `rflow-mr-brain` |
| `skills/infer_image-from-mask.md` (riga 66) | La CLI scrive due file: immagine ed etichetta | Scrive solo l'immagine (`infer_image_from_mask.py`, righe 503–520) |
| `skills/infer_mask-only_via_real_aug.md` (riga 82) | Le augmentation si fermano dopo 1000 tentativi | Solo BraTS ha questo limite; fegato, polmone, pancreas, osso non si fermano mai; il colon si ferma a 40 |
| `skills/infer_mask-only_via_real_aug.md` (righe 31–32) | Il ripiego ignora il filtro sulle strutture | Lo rispetta (`sample.py`, righe 621–629 e 573–579) |
| `docs/inference.md` (riga 210) | Il controllo di qualità fallito riprova fino a 2 volte | Con un solo campione salva comunque (`sample.py`, righe 335–341) |
| `skills/train_controlnet_image-from-mask.md` (riga 68) | La rete DDPM ha `num_class_embeds = 128` | Non ce l'ha (`config_network_ddpm.json`) |
| `docs/training.md` (riga 92) | Comando del ControlNet con il file di configurazione sbagliato | Il comando fallirebbe in `train_controlnet.py`, riga 260 |
| `docs/performance.md` (riga 9) | Finestra `>= [64, 64, 32]` per la configurazione 16g 256×256×128 | Nella configurazione è `[96, 96, 96]` |
| `skills/download-models.md` | Opzioni `--token` e `--overwrite` | Non esistono (`download_model_data.py`, righe 191–206) |
| `scripts/sample.py` (righe 19–20) e `scripts/infer_image_from_mask.py` (riga 40) | Rimandano ai file `skills/mask-generation.md` e `skills/image-from-mask.md` | I file reali si chiamano `infer_mask-only*.md` e `infer_image-from-mask.md` |

Nota: le skill sulla TC (`infer_mask-only_*`, `infer_mask-image-paired`, `download-models`) sono in gran parte corrette. Le discrepanze riguardano soprattutto i dettagli operativi.

---

## 13. Che cosa significa per il tuo progetto

Il tuo obiettivo, dal `RIASSUNTO_PROGETTO.md`: creare mappe con tumore cerebrale partendo dai tuoi 30 casi, e in seguito creare risonanze dalla mappa con un ControlNet.

### 13.1 Quello che c'è già

- Un modello per la **risonanza del cervello**: `rflow-mr-brain`, con le sequenze T1, T2, FLAIR e SWI.
- Impostazioni di dimensione e spaziatura coerenti con il download (`environment_maisi_diff_model_rflow-mr-brain.json` e `config_maisi_diff_model_rflow-mr-brain.json`).
- Numeri già pronti per le sequenze: T1 = 9 (o 29 senza cranio), T2 = 10 (o 30), FLAIR = 11 (o 31).
- Un'augmentation pensata per i tumori cerebrali (`augmentation_tumor_only`, con le etichette BraTS 401–403).

### 13.2 Quello che manca

1. **Il ControlNet per il cervello** non risulta scaricabile con il codice attuale. Va verificato su Hugging Face.
2. **Un modello per creare mappe cerebrali con tumore.** Il modello delle maschere esistente è addestrato sul corpo intero in TC, non sul cervello. Per le mappe cerebrali serve addestrarlo, oppure scegliere un'altra strada.
3. **Un'etichetta per il tumore cerebrale.** Nel dizionario principale c'è il cervello (numero 22), ma non il tumore. Il numero 176 del file alternativo non è letto dal codice.

### 13.3 La scelta dell'etichetta del tumore

Il `RIASSUNTO_PROGETTO.md` aveva deciso di usare il numero 176. Il codice dice che non si può fare, perché il modello legge solo il dizionario a 132 numeri. Le strade sono due:

- **Usare un numero libero** (tra 133 e 199) per una nuova struttura "tumore cerebrale". Richiede di aggiungerla alla tabella di traduzione `label_dict_124_to_132.json` e di riaddestrare il modello delle maschere, perché il numero di canali in uscita cambia. È la strada corretta ma più lunga.
- **Riusare una struttura vicina**, per esempio la lesione ossea (128). È più semplice, ma semanticamente sbagliato.

È una tua decisione. Va scritta da qualche parte, perché cambia il lavoro di addestramento.

### 13.4 Licenze

- `NV-Generate-MR-Brain` e `rflow-mr-brain`: uso libero, anche commerciale.
- `rflow-mr`: solo ricerca.
- Dati MR-RATE: solo ricerca, licenza CC-BY-NC.

Se il progetto è di ricerca, sei a posto. Se in futuro vuoi usarlo commercialmente, il modello cerebrale sì, ma i dati MR-RATE no.

### 13.5 Un percorso ragionevole

1. **Verificare il modello del cervello**: scaricare `rflow-mr-brain`, controllare che i file siano quelli attesi, cercare su Hugging Face se esiste un ControlNet per il cervello.
2. **Decidere l'etichetta del tumore** (vedi 13.3) e scrivere la tabella che collega le 183 strutture dei tuoi dati alle strutture del modello. È il punto più rischioso di tutto il progetto.
3. **Preparare i dati**: maschere in formato NIfTI, con le etichette come numeri interi, nella spaziatura giusta. Poi creare gli embedding con `diff_model_create_training_data.py`, e scrivere a mano il file accompagnatore JSON richiesto da `diff_model_train.py`.
4. **Provare la pipeline su un solo caso** prima di addestrare: la creazione di un'immagine dalla tua maschera deve funzionare.
5. **Addestrare** con i 24 casi di training e controllare con i 6 di validazione. Il ControlNet attuale non valuta su dati nuovi, quindi la valutazione va scritta.
6. **Valutare**: Dice sul tumore sui casi di validazione, e controllare che le maschere generate non siano copie dei tuoi casi (distanza dai casi di training). Il FID da solo non basta.

### 13.6 Due lezioni dal paper per il tuo caso

- La **perdita contrastiva per regione** aiuta a far comparire tumori piccoli e poco contrastati, come il tuo. Il codice la implementa, ma va attivata nella configurazione.
- Con **30 casi** il rischio principale è che il modello **copi** i tuoi casi invece di crearne di nuovi. Il paper usa circa 107.000 volumi, quindi la situazione è molto diversa. Il controllo di distanza dai casi di training serve proprio a questo.

---

## 14. Glossario

- **Autoencoder (VAE)**: comprime un'immagine in una versione piccola e la ricostruisce.
- **AdaGN**: un modo di regolare un blocco del modello con due numeri (scala e spostamento) calcolati dalla condizione.
- **Augmentation**: variazione artificiale dei dati (ruotare, deformare, spostare) per averne di più.
- **Batch**: quanti esempi vengono elaborati insieme.
- **ControlNet**: il componente che legge la maschera e guida il modello principale.
- **CFG (classifier-free guidance)**: combina la predizione con e senza condizione, per far seguire di più la condizione.
- **Dice**: quanto due maschere coincidono, da 0 a 1.
- **Diffusione**: creare un'immagine partendo dal rumore, passo dopo passo.
- **Embedding**: una versione compressa dell'immagine, pronta per l'addestramento.
- **Epoch**: un giro completo sui dati di addestramento.
- **FID**: quanto le immagini sembrano reali, più basso è meglio.
- **FOV (campo di vista)**: la dimensione reale del volume.
- **Hounsfield (HU)**: la scala dei valori della TC.
- **Learning rate**: quanto il modello cambia a ogni correzione.
- **Maschera**: la mappa delle strutture.
- **Modalità**: il tipo di scansione (TC, T1, T2, FLAIR…), codificato con un numero.
- **NIfTI**: il formato standard dei volumi medici.
- **Percorso A / B**: mappa creata da zero (A) oppure presa dall'archivio (B).
- **Rectified flow (RFlow)**: variante della diffusione con percorsi più dritti, quindi meno passi.
- **Scale factor (fattore di scala)**: un numero che porta i valori compressi a una scala comoda per il modello.
- **Scheduler**: il programma che decide il rumore passo per passo.
- **Seme casuale (seed)**: il numero che fissa il caso, per riprodurre un risultato.
- **Spaziatura (spacing)**: quanti millimetri misura un voxel.
- **Spazio latente**: la versione compressa dell'immagine.
- **Tensore**: una griglia di numeri con più dimensioni.
- **Voxel**: un cubetto del volume, cioè un pixel 3D.
