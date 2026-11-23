# Guida allo studio di NV-Generate-CTMR e MAISI-v2

Questo documento spiega il progetto in modo semplice: che cos'è, come funziona, a cosa serve ogni file del codice e come useresti il programma per il tuo lavoro.

**Da dove vengono le informazioni**
- Il **paper** MAISI-v2 (arXiv 2508.05772, Zhao et al., NVIDIA).
- Il **blog** di NVIDIA del 22 maggio 2026, letto come riassunto da una pagina web.
- Il **codice**, letto riga per riga ma **non eseguito**: non ho una GPU né i pesi dei modelli. Quello che è marcato "da verificare" va provato sul tuo computer prima di fidarsene.

**Indice**
1. [Di che cosa parla il progetto](#1-di-che-cosa-parla-il-progetto)
2. [Parole chiave](#2-parole-chiave)
3. [Il paper, spiegato semplice](#3-il-paper-spiegato-semplice)
4. [Il blog e i modelli disponibili](#4-il-blog-e-i-modelli-disponibili)
5. [Come è fatto il codice](#5-come-è-fatto-il-codice)
6. [A cosa serve ogni file](#6-a-cosa-serve-ogni-file)
7. [I file di configurazione](#7-i-file-di-configurazione)
8. [Le etichette delle strutture](#8-le-etichette-delle-strutture)
9. [Il percorso di un'immagine, passo per passo](#9-il-percorso-di-unimmagine-passo-per-passo)
10. [Come si usa in pratica](#10-come-si-usa-in-pratica)
11. [Problemi noti](#11-problemi-noti)
12. [Dove la documentazione non corrisponde al codice](#12-dove-la-documentazione-non-corrisponde-al-codice)
13. [Che cosa significa per il tuo progetto](#13-che-cosa-significa-per-il-tuo-progetto)
14. [Piccolo glossario](#14-piccolo-glossario)

---

## 1. Di che cosa parla il progetto

Il progetto crea **immagini mediche finte ma realistiche** (per ora TC, cioè CT, e in parte risonanza, cioè MRI). Ogni immagine viene creata insieme a una **mappa delle strutture**: per ogni punto del volume il programma sa se è fegato, osso, tumore, e così via.

Perché serve? Per addestrare altri programmi che riconoscono le strutture (chiamati modelli di segmentazione), senza usare solo i dati reali, che sono pochi e protetti dalla privacy.

Ci sono due modi di usarlo:

1. **Creare tutto da zero**: il programma inventa una mappa delle strutture, poi crea l'immagine che corrisponde.
2. **Partire da una mappa che hai già**: per esempio la tua segmentazione di un tumore. Il programma crea l'immagine che corrisponde a quella mappa. È il modo più vicino al tuo lavoro.

---

## 2. Parole chiave

- **Modello di diffusione**: impara a creare un'immagine partendo dal rumore, come una TV che si "mette a fuoco" passo dopo passo. Molti passi = più lento.
- **Rectified flow**: una variante più veloce, che fa il percorso dal rumore all'immagine in modo più diretto, quindi con meno passi.
- **Spazio latente**: un modo compresso di rappresentare l'immagine. Si lavora sulla versione compressa, che è più piccola e veloce, e poi si decomprime.
- **VAE (autoencoder)**: la parte che comprime e decomprime. Un "traduttore" tra immagine e versione compressa.
- **ControlNet**: un componente aggiuntivo che guarda la mappa delle strutture e dice al modello principale "metti il tumore qui". Il modello principale resta fermo.
- **Maschera**: la mappa delle strutture. Ogni punto ha un numero: 0 è lo sfondo, 1 il fegato, 23 il tumore del polmone, e così via.
- **Etichetta**: quel numero. Il modello conosce solo una lista fissa di etichette, quindi cambiare numero vuol dire cambiare struttura.
- **Spaziatura**: quanti millimetri misura un punto (un voxel) del volume.
- **FOV (campo di vista)**: la dimensione reale del volume, cioè numero di punti × spaziatura. Se il FOV è strano, il risultato peggiora.
- **FID**: un punteggio di quanto le immagini sembrano reali. Più basso è meglio. Non dice nulla sulla correttezza medica.
- **Dice**: un punteggio, da 0 a 1, di quanto due maschere coincidono. Più alto è meglio.

---

## 3. Il paper, spiegato semplice

### 3.1 Il problema
I modelli precedenti avevano tre difetti:
1. funzionavano solo per un organo o per una risoluzione precisa;
2. erano lenti (centinaia di passi, decine di minuti per un volume grande);
3. non seguivano bene la mappa: il tumore spesso non compariva dove doveva.

### 3.2 Che cosa propone MAISI-v2
- Usa il **rectified flow**: stesso risultato con molti meno passi. Nel paper circa **33 volte più veloce** nella parte di creazione dell'immagine: 6 secondi invece di 198 secondi.
- Aggiunge una nuova regola di addestramento, la **perdita contrastiva per regione**. Idea: creare due immagini con lo stesso punto di partenza, una con il tumore e una con il tumore sostituito dal tessuto normale. Le due devono essere **diverse dentro il tumore** e **uguali fuori**. Così il modello impara a far comparire il tumore dove c'è scritto.
- Il codice e i pesi sono pubblicati.

### 3.3 Come è stato addestrato
- Dati: circa **11.000 TC** da fonti pubbliche e due dataset interni. Dopo la preparazione, circa **107.000 volumi**. Tutto **TC**, non risonanza.
- Addestramento in tre fasi: prima su immagini piccole (per stabilizzare), poi su immagini grandi, poi una fase finale che bilancia i dataset diversi.
- Il tempo: circa tre settimane su 64 schede grafiche molto potenti.
- Il controllo del tumore (ControlNet) è stato addestrato a parte, in circa due giorni su 8 schede.

### 3.4 Che cosa dicono i risultati
- **Velocità**: creare un volume 512×512×512 richiede circa 6 secondi (più 15 per la decompressione), contro 198 del metodo precedente. La qualità è simile.
- **Utilità per addestrare altri modelli**: usando le immagini sintetiche come dati in più, un modello di segmentazione dei tumori migliora. Il miglioramento, sul test con dati reali, è di circa +3% per il fegato, +7% per il polmone, +6% per il pancreas, +4% per il colon, +3% per le ossa. Il paper dice che è statisticamente significativo in 4 casi su 5. Attenzione: nella copia PDF che ho letto alcuni simboli di significatività si sono persi, quindi controlla la tabella originale prima di citarla.
- **Un risultato importante**: senza la regola contrastiva, sul colon il miglioramento era **negativo** (−4.7%). Con la regola è diventato positivo. Quindi la regola conta davvero.

### 3.5 Limiti dichiarati dagli autori
- Solo TC, per ora.
- Solo la segmentazione come test: non hanno testato altri compiti.
- Servono schede molto grandi per i volumi più grossi (40 GB per creare, 80 GB per addestrare).

Un limite che gli autori non dicono: un punteggio FID basso non garantisce che il tumore sia nel posto giusto dal punto di vista medico. Per questo bisogna controllare anche il Dice sul tumore.

### 3.6 Attenzione al tumore con rectified flow
Il paper ammette che, con il rectified flow, il tumore a volte **segue meno bene la mappa** rispetto al metodo precedente, soprattutto nel colon. È un punto da tenere d'occhio: anche il tuo tumore è piccolo e poco contrastato.

---

## 4. Il blog e i modelli disponibili

Il blog di NVIDIA presenta il progetto come un insieme completo, aperto a tutti. Dice che è il primo programma aperto che gestisce in un solo modello volumi di misure diverse e tutto il corpo. Sono affermazioni dell'azienda, da prendere con cautela.

Modelli disponibili:

| Modello | Parte del corpo | Quando | Uso | Licenza |
|---|---|---|---|---|
| `ddpm-ct` | Tutto il corpo, TC | ago 2024 | Solo immagine e coppia immagine/maschera | Libera anche commerciale |
| `rflow-ct` | Tutto il corpo, TC | mar 2025 | Solo immagine e coppia immagine/maschera | Libera anche commerciale |
| `rflow-mr` | Cervello, prostata, addome, seno | ott 2025 | Solo immagine | **Solo ricerca** |
| `NV-Generate-MR-Brain` | Cervello intero o senza cranio | mar 2026 | Immagine; passaggio da una sequenza all'altra | Libera anche commerciale |

Il modello per il **cervello** è quello che ti interessa. Crea risonanze di tipo T1, T2, FLAIR e SWI. Il blog dice che ha anche un componente ControlNet per le strutture e per passare da una sequenza all'altra.

**Attenzione alla licenza**: `rflow-mr` è solo per ricerca. I dati di addestramento del cervello (MR-RATE) sono per ricerca, con licenza CC-BY-NC; per uso commerciale serve una licenza separata.

**Attenzione al ControlNet del cervello**: lo script di download scarica per il cervello solo l'autoencoder e il modello principale, non il ControlNet. Quindi il ControlNet per il cervello dichiarato dal blog non risulta scaricabile con il codice attuale. Da verificare su Hugging Face.

---

## 5. Come è fatto il codice

### 5.1 Le cartelle

- **`scripts/`**: i programmi veri e propri, scritti in Python. 21 file.
- **`configs/`**: file di testo (JSON) che dicono al programma *come* lavorare: quali modelli, quale risoluzione, quali nomi per le strutture. Non sono codice, sono impostazioni.
- **`docs/`** e **`skills/`**: spiegazioni scritte per le operazioni comuni. Sono documenti, non programmi.
- **Notebook (`.ipynb`)**: tutorial interattivi. Sono pesanti perché contengono anche le immagini dei risultati.
- **`.github/`**: un controllo automatico che verifica lo stile del codice. **Non ci sono test** che controllano che il programma funzioni.

### 5.2 Come si lancia un programma

Dalla cartella principale del progetto, si lancia così:

```bash
python -m scripts.nome_del_file
```

Esempio: `python -m scripts.inference` lancia il file `scripts/inference.py`. Il programma non si installa come pacchetto: si lancia direttamente.

### 5.3 Il ruolo delle librerie esterne

Gran parte dei "mattoni" dei modelli (le reti neurali) **non si trovano nel progetto**. Arrivano dalla libreria **MONAI**, che è una libreria per l'imaging medico. Quando nel codice leggi `AutoencoderKlMaisi` o `DiffusionModelUNetMaisi`, la definizione è in MONAI, non qui.

### 5.4 Come si collegano i file

In parole semplici:
- **`inference.py`** è il programma principale: lo lanci tu, e lui chiama gli altri.
- **`sample.py`** è il "direttore d'orchestra": decide se creare la mappa da zero o prenderla da un archivio, e gestisce i controlli.
- **`sample_mask.py`** crea la mappa delle strutture da zero.
- **`infer_image_from_mask.py`** crea l'immagine partendo da una mappa, per esempio la tua.
- **`utils_infer.py`** contiene il cuore della creazione dell'immagine, usato da più programmi.
- **`utils.py`** è il file con gli strumenti generali: pulizia delle mappe, caricamento dei dati, funzioni di supporto.

Il file più grande e centrale è `utils.py`, con quasi 800 righe.

---

## 6. A cosa serve ogni file

Per ogni file: a che cosa serve, cosa fa, come lo lanci e che cosa legge e scrive.

### 6.1 `inference.py`: crea una coppia immagine + mappa (TC)

**A che cosa serve.** È il programma principale per creare una coppia immagine-mappa di TC. Carica tutti i modelli e fa il lavoro completo.

**Come si lancia.** Serve dire quale file di rete, quale file di impostazioni e quale file di ambiente usare. Esempio nel documento [sezione 10.3](#103-creare-una-coppia-immagine--mappa-tc).

**Che cosa fa, in ordine.**
1. Prepara la memoria della scheda grafica.
2. Se hai indicato dimensioni o età, crea la mappa da zero. Altrimenti prende una mappa dall'archivio.
3. Legge le impostazioni e controlla che siano coerenti.
4. Verifica che la maschera sia del tipo giusto (TC o MR).
5. Carica i pesi dei modelli.
6. Crea le coppie e le salva.

**Che cosa scrive.** Due file per ogni coppia: l'immagine e la mappa delle strutture.

**Punto debole.** Il programma richiede una scheda grafica NVIDIA: il nome `cuda` è scritto direttamente nel codice.

### 6.2 `infer_image_from_mask.py`: crea un'immagine da una maschera che hai già

**A che cosa serve.** È il file più utile per il tuo lavoro. Prendi una tua mappa (una segmentazione in formato NIfTI) e il programma crea un'immagine che corrisponde.

**Come si lancia.** Con `--mask` indichi il tuo file, poi file di rete, ambiente e impostazioni.

**Che cosa fa.**
- Controlla la tua maschera: se ci sono numeri di struttura non previsti, avvisa; se manca la struttura "corpo" (numero 200), avvisa; se la dimensione o la spaziatura non vanno bene, la ricampiona.
- Crea l'immagine e tiene solo le parti che stanno nel corpo.

**Che cosa scrive.** Solo l'immagine. **Non salva la maschera** (ma alcuni documenti dicono il contrario).

### 6.3 `infer_image_from_mask_batch.py`: la stessa cosa, su tanti casi insieme

Legge un file (manifest JSON) con un elenco di casi e crea le immagini di tutti. Usa solo la parte di validazione dei dati, quindi non tutti i casi.

### 6.4 `sample.py`: il direttore d'orchestra

**A che cosa serve.** Coordina tutto il lavoro di `inference.py`. Decide se creare la mappa da zero (Percorso A) o prenderla dall'archivio (Percorso B), e poi decide se la coppia è accettabile.

**Cosa contiene, in parole semplici.**
- Una classe `LDMSampler` che tiene insieme tutti i modelli.
- Il ciclo principale che crea le coppie.
- Funzioni per cercare una mappa simile nell'archivio se quella esatta non c'è.
- Un controllo di qualità sui valori dei tessuti: se il fegato ha valori troppo strani, la coppia viene scartata. Ma con un solo campione la coppia viene salvata comunque (vedi [sezione 11](#11-problemi-noti)).

### 6.5 `sample_mask.py`: crea la mappa delle strutture da zero

**A che cosa serve.** Crea una mappa nuova, da zero, usando un vettore di 19 numeri che descrive che cosa vuoi: quali organi, quale dimensione, e alcuni dati del paziente (età, sesso, peso, altezza, BMI).

**Cosa c'è dentro.**
- Una lista di 14 organi o tumori, ciascuno con il suo posto nel vettore (per esempio il fegato è il primo, il tumore del polmone il decimo).
- Il controllo dei dati in ingresso: dimensioni, spaziatura, lunghezza minima del volume, e così via.

### 6.6 `utils_infer.py`: il cuore della creazione di immagini

**A che cosa serve.** Contiene il "motore" che crea l'immagine partendo dal rumore e dalla mappa. È usato da più programmi, quindi è condiviso.

**Come funziona, semplice.** Parte dal rumore. A ogni passo il ControlNet guarda la mappa e suggerisce dove mettere le strutture; il modello principale corregge l'immagine; poi si passa al passo successivo. Alla fine l'immagine compressa viene decompressa.

### 6.7 `utils.py`: gli attrezzi generali

Il file più grande. Le sue parti principali:
- **Pulizia delle mappe**: sistema le strutture create dal modello (per esempio riempie i buchi nei polmoni o nei reni).
- **Caricamento dei dati**: legge i file e li prepara per il training e per la creazione.
- **Regioni del corpo**: riconosce se la mappa è della testa, del torace, dell'addome o del bacino.
- **Perdite**: le formule che misurano quanto il modello sbaglia durante il training.

### 6.8 `find_masks.py`: cerca una mappa nell'archivio

**A che cosa serve.** Quando non crei la mappa da zero, il programma cerca nell'archivio una mappa che abbia le strutture che chiedi, nella regione giusta e con la spaziatura giusta.

Se la mappa cercata include un tumore, lo scarta se non c'è il tumore nell'elenco. Serve a non mescolare casi.

### 6.9 `quality_check.py`: il controllo di qualità

**A che cosa serve.** Guarda l'immagine creata e verifica che i valori dei tessuti siano plausibili. Per esempio: la mediana dei valori del fegato deve stare in un intervallo ragionevole.

Controlla solo sette tipi di tessuto: fegato, milza, pancreas, rene, polmone, osso, cervello. **Non controlla i tumori.**

### 6.10 `augmentation.py`: variazioni sulle maschere reali

**A che cosa serve.** Prende una mappa reale dall'archivio e la varia un po': la ruota, la allunga, la sposta, la deforma. Così il modello vede molte varianti invece di una sola.

Ogni tipo di tumore ha la sua funzione. Ci sono anche funzioni che tolgono i tumori dalla mappa (serve per creare l'immagine "senza tumore" necessaria in alcuni casi).

**Attenzione.** Alcune di queste funzioni girano senza limite se non trovano una variante valida. Se la mappa è difficile, il programma può sembrare bloccato. Vedi [sezione 11](#11-problemi-noti).

### 6.11 `diff_model_setting.py`: le impostazioni comuni

Carica i file di impostazioni e li unisce. Contiene anche una funzione per lanciare i programmi su più schede grafiche, ma **nessun programma la usa**: è codice inutilizzato.

### 6.12 `diff_model_train.py`: addestrare il modello dell'immagine

**A che cosa serve.** Insegna al modello principale a creare immagini, usando immagini reali già compresse (gli "embedding", cioè le versioni compresse preparate dal file successivo).

**Punti importanti, in parole semplici.**
- Ogni tanto il programma "dimentica" il tipo di immagine (il 10% delle volte), così il modello impara anche a lavorare senza indicazioni.
- Alla fine di ogni giro salva un file con i pesi del modello.

**Attenzione.** Si aspetta un file di accompagnamento (in formato JSON) per ogni immagine compressa, ma nessun programma del progetto lo crea. Va fatto a mano o da un notebook. Vedi [sezione 11](#11-problemi-noti).

### 6.13 `train_controlnet.py`: addestrare il controllo sulle strutture

**A che cosa serve.** Addestra il ControlNet, cioè la parte che guarda la mappa e dice al modello dove mettere le strutture. Il modello principale resta fermo.

**Punti importanti.**
- Usa la "perdita contrastiva" del paper, se attivata nella configurazione.
- Salva due file: l'ultimo stato e il migliore.

**Attenzione.** Non usa i dati di validazione: li scarta. Quindi non c'è un controllo di quanto il modello migliora su dati nuovi.

### 6.14 `diff_model_create_training_data.py`: preparare i dati di training

**A che cosa serve.** Prende le immagini reali e le comprime con il VAE, producendo versioni compresse da dare in pasto al training. È il primo passo prima di `diff_model_train.py`.

**Attenzione.** Non crea il file di accompagnamento di cui sopra.

### 6.15 `diff_model_infer.py`: creare solo immagini, senza mappa

**A che cosa serve.** Crea immagini di TC o di risonanza **senza** usare una mappa. Usa solo il modello principale. Utile, per esempio, per creare immagini di risonanza.

Il nome del file di uscita contiene tutti i parametri usati (seme, dimensione, spaziatura, modalità), così sai com'è stato creato.

### 6.16 `download_model_data.py`: scaricare i pesi

**A che cosa serve.** Scarica i modelli da Hugging Face, il sito dove sono pubblicati.

**Come si usa.** Con `--version` scegli il modello (per esempio `rflow-mr-brain`) e con `--model_only` scarichi solo i pesi.

**Attenzione.** L'opzione `--root_dir` sposta solo i dati, non i modelli: i modelli finiscono sempre nella cartella `models` dove lanci il comando.

### 6.17 `adagn_unet.py`: il modello delle mappe

Definisce il modello che crea la mappa da zero (Percorso A). È una variante del modello principale che riceve il vettore di 19 numeri e lo usa per regolare l'elaborazione a ogni passo.

### 6.18 `transforms.py`: le trasformazioni dei dati

Contiene le istruzioni per preparare le immagini: normalizzare i valori, aggiungere variazioni casuali (per la risonanza), ritagliare pezzi per l'addestramento del VAE.

### 6.19 `utils_plot.py`: disegnare i risultati

Funzioni per mostrare le immagini e le mappe a schermo. Non è usato dai programmi, solo dai notebook.

### 6.20 `compute_fid_2-5d_ct.py`: misurare la qualità

Calcola il punteggio FID (quanto le immagini sembrano reali) confrontando le immagini create con quelle reali. Per farlo scarica una rete da internet la prima volta.

### 6.21 `__init__.py`

File vuoto, con solo la licenza. Serve a Python per riconoscere la cartella `scripts` come un pacchetto.

---

## 7. I file di configurazione

I file in `configs/` sono come le impostazioni di un programma: non cambiano il codice, ma dicono come si comporta.

### 7.1 Il modello: `config_network_rflow.json`
Descrive la **forma** dei modelli: quanti strati hanno, quanti canali, quante strutture riconoscono. Non lo modifichi a meno che tu non stia cambiando architettura.

### 7.2 La creazione di immagini: `config_infer.json` e le varianti per GPU
Dicono quanti campioni creare, quale risoluzione e quale spaziatura usare, quanti passi di calcolo.

Ci sono varianti pronte per diverse quantità di memoria della scheda grafica. Esempio: `config_infer_16g_256x256x256.json` è per una scheda da 16 GB con volumi 256×256×256. Se la tua scheda è più piccola della richiesta, il programma si blocca.

### 7.3 L'addestramento: `config_maisi_diff_model_*.json` e `config_maisi_controlnet_train_*.json`
Impostazioni per l'addestramento: quante epoche (giri sui dati), a che velocità imparare, quanto grandi sono i gruppi di dati. Ci sono varianti per ogni modello (TC veloce, TC lento, risonanza, risonanza del cervello).

### 7.4 I file di ambiente: `environment_*.json`
Dicono **dove** si trovano i file: pesi, archivio delle mappe, dizionari. Ogni coppia "rete + ambiente" deve essere della stessa famiglia, altrimenti il programma non trova i file giusti.

### 7.5 `modality_mapping.json`: il tipo di scansione
Ogni tipo di scansione ha un numero: 1 per la TC, 9 per la risonanza T1, 10 per la T2, 11 per la FLAIR. Il numero viene dato al modello per dirgli che tipo di immagine creare.

Regola pratica: numeri da 1 a 7 sono TC, da 8 in su sono risonanze. Il numero **0** è speciale: è il "nessun tipo", usato durante il training.

### 7.6 `image_median_statistics_ct.json`
Contiene i valori tipici dei tessuti (fegato, milza, osso, eccetera). Serve al controllo di qualità.

---

## 8. Le etichette delle strutture

Il modello conosce **132 strutture**, ciascuna con un numero. Il file `configs/label_dict.json` è l'elenco. Alcuni esempi: fegato 1, milza 3, pancreas 4, cervello 22, tumore del polmone 23, tumore del pancreas 24, tumore del fegato 26, tumore del colon 27, lesione ossea 128, corpo 200.

Alcuni numeri sono **liberi**: il modello li ha lasciati per aggiungere strutture nuove. Questo è importante per il tuo progetto, perché il tumore cerebrale non è nell'elenco.

Due punti per non confondersi:
- Il file `label_dict_ctmr.json` ha un elenco diverso, con "brain tumor" al numero 176. **Nessun programma lo legge**, quindi il numero 176 non serve a niente per il codice attuale.
- Il modello che crea le mappe da zero usa 125 numeri in uscita. Un file (`label_dict_124_to_132.json`) li traduce nei 132 numeri finali.

---

## 9. Il percorso di un'immagine, passo per passo

### 9.1 Percorso A: mappa creata da zero
Si usa quando chiedi organi o dimensioni specifiche.

1. Il programma prepara il vettore di 19 numeri con quello che hai chiesto.
2. Il modello crea una mappa a partire dal rumore, in circa 100 passi.
3. La mappa viene decompressa e tradotta nei numeri delle 132 strutture.
4. Una serie di regole sistema la mappa (per esempio pulisce il corpo, rimette il tumore dove serve).
5. Se manca una struttura richiesta, riprova fino a cinque volte; poi va avanti con un avviso.
6. La mappa viene adattata alla dimensione e alla spaziatura richieste.
7. Si passa alla creazione dell'immagine (vedi 9.3).

### 9.2 Percorso B: mappa presa dall'archivio
Si usa quando non chiedi niente di specifico.

1. Il programma cerca nell'archivio una mappa esatta.
2. Se non ne trova abbastanza, cerca quelle più simili.
3. Sceglie una mappa a caso e la varia (vedi `augmentation.py`).
4. Si passa alla creazione dell'immagine.
5. Salva solo le strutture che hai chiesto.

### 9.3 Creare l'immagine dalla mappa
1. La mappa viene trasformata in 8 "strati" binari: ogni strato dice dove si trova una parte dei numeri. Il ControlNet legge questi strati.
2. Si parte dal rumore.
3. A ogni passo il ControlNet suggerisce e il modello principale corregge.
4. L'immagine compressa viene decompressa.
5. I valori vengono riportati alla scala dei tessuti (unità Hounsfield per la TC).
6. Le parti fuori dal corpo vengono azzerate.
7. Il controllo di qualità verifica i valori (solo per la TC).

### 9.4 Come si addestra
Il training si fa in quattro passi:

1. **Il VAE**: impara a comprimere e decomprimere le immagini. Si fa dal notebook `train_vae_tutorial.ipynb`, non con un programma separato.
2. **I dati compressi**: `diff_model_create_training_data.py` comprime tutte le immagini reali.
3. **Il modello principale**: `diff_model_train.py` impara a creare immagini compresse.
4. **Il ControlNet**: `train_controlnet.py` impara a leggere le mappe e guidare il modello principale. Il modello principale resta fermo.

### 9.5 Solo immagini, senza mappa
`diff_model_infer.py` crea immagini senza usare il ControlNet. Quindi non controlli dove si trovano le strutture.

---

## 10. Come si usa in pratica

### 10.1 Preparazione
```bash
pip install -r requirements.txt
```
Serve Python 3.11 o superiore e una scheda grafica NVIDIA con almeno 16 GB di memoria. Per le immagini più grandi servono 40 GB in creazione e 80 GB in addestramento.

### 10.2 Scaricare i pesi
```bash
python -m scripts.download_model_data --version rflow-ct --root_dir "./" --model_only
python -m scripts.download_model_data --version rflow-mr-brain --root_dir "./" --model_only
```
I pesi finiscono nella cartella `models`. La dimensione totale non l'ho verificata.

### 10.3 Creare una coppia immagine + mappa (TC)
```bash
python -m scripts.inference -t ./configs/config_network_rflow.json -i ./configs/config_infer.json -e ./configs/environment_rflow-ct.json --random-seed 0 --version rflow-ct
```

### 10.4 Creare un'immagine dalla tua mappa
```bash
python -m scripts.infer_image_from_mask --mask percorso/della/tua_mappa.nii.gz -t ./configs/config_network_rflow.json -e ./configs/environment_rflow-ct.json -i ./configs/config_infer.json
```
Questo comando usa il modello TC. **Per il cervello serve un altro modello e questo comando non è pensato per la risonanza.**

### 10.5 Addestrare il ControlNet
Esempio dalla documentazione (da verificare):
```bash
python -m scripts.train_controlnet -t ./configs/config_network_rflow.json -c ./configs/config_maisi_controlnet_train_rflow-ct.json -e ./configs/environment_maisi_controlnet_train_rflow-ct.json -g 1
```
Attenzione: la documentazione a volte indica il file di configurazione sbagliato. Controlla che sia quello del ControlNet, non della diffusione.

### 10.6 Controllo del codice
```bash
pre-commit run --all-files
```
Controlla lo stile del codice. Non ci sono test che verificano il funzionamento.

---

## 11. Problemi noti

Ordinati per importanza. "Verificato" = l'ho letto nel codice. "Da verificare" = va provato.

1. **Un file di ambiente rompe il programma.** `environment_rflow-mr.json` ha alcune voci vuote (`null`), e `inference.py` si blocca. (Verificato leggendo.)

2. **Il seme è una stringa.** `--random-seed` non è dichiarato come numero. Se il programma non parte, è il primo sospetto. (Da verificare.)

3. **Un nome non più valido in NumPy.** `utils.py` usa `np.long`, che nelle versioni moderne di NumPy non esiste più. Se il programma lo usa davvero, si blocca. (Da verificare.)

4. **La scheda grafica è obbligatoria.** Il nome `cuda` è scritto direttamente in più punti. Senza una scheda NVIDIA non funziona.

5. **Il ControlNet non valuta i dati nuovi.** I dati di validazione vengono scartati, quindi non sai se il modello migliora davvero.

6. **Manca un file di accompagnamento.** `diff_model_train.py` ne ha bisogno, ma nessun programma lo crea. (Da verificare se il notebook lo crea.)

7. **Il controllo di qualità non rigenera.** Se crei un solo campione e non passa il controllo, viene salvato lo stesso. La documentazione dice il contrario.

8. **Ricerca di strutture in Percorso A.** Dopo cinque tentativi, se manca una struttura richiesta, il programma va avanti con un avviso. La coppia viene salvata anche se non è quella che volevi.

9. **Percorsi di default verso file che non esistono.** Funziona solo se passi i percorsi giusti da riga di comando, come fa la documentazione.

10. **Alcune funzioni possono non finire mai.** Le variazioni delle mappe per fegato, polmone, pancreas e osso non hanno un limite di tentativi. Se la mappa è difficile, il programma può sembrare bloccato.

11. **Preset con passi diversi.** Due preset usano 1000 passi invece di 30. Potrebbe essere un errore di copia. (Da verificare.)

12. **Codice inutilizzato.** La funzione `run_torchrun` non è mai chiamata. `utils_plot.py` è usato solo dai notebook. Ci sono anche alcune ripetizioni.

13. **Un commento sbagliato.** In `diff_model_create_training_data.py` un commento dice che una variabile può non essere definita: non è vero.

14. **Il dizionario delle strutture per il tuo tumore.** Non è ancora chiaro quale dizionario usare per il tumore cerebrale. Vedi [sezione 13](#13-che-cosa-significa-per-il-tuo-progetto).

15. **Valori fissi nella pulizia delle mappe.** Alcune soglie (per esempio −800 HU per trovare l'aria) sono scritte nel codice senza spiegazioni, a parte il commento.

16. **Il significato della dimensione.** Non è chiaro cosa rappresenti il numero di `controllable_anatomy_size`. La skill dice che il tumore del fegato, nella generazione da zero, viene riconosciuto male.

---

## 12. Dove la documentazione non corrisponde al codice

Non è un giudizio: è una lista di cose da controllare prima di fidarsi.

| Dove | Cosa dice | Cosa fa il codice |
|---|---|---|
| `docs/inference.md` e `skills/infer_image-only.md` | Il "CFG" della risonanza è 10 | È 15 per `rflow-mr` e 2 per `rflow-mr-brain` |
| `skills/infer_image-from-mask.md` | Scrive due file (immagine ed etichetta) | Scrive solo l'immagine |
| `skills/infer_mask-only_via_real_aug.md` | Le variazioni si fermano dopo 1000 tentativi | Solo per il percorso BraTS; le altre possono non fermarsi mai |
| `skills/infer_mask-only_via_real_aug.md` | Il percorso di riserva ignora il filtro sulle strutture | Lo rispetta |
| `docs/inference.md` | Il controllo di qualità fallito riprova fino a 2 volte | Con un solo campione salva comunque |
| `skills/train_controlnet_image-from-mask.md` | La rete DDPM ha `num_class_embeds=128` | Non ce l'ha |
| `docs/training.md` | Comando per il ControlNet con file di configurazione sbagliato | Il comando fallirebbe |
| `docs/performance.md` | Finestra di calcolo `[64,64,32]` | Nella configurazione è `[96,96,96]` |
| `skills/download-models.md` | Opzioni `--token` e `--overwrite` | Non esistono |
| `scripts/sample.py` e `scripts/infer_image_from_mask.py` | Rimandano a file `skills/mask-generation.md` e `skills/image-from-mask.md` | I file reali hanno altri nomi |

---

## 13. Che cosa significa per il tuo progetto

Il tuo obiettivo è creare mappe con tumore cerebrale, partendo dai tuoi 30 casi, e poi creare MRI dalla mappa.

### 13.1 Quello che c'è già
- Un modello per la **risonanza del cervello** (`rflow-mr-brain`), con le sequenze T1, T2, FLAIR e SWI.
- Le impostazioni di dimensione e spaziatura coerenti con il download.
- Numeri già pronti per le sequenze: T1 è 9 (o 29 se senza cranio), T2 è 10 (o 30), FLAIR è 11 (o 31).

### 13.2 Quello che manca
1. **Il ControlNet per il cervello** non risulta scaricabile con il codice attuale. Va verificato su Hugging Face.
2. **Un modello per creare mappe del cervello**. Il modello per le mappe esistente è addestrato sul corpo intero in TC. Per le mappe cerebrali con tumore va addestrato o si deve scegliere un'altra strada.
3. **Un'etichetta per il tumore cerebrale.** Nel dizionario principale il cervello c'è (numero 22), ma il tumore no. Il numero 176 del file alternativo non è letto dal codice.

### 13.3 La scelta sull'etichetta del tumore
Il documento `RIASSUNTO_PROGETTO.md` aveva deciso di usare il numero 176. Il codice dice che non è possibile: il modello legge solo il dizionario a 132 numeri. Le strade sono due:
- **Usare un numero libero** (tra 133 e 199) per una nuova struttura "tumore cerebrale". Richiede di aggiungerla alla traduzione e di riaddestrare il modello delle mappe, perché il numero di uscita cambia.
- **Riusare una struttura vicina**, per esempio la lesione ossea (128). È più semplice, ma semanticamente è sbagliato.

È una tua decisione, va scritta da qualche parte e cambia il lavoro di addestramento.

### 13.4 Licenze
- `NV-Generate-MR-Brain` e `rflow-mr-brain`: uso libero anche commerciale.
- `rflow-mr`: solo ricerca.
- Dati MR-RATE: solo ricerca.

Se il progetto è di ricerca, sei a posto. Se in futuro vuoi uso commerciale, il modello cerebrale sì, ma i dati MR-RATE no.

### 13.5 Un percorso sensato
1. **Verificare il modello del cervello**: scaricare `rflow-mr-brain` e controllare che i file siano quelli attesi. Cercare su Hugging Face se esiste un ControlNet per il cervello.
2. **Decidere l'etichetta del tumore** e scrivere la tabella che collega le 183 strutture dei tuoi dati alle strutture del modello. È il punto più rischioso di tutto il progetto.
3. **Preparare i dati**: mappe in formato NIfTI, con le etichette come numeri interi, nella spaziatura giusta. Poi creare le versioni compresse con `diff_model_create_training_data.py`, e scrivere a mano il file di accompagnamento JSON.
4. **Provare la pipeline su un solo caso** prima di addestrare: la creazione di un'immagine dalla tua mappa deve funzionare.
5. **Addestrare** con i 24 casi di training e controllare con i 6 di validazione. Il ControlNet attuale non valuta sui dati nuovi, quindi la valutazione va scritta.
6. **Valutare**: Dice sul tumore sui casi di validazione, e controllare che le mappe generate non siano copie dei tuoi casi. Il FID da solo non basta.

### 13.6 Due lezioni dal paper per il tuo caso
- La **regola contrastiva** aiuta a far comparire i tumori piccoli e poco contrastati, come il tuo. Il codice la implementa, ma va attivata nella configurazione.
- Con **30 casi** il rischio principale è che il modello **copi** i tuoi casi invece di crearne di nuovi. Il paper usa circa 107.000 volumi: la situazione è molto diversa. Il controllo di distanza dai casi di training serve proprio a questo.

---

## 14. Piccolo glossario

- **Autoencoder (VAE)**: comprime e decomprime le immagini.
- **ControlNet**: il componente che guarda la mappa e guida il modello.
- **Dice**: quanto due maschere coincidono, da 0 a 1.
- **Diffusione**: creare un'immagine partendo dal rumore, passo dopo passo.
- **FID**: quanto le immagini sembrano reali, più basso è meglio.
- **FOV**: la dimensione reale del volume.
- **Hounsfield (HU)**: la scala dei valori della TC.
- **Maschera**: la mappa delle strutture.
- **NIfTI**: il formato dei file medici con le immagini.
- **Percorso A / B**: mappa creata da zero (A) oppure presa dall'archivio (B).
- **Rectified flow**: versione della diffusione più veloce, con meno passi.
- **Spaziatura**: quanti millimetri misura un punto del volume.
- **Spazio latente**: la versione compressa dell'immagine.
