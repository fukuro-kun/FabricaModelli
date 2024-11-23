# Training eines deutschen Whisper-Modells für RealtimeSTT

Spracherkennung in Echtzeit stellt besondere Anforderungen an die Genauigkeit und Geschwindigkeit der verwendeten Modelle. Während OpenAIs Whisper-Architektur bereits beeindruckende Ergebnisse für verschiedene Sprachen liefert, zeigt sich bei der deutschen Sprache noch Optimierungspotential. Insbesondere für Anwendungen wie RealtimeSTT, die eine verzögerungsfreie Transkription ermöglichen sollen, ist ein speziell optimiertes deutsches Modell von entscheidender Bedeutung.

Dieser Guide dokumentiert den Prozess des Trainings eines solchen spezialisierten deutschen Whisper-Modells. Mit dem Ziel, die aktuelle Benchmark-Wortfehlerrate von 4.77% zu erreichen oder zu übertreffen, nutzen wir einen sorgfältig kuratierten deutschen Datensatz und moderne Trainingstechniken. Der Guide ist dabei so gestaltet, dass das Training auch auf Consumer-Hardware durchführbar ist - ein wichtiger Aspekt für die Reproduzierbarkeit und Weiterentwicklung des Modells.

## 1. Übersicht und Systemvoraussetzungen

### 1.1 Projekt-Übersicht

Wir trainieren ein deutsches Spracherkennungsmodell basierend auf OpenAIs Whisper-Architektur, um die Echtzeit-Spracherkennung in RealtimeSTT zu verbessern. Unser Ziel ist es, die Wortfehlerrate (WER) von 4.77% des aktuellen Benchmarks (primeline-whisper-large-v3-turbo-german) zu erreichen oder zu übertreffen.

### 1.2 Dataset-Struktur

Wir verwenden den `flozi00/asr-german-mixed` Datensatz mit folgender Struktur:

```python
Dataset({
    features: ['audio', 'transkription', 'source'],
    num_rows: 970064  # Training
})

# Beispiel-Struktur eines einzelnen Datenpunkts:
{
    "audio": {
        "array": numpy.ndarray,      # Audio-Samples als 1D Array
        "sampling_rate": 16000,      # Sampling-Rate (immer 16kHz)
        "path": str                  # Original Audiodatei-Pfad
    },
    "transkription": str,           # Deutsche Transkription
    "source": str                   # Datenquelle (z.B. "common_voice")
}
```

**WICHTIG**: Diese Struktur ist kritisch für das Training. Insbesondere:
- Das `audio`-Feld muss ein Dictionary sein
- `audio["array"]` enthält die Audio-Samples als numpy.ndarray
- `audio["sampling_rate"]` muss 16000 sein
- `transkription` enthält den deutschen Text

Zum Debuggen der Dataset-Struktur kann das Skript `src/debug_dataset.py` verwendet werden:
```bash
python3 ./src/debug_dataset.py
```

### 1.2.1 Wichtiger Hinweis zu deutschen Benennungen

⚠️ **ACHTUNG**: Der Datensatz verwendet bewusst deutsche Benennungen!

- Der Key für Transkriptionen heißt `transkription` (nicht `text` oder `transcription`)
- Diese Benennung ist Teil der Datensatz-Spezifikation und darf NICHT geändert werden
- Häufige Fehlerquelle: Code, der fälschlicherweise nach `text` oder `transcription` sucht

**Typischer Fehler:**
```python
# FALSCH ❌
text = batch["text"]  # KeyError: 'text'

# RICHTIG ✅
text = batch["transkription"]
```

**Validierung der Struktur:**
```python
# Korrekte Keys überprüfen
REQUIRED_KEYS = {"audio", "transkription"}
REQUIRED_AUDIO_KEYS = {"array", "sampling_rate"}

# Validierung
missing_keys = REQUIRED_KEYS - set(batch.keys())
if missing_keys:
    raise ValueError(f"Dataset-Struktur ungültig. Fehlende Keys: {missing_keys}")
```

Diese deutsche Benennung ist eine bewusste Designentscheidung und dient als Gedankenstütze für die konsistente Verwendung deutscher Bezeichner im Projekt.

### 1.3 Hardware-Anforderungen

- 2× NVIDIA GPU mit CUDA-Support
  - VRAM: 16GB pro GPU
  - Multi-GPU Training über NCCL
  - Unterstützung für Mixed Precision (FP16)
- 64GB RAM
- 6 CPU Cores
- ~4TB Speicherplatz
  - ~700GB Dataset Cache
  - ~100GB Feature Cache
  - ~1.5TB HuggingFace Cache
  - ~1.5TB Zusätzlicher Speicher

### 1.4 Verzeichnisstruktur

```
${BASE_DIR}/
├── cache/                    # Zentrales Cache-Verzeichnis
│   ├── huggingface/         # HuggingFace Cache
│   ├── datasets/            # Dataset Cache
│   ├── audio/              # Audio Cache
│   └── features/           # Feature Cache
├── models/
│   └── whisper-large-v3-turbo-german/  # Trainiertes Modell
├── logs/                    # Training & Konvertierung Logs
└── whisper/
    ├── src/                # Quellcode
    ├── config/             # Konfigurationsdateien
    └── docs/               # Dokumentation
```

## 2. Setup und Installation

### 2.1 Python-Umgebung einrichten

Für das Training benötigen wir eine isolierte Python-Umgebung. Dies verhindert Konflikte zwischen Paketen und ermöglicht eine saubere Installation:

```bash
# Python Virtual Environment erstellen
python -m venv venv

# Umgebung aktivieren
source venv/bin/activate  # Linux

# Pip upgraden (wichtig für moderne Dependency Resolution)
pip install --upgrade pip setuptools wheel
```

### 2.2 Dependencies installieren

Die Requirements-Datei enthält alle notwendigen Pakete mit exakten Versionen für Reproduzierbarkeit:

```bash
# Basis-Dependencies
pip install -r requirements.txt

# PyTorch mit CUDA-Support (falls nicht in requirements.txt)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2.3 Projektstruktur einrichten

Das Training benötigt eine spezifische Verzeichnisstruktur. Diese wird automatisch erstellt, wenn sie nicht existiert:

```bash
# Basis-Verzeichnis setzen (anpassen an Ihren Pfad)
export BASE_DIR="/pfad/zum/projekt"

# Unterverzeichnisse erstellen
mkdir -p "${BASE_DIR}/cache/datasets"    # Dataset Cache
mkdir -p "${BASE_DIR}/cache/huggingface" # HuggingFace Cache
mkdir -p "${BASE_DIR}/cache/audio"       # Verarbeitete Audio-Dateien
mkdir -p "${BASE_DIR}/cache/features"    # Extrahierte Features
mkdir -p "${BASE_DIR}/models"            # Trainierte Modelle
mkdir -p "${BASE_DIR}/logs"              # Training Logs
```

### 2.4 Umgebungsvariablen setzen

Die Umgebungsvariablen steuern, wo das Training Daten speichert und lädt. Am besten in `.env` oder `.bashrc` speichern:

```bash
# Projekt-Pfade
export BASE_DIR="/pfad/zum/projekt"
export MODEL_DIR="${BASE_DIR}/models"
export LOG_DIR="${BASE_DIR}/logs"
export CONFIG_DIR="${BASE_DIR}/config"
export CACHE_DIR="${BASE_DIR}/cache"

# HuggingFace Cache (verhindert doppelte Downloads)
export HF_HOME="${CACHE_DIR}/huggingface"
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
```

## 3. Dataset und Datenverarbeitung

### 3.1 Dataset-Spezifika

Das flozi00/asr-german-mixed Dataset wurde analysiert und hat folgende Eigenschaften:

- **Struktur** (aus debug_dataset.py):
  ```python
  Dataset({
      features: ['audio', 'transkription', 'source'],
      num_rows: 970000
  })
  
  Features = {
      'audio': Audio(sampling_rate=16000, mono=True, decode=True),
      'transkription': Value(dtype='string'),
      'source': Value(dtype='string')
  }
  
  # Beispiel-Eintrag:
  {
      "audio": {
          'path': 'common_voice_de_21815857.mp3',
          'array': np.ndarray(shape=(62208,), dtype=float32),
          'sampling_rate': 16000
      },
      "transkription": "Es geht um das Fairplay im Wettbewerb zwischen den Mannschaften.",
      "source": "common_voice_19_0"
  }
  ```

- **Audio-Eigenschaften**:
  - Sampling-Rate: 16kHz (Whisper-Standard)
  - Format: NumPy-Array (float32)
  - Mono-Audio
  - Beispiel-Shape: (62208,) ≈ 3.9 Sekunden
  - Durchschnittliche Länge: ~4 Sekunden

- **Verarbeitung im Training**:
  - Automatische Extraktion von Audio-Array und Sampling-Rate
  - Resampling nur wenn nötig (bereits korrekte Rate)
  - Robuste Fehlerbehandlung für verschiedene Audio-Formate
  - Effiziente Feature-Extraktion mit Caching
  - Quelle ('source') wird für Training nicht benötigt

- **Cache-Struktur**:
  - Dataset Cache: ${CACHE_DIR}/datasets/
  - HuggingFace Cache: ${CACHE_DIR}/huggingface/
  - Audio Cache: ${CACHE_DIR}/audio/
  - Feature Cache: ${CACHE_DIR}/features/

### 3.2 Datenverarbeitungspipeline

Das Training durchläuft mehrere aufeinanderfolgende Datenverarbeitungsphasen, die jeweils unterschiedliche Ressourcen nutzen und verschiedene Optimierungen ermöglichen:

#### Phase 1: Dataset Download und Extraktion
- **Download der Rohdaten**
  - HuggingFace Datasets lädt die komprimierten Datensatz-Dateien herunter
  - Fortschritt wird in `~/.cache/huggingface/datasets` gespeichert
  - Sehr schnell (~500-600 files/s), da nur Metadaten verarbeitet werden

- **Extraktion und Validierung**
  - Überprüfung der Dateiintegrität
  - Entpacken der komprimierten Archive
  - Erstellen eines Datensatz-Index für schnellen Zugriff

#### Phase 2: Dataset Generation
- **Split-Generierung** (~3,000 examples/s)
  - Aufteilen in Trainings- und Testdaten
  - Erstellung effizienter Datenstrukturen für schnellen Zugriff
  - Parallelisierte Verarbeitung auf CPU-Kernen (num_proc=5)
  - Batch-Größe: 32 für optimale CPU-Auslastung
  - Speicherung im Arrow-Format für optimale Performance

- **Metadaten-Verarbeitung**
  - Extraktion von Audio-Längen und Transkriptionen
  - Erstellung von Feature-Maps für effizientes Training
  - Caching von Zwischenergebnissen für spätere Durchläufe

#### Phase 3: Audio-Vorverarbeitung
- **Feature-Extraktion** (~80-120 examples/s)
  - CPU-intensive Verarbeitung der Audiodaten
  - Resampling auf 16kHz (falls nötig)
  - Mel-Spektrogramm-Berechnung
  - Normalisierung und Tokenisierung der Transkriptionen
  - Batch-Größe: 4 für VRAM-Optimierung
  - Gradient Accumulation: 6 Steps für effektive Batch-Größe von 48

- **Caching-Strategie**
  - Zwischenspeicherung verarbeiteter Features auf SSD
  - Separate Cache-Verzeichnisse für verschiedene Verarbeitungsstufen:
    ```
    ~/.cache/huggingface/         # Dataset und Modell-Cache
    ~/.cache/whisper/audio/       # Verarbeitete Audiodaten
    ~/.cache/whisper/features/    # Extrahierte Features
    ```

**WICHTIG: Padding ist ESSENZIELL**
- Padding ist nicht optional, sondern zwingend erforderlich
- Jeder Batch muss einheitliche Tensor-Dimensionen haben
- Padding geschieht batchweise (nicht global)
- Längste Sequenz im Batch bestimmt die Padding-Länge

#### Optimierungen und Best Practices
- Nutzung von 5 CPU-Kernen für Parallelverarbeitung (1 Kern für System)
- SSD-Caching für schnellen Zugriff auf verarbeitete Daten
- Verteiltes Training über NCCL-Backend
- Automatische Anpassung der Batch-Größe basierend auf verfügbarem VRAM

#### Performance-Monitoring
- Fortschrittsbalken mit Verarbeitungsgeschwindigkeit
- Logging von Ressourcenauslastung
- Automatische Erkennung von Bottlenecks
- Speicherverbrauch-Tracking für CPU und GPU

Diese mehrstufige Pipeline ermöglicht es uns, den großen Datensatz (970,064 Samples) effizient zu verarbeiten und gleichzeitig die verfügbaren Hardware-Ressourcen optimal zu nutzen. Die Caching-Strategie stellt sicher, dass aufwändige Berechnungen nur einmal durchgeführt werden müssen.

### 3.3 Cache-Struktur und Verhalten

#### 3.3.1 HuggingFace Cache-Architektur

Der HuggingFace Cache verwendet eine Git-ähnliche BLOB-basierte Struktur:

```
huggingface/
├── models--[org]--[model]/     # z.B. models--openai--whisper-large-v3-turbo
│   ├── snapshots/             # Verschiedene Modellversionen
│   │   └── [commit-hash]/     # Spezifische Version
│   │       ├── config.json
│   │       └── model.safetensors
│   └── refs/                  # Referenzen auf aktuelle Version
│       └── main              # Pointer auf aktuellen Snapshot
├── datasets--[org]--[dataset]/ # z.B. datasets--flozi00--asr-german-mixed
│   ├── snapshots/            # Dataset-Versionen
│   │   └── [commit-hash]/    # Spezifische Version
│   │       ├── dataset_info.json
│   │       └── [split]/      # train, test, etc.
│   └── refs/                 # Referenzen auf aktuelle Version
└── .locks/                   # Sperrdateien für parallelen Zugriff
```

Vorteile dieser Struktur:
- Deduplizierung durch BLOB-Storage
- Effiziente Updates (nur Delta-Downloads)
- Robuster paralleler Zugriff
- Einfache Versionierung
- Optimierte Speichernutzung

#### 3.3.2 Worker Cache-Verhalten

Jeder Worker (Prozess) im parallelen Training:
- Erstellt eigene temporäre Dateien
- Cached Features separat
- Behält Cache während der gesamten Verarbeitung
- Bereinigt Cache nach Abschluss

Speichernutzung pro Worker:
```
features/
├── shared/                    # Gemeinsam genutzte Dateien
│   ├── arrow/                # 271 Dateien × 506 MB ≈ 137 GB
│   │   └── temp_[hash].arrow
│   └── parquet/              # 273 Dateien × 493 MB ≈ 135 GB
│       └── temp_[hash].parquet
├── worker_1/                 # Worker-spezifische Caches
│   └── cache/               # Wächst während Verarbeitung
│       └── processed_[hash] # 0 GB → ~270 GB
├── worker_2/
│   └── ...
└── worker_10/
    └── ...

Gesamt-Speicherbedarf:
- Gemeinsam genutzte Dateien:
  - Arrow-Dateien: 271 × 506 MB ≈ 137 GB
  - Parquet-Dateien: 273 × 493 MB ≈ 135 GB
  - → ~272 GB shared Storage

- Pro Worker:
  - Wachsender Cache: 0 GB → ~270 GB
  - → ~270 GB pro Worker bei voller Auslastung

Systemweite Auslastung:
- Shared Storage: ~272 GB (Arrow + Parquet)
- Worker Caches: 10 × 270 GB ≈ 2.7 TB
- Dataset Cache: [noch zu ermitteln]
- → Spitzenlast: >3 TB während der Verarbeitung
```

**Kritische Herausforderungen**:
1. **Explosives Cache-Wachstum**:
   - Jeder Worker erzeugt eigene temporäre Dateien
   - Dateien werden während der Verarbeitung nicht gelöscht
   - Cache wächst kontinuierlich bis zum Verarbeitungsende

2. **Ressourcen-Engpässe**:
   - Speicherplatz auf System-SSD schnell erschöpft
   - I/O-Bottlenecks durch parallele Schreibzugriffe
   - Hohe RAM-Auslastung durch Datenpufferung

3. **Performance-Probleme**:
   - Verlangsamung bei vollem Speicher
   - Potenzielle System-Instabilität
   - Training kann bei vollem Speicher abstürzen

#### Implementierte Lösungen

Um diese Herausforderungen zu bewältigen:

1. **Cache-Verlagerung**:
   - Alle Caches auf RAID5-System verschoben
   - Bessere I/O-Performance
   - Mehr verfügbarer Speicherplatz

2. **Worker-Optimierung**:
   ```python
   train_dataset.map(
       prepare_dataset,
       num_proc=5,              # 5 Worker pro GPU
       batch_size=32,           # 32 * 16bit * 480000 Samples ≈ 30MB pro Batch
       writer_batch_size=1000,  # 1000 * 135MB ≈ 135GB Arrow/Parquet Dateien
   )
   ```

3. **Monitoring und Cleanup**:
   - Regelmäßige Cache-Bereinigung
   - Speicherverbrauch-Überwachung
   - Automatische Notfall-Bereinigung

#### Lessons Learned

- Worker-Anzahl hat direkten Einfluss auf Speicherverbrauch
- Große `writer_batch_size` reduziert Anzahl temporärer Dateien
- RAID-System essentiell für große Datasets
- Regelmäßiges Monitoring unerlässlich

### 3.4 Herausforderungen mit dem Cache-System

#### Speicherproblematik

Die parallele Verarbeitung führt zu erheblichen Speicheranforderungen:

```
{{ ... }}
```

### 3.5 Batch-Verarbeitung

Die Batch-Verarbeitung ist ein kritischer Aspekt des Trainings:

- **Batch-Dimensionen**: 
  - Alle Tensoren behalten ihre Batch-Dimension bei
  - Dynamisches Padding auf die längste Sequenz im Batch
  - Automatische Handhabung unterschiedlicher Sequenzlängen

- **Feature-Extraktion**:
  - Robuste Audio-Verarbeitung
  - Fehlertolerante Feature-Extraktion
  - Optimierte Cache-Nutzung
  - Verbesserte Fehlerbehandlung

### 3.6 Zukünftig geplante Features

**Behandlung langer Audio-Dateien** (In Entwicklung):
- Intelligente Verarbeitung von Audio-Dateien > 30 Sekunden
- Keine Kürzung zur Vermeidung von Audio-Text-Diskrepanzen
- Experimentelle Modi für verschiedene Anwendungsfälle

## 4. Training und VRAM-Management

### 4.1 VRAM-Nutzung und Batch-Konfiguration

Die VRAM-Nutzung ist ein kritischer Faktor beim Training des Whisper-Modells. Hier ist eine detaillierte Aufschlüsselung:

#### VRAM-Berechnung pro GPU (16GB):

1. **Audio-Sample VRAM (bei 30 Sekunden Audio):**
   - Audio Features: 30s × 16kHz × 4 bytes ≈ 1.92MB
   - Mel Spektrogramm: 3000 × 80 × 4 bytes ≈ 0.96MB
   - Attention Maps: ~400MB pro Sample
   - Gradienten & Aktivierungen: ~600MB
   Total pro Sample: ~1GB

2. **Fixer VRAM-Bedarf:**
   - Modell-Parameter: ~3GB
   - Optimizer States: ~2GB
   - Gradienten Buffer: ~2GB
   - Sicherheitspuffer: ~1GB
   Total fix: ~8GB

3. **Batch-Konfiguration:**
   - Batch Size: 4 Samples × 1GB = 4GB VRAM pro Batch
   - Gradient Accumulation: 6 Steps
   - Effektive Samples pro GPU: 4 × 6 = 24
   - Bei 2 GPUs: 48 Samples effektive Batch Size

Diese Konfiguration ermöglicht:
- Stabile Verarbeitung auch langer Sequenzen (bis 43 Sekunden)
- Effiziente VRAM-Nutzung (12GB von 16GB)
- 4GB Puffer für Spitzenlasten

#### Optimierungen:

1. **Gradient Accumulation:**
   - Kleinere physische Batch Size (4)
   - Mehr Accumulation Steps (6)
   - Gleiche effektive Batch Size wie vorher (48)
   - Bessere Handhabung langer Sequenzen

2. **Mixed Precision Training:**
   - FP16 für Berechnungen
   - Reduziert VRAM-Bedarf
   - Beschleunigt Training

3. **Gradient Checkpointing:**
   - Aktiviert für alle Transformer-Layer
   - VRAM-Einsparung: ~40%
   - Leicht erhöhte Verarbeitungszeit (~10%)

### 4.2 Training starten

Das Training-Skript setzt folgende Umgebungsvariablen:
- Distributed Training Konfiguration (automatisch durch torchrun)
- Dataset und Cache-Pfade aus .env
- CUDA Device Management

Training starten:
```bash
cd ${BASE_DIR}
torchrun --nproc_per_node=2 ./whisper/train_german_model.py
```

### 4.3 Performance-Metriken

Mit der optimierten Batch-Konfiguration erreichen wir folgende Performance-Werte:

1. **Training Durchsatz:**
   - ~48 Samples pro Schritt (4 × 6 × 2 GPUs)
   - ~1.2 Samples pro Sekunde
   - ~4,320 Samples pro Stunde
   - ~103,680 Samples pro Tag

2. **Speichernutzung:**
   - VRAM: 12GB von 16GB (~75% Auslastung)
   - System RAM: 40-45GB von 64GB
   - SSD Cache: ~2.3TB

3. **Zeitliche Schätzungen:**
   - Datenvorbereitung: ~4-5 Stunden
   - Training (970k Samples): ~22-24 Stunden
   - Evaluation: ~2 Stunden
   - Gesamt: ~28-31 Stunden

4. **Checkpointing:**
   - Checkpoint-Größe: ~6GB
   - Speicherintervall: Alle 1000 Steps
   - Evaluierung: Parallel zum Training
   - Best Model Tracking: Basierend auf WER

### 4.4 TensorBoard Monitoring

TensorBoard wird automatisch gestartet und ist unter `http://localhost:6006` erreichbar.

**Wichtige Metriken**:
- **Loss**: Sollte kontinuierlich sinken
- **WER (Word Error Rate)**:
  - Ziel: 4.77% (Benchmark)
  - Gut: < 7%
  - Sehr gut: < 5%
- **Learning Rate**: Startet bei 1e-5, Warmup in ersten 500 Schritten
- **GPU Utilization**: Sollte bei ~95-100% liegen

### 4.5 VRAM und Ressourcennutzung

Die Speichernutzung wurde sorgfältig optimiert für unsere RTX 4060 Ti (16GB):

**GPU-Setup**:
- 2× NVIDIA GPU mit CUDA-Support
  - VRAM: 16GB pro GPU
  - Multi-GPU Training über NCCL
  - Unterstützung für Mixed Precision (FP16)

**VRAM-Nutzung pro GPU (16GB)**:
- **Batch (6 Samples)**:
  - ~1.5GB pro Sample
  - 6 Samples × 1.5GB = ~9GB für Batch
- **Modell & Gradienten**:
  - Basis-Modell: ~3GB
  - Gradienten & Optimizer States: ~4GB
  - Gesamt: ~7GB
- **System-Reserve**: ~400MB (Display Server etc.)

**Optimierungen für effiziente VRAM-Nutzung**:
1. **Gradient Checkpointing**:
   - Aktiviert für alle Transformer-Layer
   - VRAM-Einsparung: ~40%
   - Leicht erhöhte Verarbeitungszeit (~10%)

2. **Mixed Precision (FP16)**:
   - Halbierter Speicherbedarf für Aktivierungen
   - Numerische Stabilität durch dynamische Loss-Scaling
   - Zusätzlicher Geschwindigkeitsvorteil auf modernen GPUs

3. **Gradient Accumulation**:
   - 4 Steps × 6 Samples × 2 GPUs = 48 effektive Batch-Size
   - Ermöglicht größere effektive Batches ohne VRAM-Überlastung
   - Wichtig für Trainings-Stabilität

4. **Batch-Optimierung**:
   - 6 Samples pro GPU ist optimal für:
     - VRAM-Auslastung (~95%)
     - Verarbeitungsgeschwindigkeit
     - Trainings-Stabilität
   - Größere Batches würden VRAM überlasten
   - Kleinere Batches wären ineffizient

## 5. Monitoring und Evaluation

### 5.1 TensorBoard Metriken

TensorBoard ist unter `http://localhost:6006` erreichbar mit folgenden Metriken:

- **Loss**: Sollte kontinuierlich sinken
- **WER (Word Error Rate)**:
  - Ziel: 4.77% (Benchmark)
  - Gut: < 7%
  - Sehr gut: < 5%
- **Learning Rate**: Startet bei 1e-5, Warmup in ersten 500 Schritten
- **GPU Utilization**: Sollte bei ~95-100% liegen

### 5.2 Log-Dateien

- Training-Logs: ${BASE_DIR}/training/logs/training.log
- TensorBoard-Logs: ${BASE_DIR}/training/models/whisper-large-v3-turbo-german

## 6. Fehlerbehebung

### 6.1 Häufige Probleme

- **Speicherprobleme**: Batch-Größe reduzieren
- **CPU-Überlastung**: num_proc reduzieren
- **GPU-Unterauslastung**: Batch-Größe erhöhen

### 6.2 Training fortsetzen

Bei Unterbrechung:
```bash
python whisper/src/debug_dataset.py
```

Dataset-Cache bleibt erhalten.

## 7. Zeitplan und Ressourcen

### 7.1 Zeitplan

- Datenvorbereitung: ~3-4 Stunden
- Training: ~20-24 Stunden
- Benchmark-Evaluation: ~1-2 Stunden
- Gesamtdauer: ~24-30 Stunden

### 7.2 Hardware-Auslastung

- GPU: 95-100% während Training
- CPU:
  - Hoch während Datenvorbereitung
  - Moderat während Training
- RAM: ~32 GB empfohlen

## 8. Nach dem Training

### 8.1 Modell-Konvertierung

Nach erfolgreichem Training muss das Modell für die Produktion konvertiert werden. Dieser Prozess ist ausführlich in der [Konvertierungs-Dokumentation](conversion.md) beschrieben. Die Konvertierung umfasst:

1. Export des besten Checkpoints
2. Konvertierung in das Faster-Whisper Format
3. Quantisierung für schnellere Inferenz

### 8.2 Aufräumen nach dem Training

Das Training erzeugt verschiedene temporäre Dateien und Caches. Hier ist eine sichere Aufräum-Strategie:

```bash
# Alte Checkpoints archivieren oder löschen
# (behalte nur den besten Checkpoint)
mv ${MODEL_DIR}/checkpoint-* ${MODEL_DIR}/archive/

# TensorBoard Logs komprimieren
tar -czf ${LOG_DIR}/tensorboard_$(date +%Y%m%d).tar.gz ${LOG_DIR}/tensorboard/
rm -rf ${LOG_DIR}/tensorboard/*

# Feature Cache leeren (kann neu generiert werden)
rm -rf ${CACHE_DIR}/features/*
```

**Wichtig**: Folgende Daten sollten aufbewahrt werden:
- Bester Checkpoint (höchste Accuracy)
- TensorBoard Logs (komprimiert)
- Training Konfiguration
- Evaluierungs-Ergebnisse

Die automatische Evaluierung während des Trainings speichert bereits die besten Ergebnisse. Diese finden Sie in:
- `${LOG_DIR}/metrics.json`: Alle Metriken im JSON-Format
- `${LOG_DIR}/best_wer.txt`: Beste erreichte WER
- TensorBoard Logs: Vollständiger Trainingsverlauf

### 8.3 Nächste Schritte

Für die Produktiv-Nutzung des Modells:
1. Siehe [Konvertierungs-Dokumentation](conversion.md) für die Umwandlung in das Faster-Whisper Format
2. Testen Sie das konvertierte Modell mit Beispiel-Audio
3. Messen Sie die Inferenz-Geschwindigkeit
4. Vergleichen Sie die WER mit dem Original-Modell

## 9. Support und Wartung

### 9.1 Monitoring

1. Log-Dateien prüfen
2. TensorBoard-Metriken analysieren
3. GPU-Auslastung überwachen
4. System-Ressourcen monitoren

### 9.2 Updates

- Regelmäßige Dataset-Updates
- Code-Optimierungen
- Performance-Monitoring

#### Memory-Verhalten während der Verarbeitung

Die Datenverarbeitung zeigt ein charakteristisches Speicher- und CPU-Nutzungsmuster:

1. **Initiale Ladephase** (~85-90% RAM-Auslastung)
   - Schnelle Verarbeitung (>3000 examples/s)
   - Aufbau der Dataset-Strukturen im Speicher
   - Arrow-Format-Konvertierung
   - CPU-Kerne bei 100% Auslastung

2. **Speicherdruck-Phase**
   - Temporärer Einbruch der Performance (~80-90 examples/s)
   - RAM-Nutzung nähert sich dem Maximum
   - Ungleichmäßige CPU-Auslastung (20-80% pro Kern)
   - Python Garbage Collector wird aktiv
   - Automatische Speicherbereinigung
   - I/O-Wartezeiten durch verstärktes Swapping

3. **Stabilisierungsphase**
   - RAM-Nutzung pendelt sich ein (~70% Auslastung)
   - Performance erholt sich
   - CPU-Kerne kehren zu 100% Auslastung zurück
   - Effizientere Speichernutzung durch Cache-Strategien

4. **Steady State**
   - Periodische Performance-Schwankungen
   - "Pendelbewegungen" in der Verarbeitungsgeschwindigkeit
   - CPU-Auslastung wechselt zwischen:
     * Volle Auslastung (100% alle Kerne) während aktiver Verarbeitung
     * Reduzierte, ungleichmäßige Auslastung (20-80%) während Memory-Management-Phasen
   - Ausbalancierte Nutzung von RAM und Cache
   - Typische Performance: ~85-90 examples/s mit gelegentlichen Spitzen

Diese Phasen sind normal und zeigen das Zusammenspiel von CPU- und Memory-Management:
- Die ungleichmäßige CPU-Auslastung während der Speicherdruck-Phase deutet auf I/O-Wartezeiten hin
- Volle CPU-Auslastung bei niedrigerer RAM-Nutzung zeigt optimale Verarbeitungsbedingungen
- Periodische Garbage Collection und Cache-Rotation führen zu den beobachteten Schwankungen
- Das System findet automatisch eine Balance zwischen Verarbeitungsgeschwindigkeit und Ressourcennutzung

```
## 4. Training-Konfiguration

### 4.1 VRAM-Management

Die VRAM-Nutzung ist ein kritischer Faktor beim Training des Modells. Hier ist eine detaillierte Aufschlüsselung:

1. **Pro Sample VRAM-Nutzung:**
   - Audio Features: ~2MB (30s Audio)
   - Mel Spektrogramm: ~1MB
   - Attention Maps: ~400MB
   - Gradienten & Aktivierungen: ~600MB
   - Total pro Sample: ~1GB

2. **Fixer VRAM-Bedarf:**
   - Modell-Parameter: ~3GB
   - Optimizer States: ~2GB
   - Gradienten Buffer: ~2GB
   - Sicherheitspuffer: ~1GB
   - Total Fix: ~8GB

3. **Batch-Konfiguration:**
   - Batch Size: 4 Samples × 1GB = 4GB VRAM pro Batch
   - Gradient Accumulation: 6 Steps
   - Effektive Samples pro GPU: 24
   - Gesamt (2 GPUs): 48 Samples effektive Batch Size

4. **VRAM-Nutzungsprofil:**
   - Peak VRAM: ~12GB von 16GB
   - Sicherheitspuffer: 4GB für Spitzenlasten
   - Stabile Verarbeitung auch bei langen Sequenzen (bis 43s)

### 4.2 Training Hyperparameter

```python
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,      # Reduziert von 6 auf 4
    gradient_accumulation_steps=6,       # Erhöht von 4 auf 6
    learning_rate=1e-5,
    warmup_steps=2000,
    max_steps=100000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=4,       # Angepasst an Train Batch Size
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    report_to="wandb",
)

```

```

```
