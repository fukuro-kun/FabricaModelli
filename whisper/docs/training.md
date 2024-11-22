# Training eines deutschen Whisper-Modells für RealtimeSTT

Spracherkennung in Echtzeit stellt besondere Anforderungen an die Genauigkeit und Geschwindigkeit der verwendeten Modelle. Während OpenAIs Whisper-Architektur bereits beeindruckende Ergebnisse für verschiedene Sprachen liefert, zeigt sich bei der deutschen Sprache noch Optimierungspotential. Insbesondere für Anwendungen wie RealtimeSTT, die eine verzögerungsfreie Transkription ermöglichen sollen, ist ein speziell optimiertes deutsches Modell von entscheidender Bedeutung.

Dieser Guide dokumentiert den Prozess des Trainings eines solchen spezialisierten deutschen Whisper-Modells. Mit dem Ziel, die aktuelle Benchmark-Wortfehlerrate von 4.77% zu erreichen oder zu übertreffen, nutzen wir einen sorgfältig kuratierten deutschen Datensatz und moderne Trainingstechniken. Der Guide ist dabei so gestaltet, dass das Training auch auf Consumer-Hardware durchführbar ist - ein wichtiger Aspekt für die Reproduzierbarkeit und Weiterentwicklung des Modells.

## Übersicht

Wir trainieren ein deutsches Spracherkennungsmodell basierend auf OpenAIs Whisper-Architektur, um die Echtzeit-Spracherkennung in RealtimeSTT zu verbessern. Unser Ziel ist es, die Wortfehlerrate (WER) von 4.77% des aktuellen Benchmarks (primeline-whisper-large-v3-turbo-german) zu erreichen oder zu übertreffen. Wir nutzen den flozi00/asr-german-mixed Datensatz mit 970,064 Trainings- und 9,799 Testbeispielen. Der Trainingsprozess läuft auf zwei RTX 4060 Ti GPUs und nutzt moderne Techniken wie Gradient Checkpointing und fp16 Training, um trotz der Hardware-Limitierungen (16GB VRAM pro GPU) effizient zu sein. Die gesamte Trainingszeit beträgt etwa 24-30 Stunden für 2 Epochen (40,000 Steps), wobei wir besonders auf eine effiziente Nutzung der verfügbaren Ressourcen (6 CPU-Kerne, 128GB RAM) achten.

## Systemvoraussetzungen

- NVIDIA GPU(s) mit mindestens 16GB VRAM
- CUDA >= 11.8
- Python 3.12
- Mindestens 250GB freier Speicherplatz
- Mindestens 32GB RAM

## Verzeichnisstruktur

```
/media/fukuro/raid5/RealtimeSTT/
├── training/
│   ├── data/          # Dataset und Cache
│   ├── models/        # Trainierte Modelle
│   ├── logs/          # Trainings-Logs
│   ├── requirements_training.txt  # Trainings-Abhängigkeiten
│   └── scripts/       # Training Skripte
│       ├── train_german_model.py
│       ├── convert_model.py
│       └── convert_to_faster.py
```

## 1. Einrichtung der Umgebung

### 1.1 Python Virtual Environment erstellen

```bash
cd /media/fukuro/raid5/RealtimeSTT
python3 -m venv training-venv
source training-venv/bin/activate
```

### 1.2 Benötigte Pakete installieren

Die Datei `training/requirements_training.txt` enthält alle notwendigen Abhängigkeiten für das Training:

- **Core Dependencies**:
  - torch==2.5.1 - PyTorch mit CUDA Support
  - transformers==4.46.3 - Hugging Face Transformers
  - datasets==3.1.0 - Dataset Handling
  - accelerate==1.1.1 - Distributed Training
  - tokenizers==0.20.3 - Tokenisierung
  - tensorboard==2.18.0 - Training Monitoring

- **Audio Processing**:
  - librosa - Audio Verarbeitung
  - soundfile - Audio I/O

Installation der Pakete:

```bash
# Upgrade pip
pip install --upgrade pip

# PyTorch mit CUDA 11.8 Support installieren
pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Trainings-Abhängigkeiten aus requirements_training.txt installieren
pip install -r training/requirements_training.txt
```

### 1.3 Installation verifizieren

```bash
# CUDA-Verfügbarkeit prüfen
python -c "import torch; print('CUDA verfügbar:', torch.cuda.is_available())"

# TensorBoard-Installation prüfen
python -c "import tensorboard; print('TensorBoard Version:', tensorboard.__version__)"
```

## 2. Training-Skript vorbereiten

### 2.1 Verzeichnisse erstellen

```bash
mkdir -p training/data training/models training/logs
```

## 3. Training starten

### 3.1 Multi-GPU Training starten

Das Training-Skript setzt automatisch wichtige Umgebungsvariablen für optimale Performance:

- `OMP_NUM_THREADS=1`: Optimiert OpenMP Threading
- `TOKENIZERS_PARALLELISM=true`: Aktiviert Tokenizer Parallelisierung

Training starten:

```bash
cd /media/fukuro/raid5/RealtimeSTT
torchrun --nproc_per_node=2 ./training/scripts/train_german_model.py
```

TensorBoard wird automatisch gestartet und ist unter `http://localhost:6006` erreichbar.

## 3.2 Trainings-Prozess

1. **Initialisierung**:
   - TensorBoard startet automatisch
   - Browser-Link wird angezeigt
   - Prozess wird beim Trainingsende automatisch beendet

2. **Datenvorbereitung**:
   - Normalisierung auf 16kHz Sampling Rate
   - Feature-Extraktion mit Whisper Processor
   - Vorbereitung der Transkriptionen
   - Parallele Verarbeitung mit 5 Prozessen (1 Kern für System reserviert)
   - Batch-Größe: 48 für optimale Balance
   - Automatisches Caching der verarbeiteten Features (Arrow-Format)
   - Cache-Speicherort: training/data/cache/

3. **Training**:
   - Batch-Größe: 6 pro GPU
   - Gradient Accumulation: 4 Steps
   - Effektive Batch-Größe: 48 (6 × 4 × 2 GPUs)
   - Checkpoints alle 2000 Schritte
   - Evaluierung alle 2000 Schritte
   - Dauer: ~20-24 Stunden
   - Hohe GPU-Auslastung normal

## 4. Audio-Verarbeitung

### 4.1 Sampling Rate und Resampling

Whisper erwartet Audio-Input mit 16kHz Sampling Rate. Unser Dataset (flozi00/asr-german-mixed) liegt bereits in diesem Format vor. Für andere Datasets implementieren wir hochqualitatives Resampling.

## 6. Training überwachen

### 6.1 Wichtige Metriken in TensorBoard

- **Loss**: Sollte kontinuierlich sinken
- **WER (Word Error Rate)**:
  - Ziel: 4.77% (Benchmark)
  - Gut: < 7%
  - Sehr gut: < 5%
- **Learning Rate**: Startet bei 1e-5, Warmup in ersten 500 Schritten
- **GPU Utilization**: Sollte bei ~95-100% liegen

### 6.2 Trainings-Phasen

1. **Datenvorbereitung**:
   - Dataset-Download und Caching
   - Feature-Extraktion und Caching in Arrow-Format
   - Dauer beim ersten Durchlauf: ~3-4 Stunden
   - Dauer bei Cache-Nutzung: deutlich reduziert
   - Hohe CPU-Auslastung nur beim ersten Durchlauf

2. **Training**:
   - 40,000 Schritte (~2 Epochen)
   - Checkpoints alle 2000 Schritte
   - Evaluierung alle 2000 Schritte
   - Dauer: ~20-24 Stunden
   - Hohe GPU-Auslastung normal

### 6.3 VRAM-Nutzung

- **Pro GPU (16GB)**:
  - ~9GB: Batch (6 Samples × ~1.5GB)
  - ~7GB: Modell, Gradienten, States
- **Optimierungen**:
  - Gradient Checkpointing: ~40% VRAM-Einsparung
  - fp16: Halbierter Speicherbedarf
  - Gradient Accumulation: 4 Steps = 48 effektive Batch Size

### 6.4 Log-Dateien

- Training-Logs: /media/fukuro/raid5/RealtimeSTT/training/logs/training.log
- TensorBoard-Logs: /media/fukuro/raid5/RealtimeSTT/training/models/whisper-large-v3-turbo-german

## 7. Fehlerbehebung

### 7.1 Häufige Probleme

- **Speicherprobleme**: Batch-Größe reduzieren
- **CPU-Überlastung**: num_proc reduzieren
- **GPU-Unterauslastung**: Batch-Größe erhöhen

### 7.2 Training fortsetzen

Bei Unterbrechung:

```bash
cd /media/fukuro/raid5/RealtimeSTT
torchrun --nproc_per_node=2 ./training/scripts/train_german_model.py
```

Dataset-Cache bleibt erhalten.

## 8. Nach dem Training

### 8.1 Evaluation und Benchmarking

Die Evaluation unseres Modells erfolgt auf zwei Ebenen:

#### Integrierte Evaluation

- **WER (Word Error Rate)**: Prozentsatz falsch erkannter Wörter
  - Benchmark-Ziel: ≤ 4.77% (primeline-Modell)
  - Berechnung: Levenshtein-Distanz zwischen Vorhersage und Referenz

### 8.2 Modell validieren

- WER auf Testset prüfen
- Stichproben-Tests durchführen
- Modell-Größe überprüfen

### 8.3 Modell exportieren

Modell wird gespeichert in:
/media/fukuro/raid5/RealtimeSTT/training/models/whisper-large-v3-turbo-german

Checkpoints in Unterverzeichnissen

## 9. Ressourcenverbrauch

### 9.1 Speicherplatz

- Dataset: ~136 GB
- Feature-Cache: ~50-60 GB
- Modell: ~5-10 GB
- Logs & Checkpoints: ~20-30 GB
- Gesamt: ~250 GB benötigt

### 9.2 Hardware-Auslastung

- GPU: 95-100% während Training
- CPU:
  - Hoch während Datenvorbereitung
  - Moderat während Training
- RAM: ~32 GB empfohlen

## 10. Zeitplan

- Datenvorbereitung: ~3-4 Stunden
- Training: ~20-24 Stunden
- Benchmark-Evaluation: ~1-2 Stunden
- Gesamtdauer: ~24-30 Stunden

## 11. Support

Bei Problemen:
1. Log-Dateien prüfen
2. TensorBoard-Metriken analysieren
3. GPU-Auslastung überwachen
4. System-Ressourcen monitoren
