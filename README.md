# FabricaModelli 

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/fukuro-kun/FabricaModelli/graphs/commit-activity)
[![Maintenance Mode](https://img.shields.io/badge/Maintenance%20Mode-Passive-yellow.svg)](#)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)

*Eine Werkstatt für das Training und die Optimierung von KI-Modellen*

> ℹ️ **Wartungshinweis**: Dieses Repository wird aus Zeitgründen passiv gewartet. Issues und Pull Requests werden gelesen, aber Antworten können einige Zeit in Anspruch nehmen. Vielen Dank für Ihr Verständnis! Dieses Repository befindet sich ausserdem in der Entwicklung. Derzeit ist das Training eines spezialisierten deutschen Whisper-Modells implementiert. Weitere Modelle und Werkzeuge sind in Planung.

## Über das Projekt

FabricaModelli ist eine Sammlung von Tools und Skripten für das Training und die Optimierung verschiedener KI-Modelle. Der Name kommt aus dem Lateinischen und bedeutet "Modell-Werkstatt" - ein Ort, an dem Modelle mit Präzision und Sorgfalt "geschmiedet" werden.

### Aktueller Fokus: Deutsches Whisper-Modell

Das erste fertiggestellte Werkzeug ist ein spezialisiertes Trainings-Framework für deutsche Spracherkennung basierend auf OpenAI's Whisper. 

**Hauptmerkmale:**
- Training eines deutschen Whisper-Modells mit State-of-the-Art Performance
- Optimiert für Echtzeit-Transkription
- Trainiert auf dem flozi00/asr-german-mixed Dataset (970.064 Trainingssätze)
- Ziel-WER (Word Error Rate): 4.77% oder besser
- Multi-GPU Training mit Gradient Checkpointing für effiziente Ressourcennutzung

## Implementierte Modelle

### Whisper 
- **Status**: Trainings-Pipeline implementiert
- **Basis**: Whisper Large V3
- **Datensatz**: flozi00/asr-german-mixed
  - 970.064 Trainingssätze
  - 9.799 Testsätze
- **Features**:
  - Optimiert für Echtzeit-Transkription
  - Multi-GPU Training Support
  - Gradient Checkpointing
  - FP16 Training
- [ Trainings-Guide](whisper/docs/training.md)
- [ Konvertierungs-Guide](whisper/docs/conversion.md)

## Projektstruktur

```
FabricaModelli/
├── whisper/                 # Whisper-spezifische Implementierungen
│   ├── train_german_model.py
│   ├── convert_model.py
│   ├── convert_to_faster.py
│   ├── config/
│   │   └── model_config.json
│   └── docs/
│       ├── training.md
│       └── conversion.md
├── training/
│   ├── data/              # Dataset und Cache
│   ├── models/            # Trainierte Modelle
│   └── logs/             # Trainings-Logs
├── shared/                # Gemeinsam genutzte Funktionen
│   ├── logging_utils.py
│   └── gpu_utils.py
├── .env                  # Lokale Umgebungsvariablen
├── .env.template         # Vorlage für Umgebungsvariablen
├── requirements.txt      # Projekt-Abhängigkeiten
└── README.md
```

## System-Anforderungen

### Minimale Anforderungen
- NVIDIA GPU mit 16GB VRAM
- CUDA >= 11.8
- 32GB RAM
- 3TB freier Speicherplatz
- Python 3.12

### Empfohlene Hardware
- 2x NVIDIA GPU mit je 16GB VRAM
- 64GB RAM
- 4TB freier Speicherplatz (SSD/RAID)
- 8+ CPU Kerne

## 💾 Speicheranforderungen

Das Whisper-Training hat erhebliche Speicheranforderungen:

### Shared Storage (für alle Worker)
- 271 Arrow-Dateien × 506 MB ≈ 137 GB
- 273 Parquet-Dateien × 493 MB ≈ 135 GB
- Gesamt: ~272 GB gemeinsam genutzter Speicher

### Worker Cache
- 10 parallele Worker
- Jeder Worker: Cache wächst auf bis zu ~270 GB
- Gesamt Worker Cache: ~2.7 TB

### Gesamtsystem
- Shared Storage: ~272 GB
- Worker Caches: ~2.7 TB
- Spitzenlast: >3 TB während der Verarbeitung

Details zum Cache-Management finden Sie in der [Whisper README](whisper/README.md#cache-management).

## Installation

### 1. Umgebungsvariablen einrichten

1. Kopieren Sie `.env.template` zu `.env`:
   ```bash
   cp .env.template .env
   ```

2. Passen Sie die Pfade in `.env` an Ihr System an:
   ```bash
   # Beispiel für die .env
   BASE_DIR=/pfad/zu/FabricaModelli
   DATA_DIR=${BASE_DIR}/training/data
   MODEL_DIR=${BASE_DIR}/training/models
   LOG_DIR=${BASE_DIR}/training/logs
   CONFIG_DIR=${BASE_DIR}/whisper
   ```

Die Umgebungsvariablen werden von allen Skripten verwendet, um die korrekten Pfade zu finden:
- `BASE_DIR`: Hauptverzeichnis des Projekts
- `DATA_DIR`: Speicherort für Trainingsdaten
- `MODEL_DIR`: Speicherort für trainierte Modelle
- `LOG_DIR`: Verzeichnis für Logs
- `CONFIG_DIR`: Verzeichnis für Whisper-Skripte und Konfiguration

### 2. Python-Umgebung einrichten

1. Erstellen Sie eine neue virtuelle Umgebung:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Installieren Sie die Abhängigkeiten:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 3. Modellspezifische Installation

Die weitere Installation ist modellspezifisch. Bitte folgen Sie den Anleitungen in der jeweiligen Dokumentation:

## Lizenz

Dieses Projekt ist unter der [GPL-3.0 Lizenz](LICENSE) lizenziert.

## Beitragen

Beiträge sind willkommen! Bitte erstellen Sie einen Issue oder Pull Request. Beachten Sie jedoch den Wartungshinweis am Anfang dieser README.
