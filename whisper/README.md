# Deutsches Whisper-Modell 

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.46.3-orange.svg)](https://huggingface.co/docs/transformers/index)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Dieses Verzeichnis enthält alle notwendigen Skripte und Konfigurationen für das Training und die Optimierung eines deutschen Whisper-Modells.

## Projektziel

Entwicklung eines spezialisierten deutschen Whisper-Modells mit:
- Ziel-Performance: ≤ 4.77% Word Error Rate (WER)
- Basis-Modell: Whisper Large V3
- Datensatz: flozi00/asr-german-mixed (970.064 Training, 9.799 Test)

## Dokumentation

Detaillierte Anleitungen finden Sie in:
- [Training Guide](docs/training.md) - Vollständige Anleitung zum Training des Modells
- [Konvertierungs-Guide](docs/conversion.md) - Anleitung zur Modell-Konvertierung für Produktiveinsatz

## Hardware-Anforderungen

- 2x NVIDIA RTX 4060 Ti (16GB VRAM)
- CUDA >= 11.8
- 32GB RAM
- 4TB freier Speicherplatz:
  - ~272 GB für gemeinsam genutzten Storage
    - 271 Arrow-Dateien (137 GB)
    - 273 Parquet-Dateien (135 GB)
  - ~2.7 TB für Worker-Cache (10 Worker)
  - ~1 TB Reserve für Modelle, Logs und temporäre Dateien

> **Hinweis**: Die Cache-Größe wächst während der Verarbeitung. Detaillierte Installations- und Einrichtungsanweisungen finden Sie im [Training Guide](docs/training.md).

## Projekt-Struktur

```
{{ BASE_DIR }}     # Projekt-Hauptverzeichnis
├── cache/                              # Zentraler Cache für alle Projekte
│   ├── huggingface/                    # HuggingFace Cache (Modelle & Tokenizer)
│   ├── datasets/                       # Dataset Cache
│   ├── models/                         # Vortrainierte Modelle
│   ├── hub/                           # HuggingFace Hub Cache
│   ├── audio/                         # Audio-Verarbeitungs-Cache
│   └── features/                      # Feature-Cache für Training
├── models/                            # Trainierte Modelle (Projekt-übergreifend)
├── logs/                             # Zentrale Logs
├── shared/                           # Gemeinsam genutzte Ressourcen
├── venv/                             # Python Virtual Environment
└── whisper/                          # Whisper Projekt
    ├── src/                          # Quellcode
    │   └── train_german_model.py     # Haupt-Training-Skript
    ├── config/                       # Konfigurationsdateien
    │   └── model_config.json         # Modell-Parameter
    ├── models/                       # Projekt-spezifische Modelle
    ├── logs/                        # Projekt-spezifische Logs
    ├── docs/                        # Dokumentation
    │   ├── training.md              # Training-Guide
    │   └── conversion.md            # Konvertierungs-Guide
    └── archive/                     # Archivierte Dateien
```

## Setup

1. Aktiviere die virtuelle Umgebung:
```bash
source venv/bin/activate
```

2. Starte das Training:
```bash
cd {{ BASE_DIR }}
torchrun --nproc_per_node=2 whisper/src/train_german_model.py
```

## Cache-Management

### Shared Storage (für alle Worker)
- Arrow-Dateien: 271 × 506 MB ≈ 137 GB
- Parquet-Dateien: 273 × 493 MB ≈ 135 GB
- Gesamt: ~272 GB gemeinsam genutzter Speicher

### Worker Cache
- 10 parallele Worker
- Pro Worker: Cache wächst auf bis zu ~270 GB
- Gesamt Worker Cache: ~2.7 TB

### Gesamtsystem
- Shared Storage: ~272 GB
- Worker Caches: ~2.7 TB
- Spitzenlast: >3 TB während der Verarbeitung

### Cache-Verzeichnisse
- Datasets Cache: `/cache/datasets`
- Modelle Cache: `/cache/models`
- HuggingFace Hub: `/cache/hub`
- Audio Cache: `/cache/audio`
- Features Cache: `/cache/features`

## Monitoring

- TensorBoard-Logs befinden sich im `logs` Verzeichnis
- Trainierte Modelle werden im `models` Verzeichnis gespeichert

## Dataset

- Dataset: flozi00/asr-german-mixed
- 970k Trainingssamples
- 9.8k Testsamples
- Ziel-WER: ≤ 4.77%

## Verzeichnisstruktur und Konfiguration

Die Projektstruktur basiert auf den in `.env` definierten Umgebungsvariablen:

```bash
# Basis-Verzeichnisse
BASE_DIR={{ BASE_DIR }}
CACHE_DIR={{ BASE_DIR }}/cache
MODEL_DIR={{ BASE_DIR }}/models
LOG_DIR={{ BASE_DIR }}/logs
CONFIG_DIR={{ BASE_DIR }}/whisper/config
```

> **Wichtig**: Die tatsächlichen Pfade müssen in der `.env` Datei im Hauptverzeichnis definiert werden. 
> Diese Datei sollte NICHT mit Git versioniert werden (ist in `.gitignore`).
> Eine Vorlage finden Sie in `.env.template`.

## Quick Start

1. **Environment Setup**
```bash
source venv/bin/activate
```

2. **Training starten**
```bash
cd {{ BASE_DIR }}
torchrun --nproc_per_node=2 whisper/src/train_german_model.py
```

3. **Monitoring**
- TensorBoard: http://localhost:6006
- Logs: {{ LOG_DIR }}/training.log

## Features

- Multi-GPU Training (DDP)
- Gradient Checkpointing
- FP16 Training
- Parallele Datenverarbeitung
- Effizientes Caching
- Automatische Evaluierung
- TensorBoard Integration

## Lizenz

[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html)

Dieses Projekt ist unter der GPLv3 lizenziert - siehe die [LICENSE](LICENSE) Datei für Details.
