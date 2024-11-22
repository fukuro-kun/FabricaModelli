# FabricaModelli 🏭

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/fukuro-kun/FabricaModelli/graphs/commit-activity)
[![Maintenance Mode](https://img.shields.io/badge/Maintenance%20Mode-Passive-yellow.svg)](#)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)

*Eine Werkstatt für das Training und die Optimierung von KI-Modellen*

> ℹ️ **Aktueller Status**: Dieses Repository befindet sich in der Entwicklung. Derzeit ist das Training eines spezialisierten deutschen Whisper-Modells implementiert. Weitere Modelle und Werkzeuge sind in Planung.

## 🎯 Über das Projekt

FabricaModelli ist eine Sammlung von Tools und Skripten für das Training und die Optimierung verschiedener KI-Modelle. Der Name kommt aus dem Lateinischen und bedeutet "Modell-Werkstatt" - ein Ort, an dem Modelle mit Präzision und Sorgfalt "geschmiedet" werden.

### 🎙️ Aktueller Fokus: Deutsches Whisper-Modell

Das erste fertiggestellte Werkzeug ist ein spezialisiertes Trainings-Framework für deutsche Spracherkennung basierend auf OpenAI's Whisper. 

**Hauptmerkmale:**
- Training eines deutschen Whisper-Modells mit State-of-the-Art Performance
- Optimiert für Echtzeit-Transkription
- Trainiert auf dem flozi00/asr-german-mixed Dataset (970.064 Trainingssätze)
- Ziel-WER (Word Error Rate): 4.77% oder besser
- Multi-GPU Training mit Gradient Checkpointing für effiziente Ressourcennutzung

## 🤖 Implementierte Modelle

### Whisper 🎙️
- **Status**: ✅ Trainings-Pipeline implementiert
- **Basis**: Whisper Large V3
- **Datensatz**: flozi00/asr-german-mixed
  - 970.064 Trainingssätze
  - 9.799 Testsätze
- **Features**:
  - Optimiert für Echtzeit-Transkription
  - Multi-GPU Training Support
  - Gradient Checkpointing
  - FP16 Training
- [📚 Trainings-Guide](whisper/docs/training.md)
- [🔧 Konvertierungs-Guide](whisper/docs/conversion.md)

## 📁 Projektstruktur

```
FabricaModelli/
├── whisper/                 # Whisper-spezifische Implementierungen
│   ├── train_german_model.py
│   ├── convert_model.py
│   ├── convert_to_faster.py
│   ├── config/
│   │   └── model_config.json
│   ├── docs/               # Ausführliche Dokumentation
│   │   ├── training.md
│   │   └── conversion.md
│   └── README.md
├── shared/                  # Gemeinsam genutzte Funktionen
│   ├── logging_utils.py
│   └── gpu_utils.py
└── README.md
```

## 💻 System-Anforderungen

### Minimale Anforderungen
- 🎮 NVIDIA GPU mit 16GB VRAM
- ⚡ CUDA >= 11.8
- 💾 32GB RAM
- 💿 250GB freier Speicherplatz
- 🐍 Python 3.12

### Empfohlene Hardware
- 🎮 2x NVIDIA GPU mit je 16GB VRAM
- 💾 64GB RAM
- 💿 500GB freier Speicherplatz (SSD)
- 💪 8+ CPU Kerne

## 🚀 Installation

### 1. Umgebungsvariablen einrichten

1. Kopieren Sie `.env.template` zu `.env`:
   ```bash
   cp .env.template .env
   ```

2. Passen Sie die Pfade in `.env` an Ihr System an:
   ```bash
   # Beispiel für die .env
   BASE_DIR=/media/ihr_username/pfad/zu/FabricaModelli
   DATA_DIR=${BASE_DIR}/training/data
   MODEL_DIR=${BASE_DIR}/training/models
   LOG_DIR=${BASE_DIR}/training/logs
   CONFIG_DIR=${BASE_DIR}/training/scripts
   ```

Die Umgebungsvariablen werden von allen Skripten verwendet, um die korrekten Pfade zu finden:
- `BASE_DIR`: Hauptverzeichnis des Projekts
- `DATA_DIR`: Speicherort für Trainingsdaten
- `MODEL_DIR`: Speicherort für trainierte Modelle
- `LOG_DIR`: Verzeichnis für Logs
- `CONFIG_DIR`: Verzeichnis für Konfigurationsdateien

### 2. Modellspezifische Installation

Die weitere Installation ist modellspezifisch. Bitte folgen Sie den Anleitungen in der jeweiligen Dokumentation:

- [📚 Whisper Training Setup](whisper/docs/training.md#1-einrichtung-der-umgebung)
- [🔧 Whisper Konvertierung Setup](whisper/docs/conversion.md#1-vorbereitung-der-konvertierung)

## ⚖️ Lizenz

Dieses Projekt ist unter der [GPL-3.0 Lizenz](LICENSE) lizenziert.

## 🤝 Beitragen

Beiträge sind willkommen! Bitte erstellen Sie einen Issue oder Pull Request. Beachten Sie jedoch den Wartungshinweis am Anfang dieser README.
