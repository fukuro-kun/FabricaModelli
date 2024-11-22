# Deutsches Whisper-Modell 🎙️

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.46.3-orange.svg)](https://huggingface.co/docs/transformers/index)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Dieses Verzeichnis enthält alle notwendigen Skripte und Konfigurationen für das Training und die Optimierung eines deutschen Whisper-Modells.

## 📚 Ausführliche Dokumentation

- [Training Guide](docs/training.md)
- [Konvertierungs-Guide](docs/conversion.md)

## 🤖 Modell-Details

- **Basis-Modell**: Whisper Large V3
- **Datensatz**: flozi00/asr-german-mixed
  - 970.064 Trainings-Samples
  - 9.799 Test-Samples
- **Ziel-Performance**: ≤ 4.77% Word Error Rate (WER)

## 💻 Hardware-Anforderungen

- 2x NVIDIA RTX 4060 Ti (16GB VRAM)
- 32GB RAM
- 6 CPU Cores

## 🚀 Training

### Vorbereitung

1. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

2. Konfiguration prüfen:
   - `config/model_config.json` enthält alle Trainings-Parameter
   - Passen Sie die Batch-Größe und Gradient Accumulation Steps an Ihre GPU-Konfiguration an

### Training starten

```bash
python train_german_model.py
```

Das Training dauert etwa 20-24 Stunden auf der empfohlenen Hardware.

### Trainings-Parameter

- Learning Rate: 1e-5
- Warmup Steps: 500
- Total Steps: 40.000 (~2 Epochen)
- Evaluation: Alle 2000 Steps
- Checkpoints: Alle 2000 Steps
- Batch Size: 6 pro GPU
- Gradient Accumulation Steps: 4
- Effektive Batch Size: 48

## 🔧 Modell-Konvertierung

Nach dem Training kann das Modell für schnellere Inferenz konvertiert werden:

```bash
python convert_to_faster.py
```

## 📊 Monitoring

- Detaillierte Logs in `logs/training.log`
- TensorBoard Integration
- Automatischer TensorBoard-Start

## ⚡ Performance-Optimierungen

- Distributed Training (2 GPUs)
- Gradient Checkpointing
- FP16 Training
- Hochqualitatives Audio-Resampling mit `librosa`
- Automatische Amplituden-Normalisierung

## ⚠️ Bekannte Probleme

- TensorBoard Port-Konflikte
- CUDA Dependency Issues
- Audio Resampling Challenges

## 🎯 Nächste Schritte

1. Training-Metriken überwachen
2. Modell-Performance validieren
3. Benchmark-Vergleiche durchführen
4. Basierend auf ersten Ergebnissen optimieren

## ⚖️ Lizenz

Dieses Projekt ist unter der [GPL-3.0 Lizenz](../LICENSE) lizenziert.
