# Konvertierung des deutschen Whisper-Modells in ein spezialisiertes Faster-Whisper-Modell

Nach erfolgreichem Training unseres spezialisierten deutschen Whisper-Modells ist der nächste wichtige Schritt die Optimierung für den Produktiveinsatz. Während das trainierte Modell hervorragende Erkennungsraten bietet, benötigt es für Echtzeit-Anwendungen eine spezielle Aufbereitung. Die Konvertierung zu Faster-Whisper mittels CTranslate2 ermöglicht deutlich schnellere Inferenzzeiten bei gleichbleibender Genauigkeit.

Dieser Guide führt Sie durch den zweistufigen Konvertierungsprozess: zunächst die Umwandlung des SafeTensors-Formats in das .bin-Format und anschließend die Optimierung zu einem Faster-Whisper-Modell. Dabei werden alle notwendigen Schritte und Parameter erklärt, um eine optimale Balance zwischen Geschwindigkeit und Genauigkeit zu erreichen.

## 1. Vorbereitung der Konvertierung

### 1.1 Benötigte zusätzliche Pakete

```bash
pip install ctranslate2 faster-whisper evaluate
```

### 1.2 Verzeichnisstruktur

```
${BASE_DIR}/
├── models/
│   ├── whisper-large-v3-turbo-german/        # Trainiertes Modell (SafeTensors)
│   │   ├── checkpoint-40000/
│   │   └── final/
│   └── faster-whisper-large-v3-turbo-german/ # Finales Faster-Whisper-Modell
├── scripts/
│   ├── convert_model.py         # SafeTensors zu .bin Konvertierung
│   └── convert_to_faster.py     # .bin zu Faster-Whisper Konvertierung
└── tests/
    └── test_model.py
```

## 2. Konvertierungsprozess

### 2.1 Schritt 1: SafeTensors zu .bin Format

Dieser Schritt ist notwendig, da CTranslate2 nur .bin Format unterstützt.

```bash
cd ${BASE_DIR}
python scripts/convert_model.py
```

### 2.2 Schritt 2: Konvertierung zu Faster-Whisper

Optimiert das Modell für schnelle Inferenz und validiert die deutsche Sprachkonfiguration.

```bash
python scripts/convert_to_faster.py
```

## 3. Konvertierungsparameter

### 3.1 Wichtige Parameter in convert_to_faster.py

- **quantization**: float16 für optimale Inferenz-Geschwindigkeit
- **compute_type**: float16 für GPU-Inferenz
- **language**: "de" für deutsche Spracherkennung
- **copy_files**: tokenizer.json, preprocessor_config.json

### 3.2 Validierung

- Test-Inferenz mit 1 Sekunde Stille
- Überprüfung der Sprachkonfiguration
- Benchmark-Vergleich mit anderen Modellen

## 4. Überprüfung der Konvertierung

### 4.1 Log-Dateien

- conversion.log - Protokoll der Faster-Whisper-Konvertierung
- model_conversion.log - Protokoll der .bin-Konvertierung

### 4.2 Erfolgskontrolle

Nach erfolgreicher Konvertierung sollten folgende Dateien existieren:

```
${BASE_DIR}/models/faster-whisper-large-v3-turbo-german/
├── model.bin
├── tokenizer.json
└── preprocessor_config.json
```

## 5. Fehlerbehebung

### 5.1 Häufige Probleme

- **Speicherprobleme**: Mindestens 32GB RAM für die Konvertierung nötig
- **CUDA-Fehler**: GPU sollte mindestens 16GB VRAM haben
- **Fehlende Dateien**: Pfade in den Skripten anpassen

## 6. Nächste Schritte

### 6.1 Modell-Test

Nach der Konvertierung können Sie das Modell mit RealtimeSTT testen:

```bash
python tests/test_model.py
```

### 6.2 Integration

Das konvertierte Modell kann nun in RealtimeSTT verwendet werden durch Anpassung der Konfiguration in .env:

```
WHISPER_MODEL=faster-whisper-large-v3-turbo-german
```

## 7. Performance-Optimierung

### 7.1 Inference-Parameter

- **compute_type**: float16 für beste GPU-Performance
- **num_workers**: Anzahl CPU-Threads (empfohlen: CPU-Kerne - 2)
- **device**: cuda für GPU-Inferenz
- **device_index**: 0 für erste GPU

### 7.2 Speicher-Optimierung

- **beam_size**: 1 für schnellste Inferenz
- **vad_filter**: True für effizientere Verarbeitung
- **vad_parameters**: Angepasst für deutsche Sprache
