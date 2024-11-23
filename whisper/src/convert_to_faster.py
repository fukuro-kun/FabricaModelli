"""
Konvertiert das trainierte Whisper-Modell zu CTranslate2/faster-whisper

Konvertierungs-Pipeline:
1. SafeTensors -> .bin (vorheriger Schritt)
2. .bin -> CTranslate2 (dieser Schritt)

Dieser finale Konvertierungsschritt:
1. Optimiert das Modell für schnelle Inferenz
2. Reduziert die Modellgröße durch float16 Quantisierung
3. Ermöglicht die Nutzung der Faster-Whisper Runtime
"""

import os
import sys
import json
import logging
import subprocess
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# Lade Umgebungsvariablen
load_dotenv()

# Konfiguration der Pfade
BASE_DIR = Path(os.getenv('BASE_DIR'))
MODEL_DIR = BASE_DIR / os.getenv('MODEL_DIR')
LOG_DIR = BASE_DIR / os.getenv('LOG_DIR')
CONFIG_FILE = BASE_DIR / os.getenv('CONFIG_FILE')

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_config():
    """
    Lädt die Modell-Konfiguration
    
    Die Konfiguration enthält:
    - base_name: Name des trainierten Modells
    - final_name: Name nach .bin Konvertierung
    - faster_name: Name nach CTranslate2 Konvertierung
    """
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Konfigurationsdatei nicht gefunden: {CONFIG_FILE}")
        logger.error("Bitte führen Sie zuerst das Training aus!")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error("Fehler beim Lesen der Konfigurationsdatei")
        sys.exit(1)

def validate_conversion(model_path):
    """
    Validiere das konvertierte Modell
    
    Tests:
    1. Modell-Ladung mit Faster-Whisper
    2. Inferenz mit 1 Sekunde Stille
    3. Überprüfung der deutschen Spracheinstellung
    """
    try:
        # Initialisiere Modell mit float16 für Geschwindigkeit
        model = WhisperModel(str(model_path), device="cuda", compute_type="float16")
        
        # Test-Inferenz mit leerem Audio (1 Sekunde Stille)
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 Sekunde @ 16kHz
        segments, info = model.transcribe(dummy_audio, language="de")
        
        # Validiere Sprach-Konfiguration
        if info.language != "de":
            logger.warning(f"Unerwartete Sprache erkannt: {info.language}")
        
        logger.info("Modell-Validierung erfolgreich")
        return True
    except Exception as e:
        logger.error(f"Validierung fehlgeschlagen: {e}")
        return False

def convert_model():
    """
    Konvertiert das Modell zu CTranslate2/faster-whisper Format
    
    Schritte:
    1. Konvertierung mit ct2-transformers-converter
    2. Kopieren der Tokenizer-Konfiguration
    3. Validierung des konvertierten Modells
    4. Bereitstellung für RealtimeSTT
    """
    try:
        # Lade Konfiguration
        config = load_model_config()
        input_name = config["final_name"]
        output_name = config["faster_name"]
        
        input_dir = MODEL_DIR / input_name
        output_dir = MODEL_DIR / output_name
        
        if not input_dir.exists():
            logger.error(f"Eingabeverzeichnis nicht gefunden: {input_dir}")
            return False
            
        # Erstelle Ausgabeverzeichnis
        output_dir.mkdir(exist_ok=True)
        
        # Konvertiere mit ct2-transformers-converter
        logger.info("Starte Konvertierung...")
        cmd = [
            "ct2-transformers-converter",
            "--model", str(input_dir),
            "--output_dir", str(output_dir),
            "--quantization", "float16",  # Schnellere Inferenz
            "--copy_files", "tokenizer.json", "preprocessor_config.json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Konvertierung fehlgeschlagen: {result.stderr}")
            return False
            
        logger.info(f"Modell erfolgreich konvertiert: {output_dir}")
        
        # Validiere Konvertierung
        if not validate_conversion(output_dir):
            return False
            
        logger.info("\nKonvertierung und Validierung erfolgreich!")
        logger.info("Das Modell kann nun in RealtimeSTT verwendet werden.")
        logger.info("Konfigurieren Sie dazu die .env Datei:")
        logger.info(f'WHISPER_MODEL="{output_dir}"')
        return True
        
    except Exception as e:
        logger.error(f"Fehler bei der Konvertierung: {e}")
        return False

def main():
    if not convert_model():
        sys.exit(1)

if __name__ == "__main__":
    logger.info("=== Starte Modell-Konvertierung ===")
    main()
