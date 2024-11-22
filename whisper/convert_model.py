#!/usr/bin/env python3
"""
Konvertiert das trainierte Whisper-Modell von SafeTensors nach .bin Format

Konvertierungs-Pipeline:
1. SafeTensors -> .bin (dieser Schritt)
2. .bin -> CTranslate2 (nächster Schritt)

Dieser erste Konvertierungsschritt ist notwendig, da:
1. CTranslate2 nur .bin Format unterstützt
2. SafeTensors zwar sicherer ist, aber nicht von allen Tools unterstützt wird
"""

import os
import sys
import json
import logging
from pathlib import Path
from model_utils import get_model_path

# Konfiguration der Pfade
BASE_DIR = Path("/media/fukuro/raid5/RealtimeSTT")
MODEL_DIR = BASE_DIR / "training/models"
LOG_DIR = BASE_DIR / "training/logs"
CONFIG_FILE = BASE_DIR / "training/scripts/model_config.json"

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / 'model_conversion.log')
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

def convert_model():
    """
    Konvertiert das Modell von SafeTensors nach .bin
    
    Dieser Schritt ist notwendig für:
    1. Kompatibilität mit CTranslate2
    2. Vorbereitung für die Faster-Whisper Konvertierung
    """
    try:
        # Lade Konfiguration
        config = load_model_config()
        model_name = config["final_name"]
        
        model_path = MODEL_DIR / model_name
        if not model_path.exists():
            raise ValueError(f"Modell nicht gefunden: {model_path}")
            
        # Erzwinge Konvertierung auch wenn .bin existiert
        converted_path = get_model_path(str(model_path), force_convert=True)
        logger.info(f"Modell erfolgreich konvertiert: {converted_path}")
        
        logger.info("\nNächster Schritt: Führen Sie bitte aus:")
        logger.info(f"python convert_to_faster.py")
        return True
        
    except Exception as e:
        logger.error(f"Fehler bei der Konvertierung: {e}")
        return False

def main():
    if not convert_model():
        sys.exit(1)

if __name__ == "__main__":
    main()
