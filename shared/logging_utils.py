"""
Logging-Utilities für FabricaModelli
"""

import logging
import sys
from pathlib import Path

def setup_logging(log_dir: str = "logs", log_file: str = "training.log") -> logging.Logger:
    """
    Richtet ein konfiguriertes Logging-System ein.
    
    Args:
        log_dir: Verzeichnis für Log-Dateien
        log_file: Name der Log-Datei
    
    Returns:
        logging.Logger: Konfigurierter Logger
    """
    # Erstelle Log-Verzeichnis
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Konfiguriere Logger
    logger = logging.getLogger("FabricaModelli")
    logger.setLevel(logging.INFO)
    
    # Formatter für detaillierte Ausgabe
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File Handler
    file_handler = logging.FileHandler(log_path / log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
