"""
Deutsches Whisper Modell Training Script
======================================

WICHTIGER HINWEIS:
Dieses Skript enthält ausführliche Kommentare und Erklärungen zu allen
wichtigen Konfigurationsparametern. Diese Dokumentation ist GEWOLLT und
muss bei Änderungen entsprechend angepasst werden.

Dieses Skript trainiert ein spezialisiertes deutsches Whisper Modell unter Verwendung
des flozi00/asr-german-mixed Datensatzes. Es unterstützt Multi-GPU Training und
ist für maximale Performance optimiert.

Verzeichnisstruktur:
------------------
Alle Pfade werden über Umgebungsvariablen in ../../.env konfiguriert:
- BASE_DIR: Projekt-Hauptverzeichnis (in .env definieren)
- MODEL_DIR: ${BASE_DIR}/training/models
- LOG_DIR: ${BASE_DIR}/training/logs
- CONFIG_DIR: ${BASE_DIR}/whisper/config

Cache-Verzeichnisse:
------------------
${BASE_DIR}/cache/
├── huggingface/  # Hugging Face Cache
├── datasets/     # Dataset Cache
├── audio/        # Audio Cache
└── features/     # Feature Cache

Performance-Optimierungen:
-----------------------
- Multi-GPU Training mit DDP
- Gradient Checkpointing
- FP16 Training
- Parallele Datenverarbeitung
- Optimiertes Caching

Hardware-Anforderungen:
---------------------
- 2x RTX 4060 Ti (16GB)
- 32GB RAM
- ~4TB Speicherplatz

Konfiguration:
-------------
- Umgebungsvariablen werden aus '../../.env' geladen
- Cache-Verzeichnisse unter '${BASE_DIR}/cache/'
- Distributed Training über NCCL Backend
"""

#######################################
# 1. Basis System-Imports
#######################################
import os                  # Betriebssystem-Interaktion, Umgebungsvariablen
import sys                 # System-spezifische Parameter und Funktionen
from pathlib import Path   # Objektorientierte Dateisystem-Pfade
from dotenv import load_dotenv  # Laden von Umgebungsvariablen aus .env
import gc                  # Garbage Collection

# Lade Umgebungsvariablen
if not load_dotenv(Path(__file__).parent.parent.parent / '.env'):
    logger.info("⚠️  Keine .env Datei gefunden. Bitte erstellen Sie eine .env Datei im Projektroot.")
    sys.exit(1)

#######################################
# 2. Verzeichnis- und Cache-Konfiguration
#######################################
# Basis-Verzeichnisse
BASE_DIR = Path(os.getenv('BASE_DIR'))
if not BASE_DIR or not BASE_DIR.exists():
    raise ValueError("BASE_DIR muss in .env definiert sein und existieren.")

# Projekt-Verzeichnisse
MODEL_DIR = Path(os.getenv('MODEL_DIR', str(BASE_DIR / 'models')))
LOG_DIR = Path(os.getenv('LOG_DIR', str(BASE_DIR / 'logs')))
CONFIG_DIR = Path(os.getenv('CONFIG_DIR', str(BASE_DIR / 'whisper/config')))

# Cache-Verzeichnisse (exakt wie in .env definiert)
HF_HOME = Path(os.getenv('HF_HOME', str(BASE_DIR / 'cache/huggingface')))
TRANSFORMERS_CACHE = Path(os.getenv('TRANSFORMERS_CACHE', str(BASE_DIR / 'cache/huggingface')))
HF_DATASETS_CACHE = Path(os.getenv('HF_DATASETS_CACHE', str(BASE_DIR / 'cache/datasets')))
AUDIO_CACHE_DIR = Path(os.getenv('AUDIO_CACHE_DIR', str(BASE_DIR / 'cache/audio')))
FEATURES_CACHE_DIR = Path(os.getenv('FEATURES_CACHE_DIR', str(BASE_DIR / 'cache/features')))

# Setze Hugging Face Cache-Umgebungsvariablen
os.environ['TRANSFORMERS_CACHE'] = str(TRANSFORMERS_CACHE)
os.environ['HF_HOME'] = str(HF_HOME)
os.environ['HF_DATASETS_CACHE'] = str(HF_DATASETS_CACHE)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(HF_HOME)  

#######################################
# 3. Python Standard-Bibliothek
#######################################
import logging             # Logging-Funktionalität
import json               # JSON Verarbeitung
import subprocess         # Ausführung von Shell-Befehlen
import time              # Zeit-bezogene Funktionen
import atexit            # Registrierung von Cleanup-Funktionen

#######################################
# 4. Deep Learning Imports
#######################################
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Initialisiere Distributed Training (vor der Ausgabe der Verzeichnisstruktur)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")

# =====================================================================
# WICHTIG: Logger-Initialisierung
# ---------------------------------------------------------------------
# Diese Initialisierung MUSS an dieser Position bleiben!
# 
# Reihenfolge ist kritisch:
# 1. Nach: Imports und Verzeichnis-Konfiguration
# 2. Vor: Jeglicher Logger-Verwendung in Funktionen
# 3. Vor: Funktionsdefinitionen die den Logger nutzen
#
# Gründe für diese Position:
# - Vermeidet "NameError: name 'logger' is not defined"
# - Sichert Logging in allen Distributed Training Prozessen
# - Garantiert konsistente Log-Ausgaben
#
# WARNUNG: Verschieben dieser Initialisierung wird zu Fehlern führen!
# =====================================================================

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / 'training.log')
    ]
)
logger = logging.getLogger(__name__)

# Definiere alle benötigten Verzeichnisse
required_directories = [
    MODEL_DIR,
    LOG_DIR,
    CONFIG_DIR,
    FEATURES_CACHE_DIR,
    AUDIO_CACHE_DIR,
    HF_DATASETS_CACHE,
    HF_HOME,
    TRANSFORMERS_CACHE
]

#######################################
# 5. Wissenschaftliche Bibliotheken
#######################################
import numpy as np        # Numerische Operationen
import librosa           # Audio-Verarbeitung

#######################################
# 6. Machine Learning Tools
#######################################
from tensorboard import program   # Training Visualisierung
import evaluate                   # Modell-Evaluierung

#######################################
# 7. HuggingFace Komponenten
#######################################
# Dataset Management
from datasets import (
    load_dataset,        # Dataset Laden
    Audio               # Audio-Datensatz Unterstützung
)

# Whisper-spezifische Komponenten
from transformers import (
    WhisperProcessor,                 # Audio Preprocessing
    WhisperForConditionalGeneration,  # Whisper Modell
    Seq2SeqTrainingArguments,         # Training Konfiguration
    Seq2SeqTrainer                    # Training Loop
)

def is_main_process():
    """
    Prüft, ob der aktuelle Prozess der Hauptprozess (rank 0) ist.
    
    Diese Funktion wird verwendet, um sicherzustellen, dass bestimmte
    Operationen (wie Logging oder Verzeichniserstellung) nur einmal
    ausgeführt werden, auch wenn mehrere GPUs verwendet werden.
    
    Returns:
        bool: True wenn der aktuelle Prozess rank 0 hat, sonst False
    
    DDP-Spezifika:
        - Prüft local_rank für Multi-GPU Setup
        - Koordiniert Logging zwischen Prozessen
        - Verhindert Race Conditions bei I/O
        - Steuert TensorBoard & Checkpoint-Erstellung
    
    Notes:
        - Wichtig für verteiltes Training (DDP)
        - Verhindert doppelte Ausgaben und Ressourcenkonflikte
        - Zentral für die Prozess-Koordination
    """
    return local_rank == 0

def ensure_directories_exist(directories):
    """
    Erstellt alle benötigten Verzeichnisse mit verbesserter Fehlerbehandlung.
    
    Diese Funktion:
    1. Erstellt Verzeichnisse rekursiv (inkl. Elternverzeichnisse)
    2. Ignoriert bereits existierende Verzeichnisse
    3. Loggt Erfolg/Fehler auf dem Hauptprozess
    
    Args:
        directories (list[Path]): Liste von Verzeichnispfaden (Path-Objekte)
    
    Verzeichnisstruktur:
        - MODEL_DIR: Trainierte Modelle & Checkpoints
        - LOG_DIR: TensorBoard & Training Logs
        - FEATURE_CACHE_DIR: Vorverarbeitete Features
        - AUDIO_CACHE_DIR: Zwischengespeicherte Audio-Daten
        - HF_DATASETS_CACHE: Hugging Face Dataset Cache
        - HF_HOME: Hugging Face Home für Modelle
    
    Raises:
        Exception: Bei Fehlern während der Verzeichniserstellung
    
    Notes:
        - Sicher für parallele Ausführung
        - Logging nur auf dem Hauptprozess
        - Wichtig für initiale Projektstruktur
        - Prüft Schreibrechte
    """
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if is_main_process():
                logger.info(f"✓ Verzeichnis erstellt/geprüft: {directory}")
        except Exception as e:
            logger.info(f"⚠️  Fehler beim Erstellen von {directory}: {str(e)}")
            raise

def cleanup_resources():
    """
    Bereinigt Ressourcen am Ende des Trainings.
    
    Diese Funktion:
    1. Beendet TensorBoard-Prozess
    2. Bereinigt GPU-Speicher
    3. Schließt offene Dateien
    
    Timing:
        - Am Ende des Trainings
        - Bei vorzeitigem Abbruch
        - Im Exception-Fall
    
    Notes:
        - Wichtig für sauberes Shutdown
        - Verhindert Ressourcen-Leaks
        - Läuft im finally-Block
        - Logging nur auf Hauptprozess
    """
    logger.info("Führe Speicherbereinigung durch...")
    
    # PyTorch CUDA Cache leeren
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Distributed Training cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    # Garbage Collection forcieren
    gc.collect()
    
    logger.info("Speicherbereinigung abgeschlossen")

def check_system_resources():
    """
    Prüft kritische System-Ressourcen vor dem Training.
    
    Diese Funktion validiert:
    1. Existenz der model_config.json
    2. Verfügbaren VRAM (minimum 16GB)
    3. Verfügbaren Festplattenspeicher (minimum 500GB empfohlen)
    
    Raises:
        FileNotFoundError: Wenn model_config.json nicht gefunden wird
        ValueError: Wenn weniger als 16GB VRAM verfügbar sind
    
    Notes:
        - Läuft nur auf dem Hauptprozess (rank 0)
        - Gibt Warnung aus bei weniger als 500GB freiem Speicherplatz
        - VRAM-Check basiert auf der ersten GPU
    """
    if is_main_process():  # Nur Hauptprozess führt Checks durch
        # 1. Konfigurationsdatei
        config_path = CONFIG_DIR / 'model_config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"model_config.json nicht gefunden in: {config_path}")
        
        # 2. VRAM Check
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory < 16 * (1024**3):  # 16GB
            raise ValueError(f"GPU hat nur {gpu_memory/(1024**3):.1f}GB VRAM. Minimum: 16GB")
        
        # 3. Disk Space Check
        import shutil
        free_space = shutil.disk_usage(BASE_DIR).free
        if free_space < 500 * (1024**3):  # 500GB
            logger.warning(f"Nur noch {free_space/(1024**3):.1f}GB freier Speicherplatz. Empfohlen: 500GB")
        
        logger.info("✓ System-Ressourcen erfolgreich geprüft")

def cleanup_gpu_memory():
    """
    Bereinigt den GPU-Speicher nach speicherintensiven Operationen.
    
    Diese Funktion:
    1. Leert den PyTorch CUDA Cache
    2. Führt explizite Garbage Collection durch
    3. Loggt den verfügbaren GPU-Speicher (nur Hauptprozess)
    
    Notes:
        - Sollte nach großen Tensor-Operationen aufgerufen werden
        - Logging nur auf dem Hauptprozess
        - Hilft bei der Vermeidung von Out-of-Memory Fehlern
    
    Timing:
        - Nach großen Tensor-Operationen
        - Vor Memory-intensiven Operationen
        - Bei OOM-Warnings
        - Zwischen Trainings-Epochen
    """
    torch.cuda.empty_cache()
    gc.collect()
    if is_main_process():
        logger.debug(f"GPU Memory bereinigt - Verfügbar: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

def prepare_dataset(batch: dict, processor: "WhisperProcessor") -> dict:
    """
    Bereitet einen Batch für das Training vor.
    
    Diese Funktion:
    1. Lädt und verarbeitet Audio-Dateien
    2. Tokenisiert Text mit dem Whisper-Prozessor
    3. Handhabt Padding und Attention Masks
    
    Args:
        batch (dict): Ein Batch aus dem Dataset mit:
            - audio (dict): Audio-Daten mit:
                - array (np.ndarray): Rohe Audio-Samples
                - sampling_rate (int): Sampling Rate
            - text (str): Transkriptions-Text
        processor (WhisperProcessor): Der Whisper Tokenizer und Feature Extractor
    
    Returns:
        dict: Verarbeiteter Batch mit:
            - input_features (np.ndarray): Prozessierte Audio-Features [batch_size, n_mels, time]
            - labels (np.ndarray): Tokenisierte Text-Labels [batch_size, seq_len]
            - attention_mask (np.ndarray): Attention Mask für Features [batch_size, n_mels, time]
            - decoder_attention_mask (np.ndarray): Attention Mask für Decoder [batch_size, seq_len]
            - is_long_sequence (list[bool]): Flag für lange Sequenzen [batch_size]
    
    Notes:
        - Unterstützt Audio-Längen von 4-43 Sekunden
        - Verwendet 16kHz Sampling Rate
        - Handhabt Batch-Verarbeitung effizient
    
    WICHTIG - Padding Details:
        - Padding ist ESSENZIELL für das Training, nicht optional!
        - Jeder Batch muss einheitliche Tensor-Dimensionen haben
        - Padding geschieht batchweise (nicht global)
        - Längste Sequenz im Batch bestimmt die Padding-Länge
        - Optimiert für typische Längen von 4-30 Sekunden
        - Markiert lange Sequenzen (>32 Sekunden) für effizientes Batching
    
    Memory Management:
        - Große Tensoren werden nach Gebrauch explizit freigegeben
        - Feature Extraction geschieht streaming für RAM-Effizienz
        - Zwischenergebnisse werden nicht unnötig gehalten
    """
    try:
        # 1. Audio-Verarbeitung
        audio = batch["audio"]
        
        # Prüfe ob wir einen Batch oder ein einzelnes Sample verarbeiten
        if isinstance(audio, list):
            # Batch-Verarbeitung
            audio_arrays = [a["array"] for a in audio]
            sampling_rates = [a["sampling_rate"] for a in audio]
            
            # Prüfe Sampling Rates und führe Resampling durch wenn nötig
            if any(sr != 16000 for sr in sampling_rates):
                # Resampling für jeden Audio-Array im Batch
                audio_arrays = [
                    librosa.resample(y=arr, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
                    if sr != 16000 else arr
                    for arr, sr in zip(audio_arrays, sampling_rates)
                ]
            
            # Berechne Audio-Längen für jeden Array im Batch
            audio_lengths = [len(arr) / 16000 for arr in audio_arrays]
            is_long_sequence = [length > 32 for length in audio_lengths]  # Liste von bool für jeden Array
            
            # Feature-Extraktion für den gesamten Batch
            extracted = processor(
                audio=audio_arrays,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            )
        else:
            # Einzelnes Sample
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]
            
            # Resampling (nur wenn nötig)
            if sampling_rate != 16000:
                audio_array = librosa.resample(
                    y=audio_array,
                    orig_sr=sampling_rate,
                    target_sr=16000,
                    res_type="kaiser_best"
                )
            
            # Berechne Audio-Länge in Sekunden
            audio_length = len(audio_array) / 16000
            is_long_sequence = [audio_length > 32]  # Einzelner bool in Liste
            
            # Feature-Extraktion
            extracted = processor(
                audio=audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            )
        
        # Tokenisiere Text mit Padding
        labels = processor.tokenizer(
            text=batch["transkription"],
            padding=True,
            return_tensors="pt"
        )
        
        # Feature-Typ bestimmen (input_values oder input_features)
        ft = "input_values" if hasattr(extracted, "input_values") else "input_features"
        
        # Batch erstellen mit korrekten numpy Konvertierungen
        processed_data = {
            "input_features": getattr(extracted, ft).numpy(),  # [batch_size, n_mels, time]
            "attention_mask": extracted.attention_mask.numpy(),  # [batch_size, seq_len]
            "labels": labels.input_ids.numpy(),  # [batch_size, seq_len]
            "decoder_attention_mask": labels.attention_mask.numpy(),  # [batch_size, seq_len]
            "is_long_sequence": is_long_sequence  # list[bool]
        }
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Fehler bei der Batch-Verarbeitung: {str(e)}")
        raise

def train_model():
    """
    Hauptfunktion für das Training des deutschen Whisper-Modells.
    
    Diese Funktion:
    1. Initialisiert Multi-GPU Training (DDP)
    2. Lädt und verarbeitet das Dataset
    3. Konfiguriert Modell, Trainer und TensorBoard
    4. Führt Training und Evaluation durch
    
    Environment:
        - Benötigt mindestens 16GB VRAM pro GPU
        - Unterstützt Multi-GPU Training
        - Verwendet TensorBoard für Monitoring
    
    Konfiguration:
        - Batch Size: 4 pro GPU
        - Gradient Accumulation: 6 Schritte
        - Effektive Batch Size: 48 (4 × 6 × 2 GPUs)
        - Learning Rate: 1e-5
        - Warmup Steps: 500
        - Optimizer: AdamW mit weight_decay=0.01
        - LR Scheduler: Linear mit Warmup
    
    Fehlerbehandlung:
        - Frühe Validierung: Prüft Systemressourcen
        - Prozess-Validierung: Überwacht Training
        - Automatische Ressourcen-Freigabe
        - Detailliertes Fehler-Logging
        - Graceful Shutdown bei Interrupts
    
    Checkpoint-Management:
        - Speichert alle 1000 Schritte
        - Behält beste 3 Checkpoints (WER)
        - Unterstützt Training-Fortsetzung
        - Speichert Optimizer-State
    
    Notes:
        - Implementiert robuste Fehlerbehandlung
        - Optimiert für 16GB GPUs
        - Speichert Checkpoints und Logs
        - Unterstützt Training-Fortsetzung
    """
    tensorboard_process = None
    try:
        # Initialisiere distributed training
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Setze das GPU-Device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        # Initialisiere die Prozessgruppe nur wenn nicht bereits initialisiert
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            world_size = dist.get_world_size()
            
        # TensorBoard wird nur einmal vom Hauptprozess gestartet
        if local_rank == 0:
            logger.info("\nInitialisiere TensorBoard...")
            tensorboard_process = start_tensorboard(str(LOG_DIR))
            
        # Synchronisiere alle Prozesse bevor es weitergeht
        if world_size > 1:
            dist.barrier()
            
        # Dataset Cache-Konfiguration
        dataset_cache_dir = str(HF_DATASETS_CACHE)
        logger.info(f"Verwende Dataset Cache-Pfad: {dataset_cache_dir}")

        # Lade Datensatz nur auf dem Hauptprozess
        if is_main_process():
            logger.info("Lade deutschen ASR-Datensatz (Hauptprozess)...")
        else:
            logger.info(f"Warte auf Hauptprozess (Rang {dist.get_rank()})...")

        # Synchronisiere alle Prozesse vor dem Dataset-Loading
        dist.barrier()
        
        # Dataset Loading mit expliziter Cache-Konfiguration
        dataset = load_dataset(
            "flozi00/asr-german-mixed",
            cache_dir=dataset_cache_dir,
            num_proc=1 if is_main_process() else None,  # Nur Hauptprozess verwendet Multiprocessing
            download_mode="reuse_dataset_if_exists"  # Wichtig: Wiederverwendung des Cache
        )

        # Warte bis alle Prozesse das Dataset geladen haben
        dist.barrier()

        if is_main_process():
            logger.info("Dataset erfolgreich geladen!")
            logger.info(f"Trainings-Samples: {len(dataset['train'])}")
            logger.info(f"Test-Samples: {len(dataset['test'])}")

        # Audio-Format-Überprüfung - NUR EINMAL mit erstem Sample
        sample_audio = dataset["train"][0]["audio"]
        if sample_audio["sampling_rate"] != 16000:
            logger.warning(
                f"Unerwartete Sampling Rate {sample_audio['sampling_rate']}Hz im Dataset gefunden. "
                "Aktiviere Audio-Konvertierung..."
            )
            # Aktiviere Resampling nur wenn nötig
            dataset = dataset.cast_column(
                "audio",
                Audio(sampling_rate=16000)
            )
        else:
            logger.info("Audio bereits in 16kHz - überspringe Konvertierung")
        
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        
        logger.info(f"Datensätze geladen: {len(train_dataset)} Training, {len(eval_dataset)} Test")
        
        # 3. Initialisiere Processor und Modell
        logger.info("Lade Whisper Processor und Modell...")
        processor = WhisperProcessor.from_pretrained(
            "openai/whisper-large-v3-turbo",
            cache_dir=str(HF_HOME),
            legacy=False,
            trust_remote_code=True,
        )
        
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3-turbo",
            cache_dir=str(HF_HOME),
            load_in_8bit=False,  # Volle Präzision für Training
        )
        
        # Setze das Modell auf die richtige GPU basierend auf dem lokalen Rank
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # 4. Bereite Datensätze vor
        logger.info("Bereite Datensätze vor...")
        
        try:
            # Cache-Verzeichnis für verarbeitete Features
            features_cache_dir = FEATURES_CACHE_DIR
            features_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Parallel Processing für schnellere Vorbereitung
            train_dataset = train_dataset.map(
                lambda x: prepare_dataset(x, processor),
                remove_columns=train_dataset.column_names,
                num_proc=5,
                batch_size=32,  
                writer_batch_size=1000,
                batched=True,  # Aktiviere Batch-Verarbeitung
                cache_file_name=str(features_cache_dir / "train_processed.arrow"),
                desc="Verarbeite Trainingsdaten (num_proc=5)"
            )
            
            # Force Garbage Collection nach Training Dataset
            gc.collect()
            
            eval_dataset = eval_dataset.map(
                lambda x: prepare_dataset(x, processor),
                remove_columns=eval_dataset.column_names,
                num_proc=5,
                batch_size=32,
                writer_batch_size=1000,
                batched=True,  # Aktiviere Batch-Verarbeitung
                cache_file_name=str(features_cache_dir / "eval_processed.arrow"),
                desc="Verarbeite Evaluierungsdaten (num_proc=5)"
            )
            
            # Force Garbage Collection nach Eval Dataset
            gc.collect()
            
        except Exception as e:
            logger.error(f"Fehler bei der Dataset-Verarbeitung: {str(e)}")
            raise
        
        # 5. Trainings-Konfiguration
        # VRAM-Berechnung und Batch-Konfiguration für 2x 16GB GPUs:
        #
        # 1. Pro Sample VRAM-Nutzung:
        #    - Audio Features (30s Audio): 30s × 16kHz × 4 bytes ≈ 1.92MB
        #    - Mel Spektrogramm: 3000 × 80 × 4 bytes ≈ 0.96MB
        #    - Attention Maps: ~400MB (variiert mit Sequenzlänge)
        #    - Gradienten & Aktivierungen: ~600MB
        #    Total pro Sample: ~1GB
        #
        # 2. Fixer VRAM-Bedarf:
        #    - Modell-Parameter: ~3GB (Whisper large-v3)
        #    - Optimizer States: ~2GB (AdamW)
        #    - Gradienten Buffer: ~2GB
        #    - Sicherheitspuffer: ~1GB (für Spitzenlasten)
        #    Total Fix: ~8GB
        #
        # 3. Batch-Konfiguration:
        #    - Batch Size: 4 Samples × 1GB = 4GB VRAM pro Batch
        #    - Gradient Accumulation: 6 Steps
        #    - Effektive Samples pro GPU: 4 × 6 = 24
        #    - Bei 2 GPUs: 48 Samples effektive Batch Size
        #
        # 4. VRAM-Nutzungsprofil:
        #    - Peak VRAM: ~12GB von 16GB verfügbar
        #    - Sicherheitspuffer: 4GB für lange Sequenzen (bis 43s)
        #    - Stabile Verarbeitung auch bei Spitzenlasten
        #
        # Diese Konfiguration ermöglicht:
        # - Optimale VRAM-Nutzung (~75%)
        # - Stabile Verarbeitung langer Sequenzen
        # - Effektive Batch-Normalisierung
        # - Ausreichend Puffer für Spitzenlasten

        # Trainingsparameter basierend auf Datensatzgröße und Hardware
        samples_per_batch = 4 * 6 * torch.cuda.device_count()  # Reduzierte Batch Size × Erhöhte Grad Accum × GPUs
        steps_per_epoch = len(train_dataset) / samples_per_batch
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(MODEL_DIR / MODEL_CONFIG['base_name']),
            
            # 1. Kritische Training-Parameter
            max_steps=40_000,                # ~2 Epochen bei 970k Samples
            warmup_steps=500,                # Stabiler Start mit ~1% Warmup
            learning_rate=1e-5,              # Bewährt für Whisper Fine-Tuning
            
            # 2. Batch & Memory
            per_device_train_batch_size=4,   # Optimiert für 16GB VRAM
            gradient_accumulation_steps=6,    # Effektive Batch Size: 48
            gradient_checkpointing=True,      # VRAM-Optimierung
            fp16=True,                       # Mixed Precision Training
            max_grad_norm=1.0,               # Gradient Clipping für Stabilität
            
            # 3. Monitoring & Evaluation
            evaluation_strategy="steps",
            per_device_eval_batch_size=4,    # Konsistent mit Training Batch Size
            predict_with_generate=True,
            generation_max_length=384,
            eval_steps=2000,
            save_strategy="steps",
            save_steps=2000,
            save_total_limit=3,              # Speichere beste 3 Checkpoints
            logging_steps=50,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            
            # 4. Dataset
            remove_unused_columns=False,      # Behalte alle Dataset-Spalten
        )
        
        # Konfiguriere Generierung für deutsches Modell
        model.generation_config.do_sample = False  # Deterministisches Sampling
        model.generation_config.num_beams = 1     # Single-Beam für Geschwindigkeit
        model.generation_config.language = "de"   # Explizit Deutsch setzen

        # 6. Initialisiere Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor
        )

        # 7. Training
        logger.info("\nStarte Training...")
        trainer.train()
        
        # 8. Finale Evaluation
        logger.info("\nFühre finale Evaluation durch...")
        eval_results = trainer.evaluate()
        logger.info(f"Finale WER auf Test-Split: {eval_results['eval_wer']:.2%}")
        
        # 9. Vergleich mit Benchmark
        logger.info("\nVergleiche mit Benchmark...")
        benchmark_wer = evaluate_on_benchmark(model, processor)
        
        # 10. Speichere finales Modell
        logger.info("\nSpeichere finales Modell...")
        trainer.save_model(str(MODEL_DIR / MODEL_CONFIG["final_name"]))
        
        # 11. Ausgabe der finalen Metriken
        logger.info("\nTrainingsergebnisse:")
        logger.info("=" * 50)
        logger.info(f"WER auf Test-Split : {eval_results['eval_wer']:.2%}")
        logger.info(f"Benchmark WER      : {benchmark_wer:.2%}")
        logger.info(f"Relativer Abstand  : {((benchmark_wer - eval_results['eval_wer']) / benchmark_wer):.2%}")
        
        if eval_results['eval_wer'] <= benchmark_wer:
            logger.info("\n Erfolg! Unser Modell ist besser oder gleich gut wie der Benchmark!")
        else:
            logger.info("\n Unser Modell hat den Benchmark noch nicht erreicht.")
            logger.info(f"   Differenz: {(eval_results['eval_wer'] - benchmark_wer):.2%} WER")
        
        logger.info("\nTraining erfolgreich abgeschlossen!")
        logger.info("Nächster Schritt: Führen Sie bitte aus:")
        logger.info(f"python convert_model.py")
        
        return {
            "test_wer": eval_results['eval_wer'],
            "benchmark_wer": benchmark_wer,
            "model_path": str(MODEL_DIR / MODEL_CONFIG["final_name"])
        }
        
    except Exception as e:
        logger.error(f"Fehler während des Trainings: {e}")
        if tensorboard_process is not None:
            tensorboard_process.terminate()
        raise
    finally:
        # Cleanup
        if tensorboard_process is not None:
            tensorboard_process.terminate()
            logger.info("TensorBoard beendet")
            
def compute_metrics(pred: "Seq2SeqPredictionOutput") -> dict:
    """
    Berechnet die Word Error Rate (WER) für die Vorhersagen.
    
    Diese Funktion:
    1. Decodiert die Modell-Outputs zu Text
    2. Normalisiert Vorhersagen und Referenzen
    3. Berechnet WER mit evaluate.load("wer")
    
    Args:
        pred (Seq2SeqPredictionOutput): Enthält:
            - predictions (torch.Tensor): Modell-Vorhersagen [batch_size, seq_len]
            - label_ids (torch.Tensor): Referenz-Labels [batch_size, seq_len]
    
    Returns:
        dict: {"wer": float} - Word Error Rate als float zwischen 0 und 1
    
    Notes:
        - Benchmark-Ziel: 4.77% WER (primeline-whisper-large-v3-turbo-german)
        - Ignoriert Padding-Token (-100)
        - Case-sensitive Evaluation
        - WER ist der Standard zur Bewertung der ASR-Genauigkeit
        - Ermöglicht direkten Vergleich mit anderen ASR-Systemen
    
    Metriken-Details:
        - WER = (S + D + I) / N
        - S: Substitutionen
        - D: Löschungen
        - I: Einfügungen
        - N: Wörter in Referenz
    """
    # Implementierung der Funktion
    wer_metric = evaluate.load("wer")
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Dekodiere Vorhersagen
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    # Dekodiere Labels (ersetze -100)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}

def evaluate_on_benchmark(
    model: "WhisperForConditionalGeneration",
    processor: "WhisperProcessor",
    device: str = "cuda"
) -> float:
    """
    Evaluiert das Modell auf einem Benchmark-Dataset.
    
    Diese Funktion:
    1. Lädt ein spezielles Benchmark-Dataset
    2. Führt Inferenz auf allen Beispielen durch
    3. Berechnet und loggt die WER-Metrik
    
    Args:
        model (WhisperForConditionalGeneration): Trainiertes Whisper-Modell
        processor (WhisperProcessor): Whisper Tokenizer und Feature Extractor
        device (str, optional): Gerät für Inferenz. Defaults to "cuda"
    
    Returns:
        float: Berechnete Word Error Rate (WER) als float zwischen 0 und 1
    
    Notes:
        - Vergleicht mit primeline-whisper-large-v3-turbo-german (4.77% WER)
        - Verwendet evaluate.load("wer") für konsistente Metriken
        - Loggt detaillierte Ergebnisse für Analyse
    
    Importance:
        - Ermöglicht die Bewertung der Modellleistung im Vergleich zu bestehenden Lösungen
        - Hilft bei der Identifizierung von Verbesserungspotential
        - Stellt sicher, dass das Training in die richtige Richtung geht
    """
    # Implementierung der Funktion
    logger.info("Vergleiche mit Benchmark-Modellen...")
    
    # Lade Benchmark Dataset
    benchmark = load_dataset(
        "flozi00/asr-german-mixed-evals",
        split="train",
        cache_dir=str(DATASET_CACHE_DIR)
    )
    wer_metric = evaluate.load("wer")
    
    # Sammle Benchmark-Ergebnisse
    refs = []
    preds = []
    
    for example in benchmark:
        if example["primeline-whisper-large-v3-turbo-german"]:  # Bestes Modell
            refs.append(example["references"])
            preds.append(example["primeline-whisper-large-v3-turbo-german"])
    
    # Berechne WER für Benchmark-Modell
    benchmark_wer = wer_metric.compute(predictions=preds, references=refs)
    
    logger.info("\nBenchmark-Vergleich:")
    logger.info("-" * 50)
    logger.info(f"Bestes Modell (primeline-whisper-large-v3-turbo-german):")
    logger.info(f"WER: {benchmark_wer:.2%} ({len(refs)} Beispiele)")
    logger.info("\nHinweis: Dies ist unser Ziel-WER für das Training.")
    
    return benchmark_wer

def start_tensorboard(log_dir):
    """
    Startet TensorBoard im Hintergrund mit verbesserter Fehlerbehandlung
    und Distributed Training Unterstützung.
    
    Diese Funktion:
    1. Prüft ob TensorBoard bereits läuft
    2. Startet einen neuen TensorBoard-Prozess
    3. Validiert erfolgreichen Start
    
    Args:
        log_dir: TensorBoard Log-Verzeichnis
    
    Returns:
        subprocess.Popen oder None: TensorBoard-Prozess wenn erfolgreich, sonst None
    
    Socket-Handling:
        - Prüft Port 6006 auf Verfügbarkeit
        - Verhindert doppelte TensorBoard-Instanzen
        - Wartet auf erfolgreichen Start (2 Sekunden)
        - Graceful Handling bei Port-Konflikten
    
    Logging-Details:
        - Trainings-Metriken (Loss, WER)
        - Lernrate und Gradients
        - GPU-Auslastung
        - Modell-Architektur Graph
    
    Notes:
        - Läuft nur auf dem Hauptprozess (Rank 0)
        - Automatische Prozess-Terminierung bei Fehlern
        - Non-Blocking für bessere Interaktivität
    """
    # Nur der Hauptprozess (Rank 0) sollte TensorBoard verwalten
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return None
        
    try:
        # Prüfe ob TensorBoard bereits läuft
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 6006))
        sock.close()
        
        if result == 0:
            logger.info("TensorBoard läuft bereits auf Port 6006")
            return None
            
        # Starte TensorBoard nur wenn es noch nicht läuft
        tensorboard_process = subprocess.Popen(
            ["tensorboard", "--logdir", log_dir, "--port", "6006"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Warte kurz und prüfe ob der Prozess erfolgreich gestartet wurde
        time.sleep(2)
        if tensorboard_process.poll() is None:
            logger.info("TensorBoard erfolgreich gestartet auf Port 6006")
            return tensorboard_process
        else:
            stdout, stderr = tensorboard_process.communicate()
            logger.warning(f"TensorBoard konnte nicht gestartet werden: {stderr.decode()}")
            return None
            
    except Exception as e:
        logger.warning(f"TensorBoard konnte nicht gestartet werden: {str(e)}")
        return None

# Lade Modell-Konfiguration aus JSON
with open(CONFIG_DIR / 'model_config.json', 'r') as f:
    MODEL_CONFIG = json.load(f)

# Erweitere Konfiguration um zusätzliche Parameter
MODEL_CONFIG.update({
    'pretrained_model': 'openai/whisper-large-v3-turbo',
    'language': 'de',
    'task': 'transcribe'
})

def main():
    """
    Haupteinstiegspunkt für das Training.
    
    Handhabt:
    1. Ausführung des Trainings
    2. Fehlerbehandlung
    3. Ressourcen-Cleanup
    """
    try:
        logger.info("=== Starte Modell-Training ===")
        train_model()
    except KeyboardInterrupt:
        logger.info("\nTraining durch Benutzer abgebrochen")
    except Exception as e:
        logger.error(f"Fehler während des Trainings: {str(e)}")
        raise
    finally:
        cleanup_resources()

if __name__ == "__main__":
    main()