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
    print("⚠️  Keine .env Datei gefunden. Bitte erstellen Sie eine .env Datei im Projektroot.")
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

# Cache-Verzeichnisse
# Prüfe erst auf .env Einstellungen, dann Fallback auf Standard-Pfade
HF_HOME = Path(os.getenv('HF_HOME', str(BASE_DIR / 'cache/huggingface')))
HF_DATASETS_CACHE = Path(os.getenv('HF_DATASETS_CACHE', str(BASE_DIR / 'cache/datasets')))
AUDIO_CACHE_DIR = Path(os.getenv('AUDIO_CACHE_DIR', str(BASE_DIR / 'cache/audio')))
FEATURES_CACHE_DIR = Path(os.getenv('FEATURES_CACHE_DIR', str(BASE_DIR / 'cache/features')))

# Setze Hugging Face Cache-Umgebungsvariablen
os.environ['HF_HOME'] = str(HF_HOME)
os.environ['TRANSFORMERS_CACHE'] = str(HF_HOME)  # Gleicher Pfad wie HF_HOME
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

# Debug: Zeige Pfade (nur vom Hauptprozess)
if local_rank == 0:
    print("\nVerzeichnisstruktur:")
    print(f"Project Root: {BASE_DIR}")
    print(f"HF Cache Dir: {HF_HOME}")
    print(f"Models Dir: {MODEL_DIR}")
    print(f"Logs Dir: {LOG_DIR}")
    print(f"Config Dir: {CONFIG_DIR}")
    print(f"HF Datasets Cache: {HF_DATASETS_CACHE}")
    print(f"Audio Cache: {AUDIO_CACHE_DIR}")
    print(f"Features Cache: {FEATURES_CACHE_DIR}")

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
    return local_rank == 0

def ensure_directories_exist(directories):
    """
    Erstellt alle benötigten Verzeichnisse mit verbesserter Fehlerbehandlung.
    
    Args:
        directories (list): Liste von Verzeichnispfaden (Path-Objekte)
    """
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if is_main_process():
                print(f"✓ Verzeichnis erstellt/geprüft: {directory}")
        except Exception as e:
            print(f"⚠️  Fehler beim Erstellen von {directory}: {str(e)}")
            raise

# Definiere alle benötigten Verzeichnisse
required_directories = [
    MODEL_DIR,
    LOG_DIR,
    FEATURE_CACHE_DIR,
    AUDIO_CACHE_DIR,
    HF_DATASETS_CACHE,
    HF_HOME
]

# Erstelle Verzeichnisse
ensure_directories_exist(required_directories)

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

def start_tensorboard(log_dir):
    """
    Startet TensorBoard im Hintergrund mit verbesserter Fehlerbehandlung
    und Distributed Training Unterstützung.
    
    Args:
        log_dir: TensorBoard Log-Verzeichnis
    
    Returns:
        subprocess.Popen oder None: TensorBoard-Prozess wenn erfolgreich, sonst None
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


def prepare_dataset(batch, processor):
    """
    Bereitet einen Batch für das Training vor
    
    Optimierungen:
    - Robuste Audio-Verarbeitung
    - Fehlertolerante Feature-Extraktion
    - Verbesserte Fehlerbehandlung
    
    Args:
        batch: Dataset Batch
        processor: WhisperProcessor Instanz
    
    Returns:
        dict: Verarbeiteter Batch mit Features und Labels
    """
    try:
        # Audio-Verarbeitung mit verbesserter Fehlerbehandlung
        audio = batch["audio"]
        
        # Prüfe Audio-Format
        if isinstance(audio, dict):
            audio_array = audio.get("array", None)
            sampling_rate = audio.get("sampling_rate", 16000)
        else:
            audio_array = audio
            sampling_rate = 16000
            
        if audio_array is None:
            raise ValueError("Ungültiges Audio-Format")
            
        # Resampling nur wenn nötig
        if sampling_rate != 16000:
            audio_array = librosa.resample(
                y=audio_array,
                orig_sr=sampling_rate,
                target_sr=16000,
                res_type="kaiser_best"
            )
            
        # Feature-Extraktion mit Fehlerbehandlung
        extracted = processor(
            audio=audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        # Bestimme Feature-Typ (input_values oder input_features)
        ft = "input_values" if hasattr(extracted, "input_values") else "input_features"
        
        # Erstelle verarbeiteten Batch
        processed_data = {
            "input_features": getattr(extracted, ft)[0],
            "attention_mask": extracted.attention_mask[0],
            "labels": processor(text=batch["transkription"]).input_ids
        }
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Fehler bei der Batch-Verarbeitung: {str(e)}")
        raise


def compute_metrics(pred):
    """
    Berechnet die WER (Word Error Rate) für die Vorhersagen
    
    Benchmark-Ziel: 4.77% WER (primeline-whisper-large-v3-turbo-german)
    
    Args:
        pred: EvalPrediction object mit predictions und label_ids
    
    Returns:
        dict: Metrik-Ergebnisse (WER)
    
    Warum diese Metrik wichtig ist:
    - WER ist der Standard zur Bewertung der Genauigkeit von ASR-Modellen
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


def evaluate_on_benchmark(model, processor, device="cuda"):
    """
    Vergleicht die Performance mit dem Benchmark-Dataset
    
    Fokus auf Vergleich mit:
    - primeline-whisper-large-v3-turbo-german (Beste: 4.77% WER)
    
    Args:
        model: Trainiertes Modell
        processor: WhisperProcessor
        device: Ziel-Device (default: "cuda")
    
    Returns:
        float: WER auf Benchmark-Dataset
    
    Warum Benchmarking wichtig ist:
    - Ermöglicht die Bewertung der Modellleistung im Vergleich zu bestehenden Lösungen
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


def train_model():
    """
    Trainiert das deutsche Whisper-Modell mit verbesserter Fehlerbehandlung
    und Multi-GPU Unterstützung
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
        logging.info(f"Verwende Dataset Cache-Pfad: {dataset_cache_dir}")

        # Lade Datensatz nur auf dem Hauptprozess
        if is_main_process():
            logging.info("Lade deutschen ASR-Datensatz (Hauptprozess)...")
        else:
            logging.info(f"Warte auf Hauptprozess (Rang {dist.get_rank()})...")

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
            logging.info("Dataset erfolgreich geladen!")
            logging.info(f"Trainings-Samples: {len(dataset['train'])}")
            logging.info(f"Test-Samples: {len(dataset['test'])}")

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
                cache_file_name=str(features_cache_dir / "eval_processed.arrow"),
                desc="Verarbeite Evaluierungsdaten (num_proc=5)"
            )
            
            # Force Garbage Collection nach Eval Dataset
            gc.collect()
            
        except Exception as e:
            logger.error(f"Fehler bei der Dataset-Verarbeitung: {str(e)}")
            raise
        
        # 5. Trainings-Konfiguration
        # Trainingsparameter basierend auf Datensatzgröße und Hardware
        samples_per_batch = 6 * 4 * torch.cuda.device_count()  # Batch Size * Grad Accum * GPUs
        steps_per_epoch = len(train_dataset) / samples_per_batch
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(MODEL_DIR / MODEL_CONFIG['base_name']),
            
            # VRAM-Optimierung für 2x 16GB GPUs:
            # 1. Batch Size & Gradient Accumulation:
            #    - 6 Samples/GPU × ~1.5GB ≈ 9GB VRAM pro Batch
            #    - 7GB VRAM Reserve für Modell, Gradienten, States
            #    - Gradient Accumulation: 4 Steps = 48 effektive Batch Size
            #    - Kein extra VRAM-Bedarf durch Accumulation
            per_device_train_batch_size=6,
            gradient_accumulation_steps=4,
            
            # 2. Speicheroptimierungen:
            #    - Gradient Checkpointing: ~40% VRAM-Einsparung
            #    - fp16: Halbierter Speicherbedarf
            #    - Deaktivierte Parameter-Suche für Multi-GPU Effizienz
            gradient_checkpointing=True,
            fp16=True,
            ddp_find_unused_parameters=False,
            
            # 3. Evaluation:
            #    - Größere Eval Batch Size (8) da keine Gradienten/States
            #    - Evaluierung alle 2k Steps (5% der max_steps)
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=384,
            eval_steps=2000,
            save_strategy="steps",
            save_steps=2000,
            
            # 4. Training Length:
            #    - ~40k Steps = ~2 Epochen bei 970k Samples
            #    - 500 Warmup Steps für stabilen Start
            max_steps=40_000,
            warmup_steps=500,
            
            # 5. Monitoring:
            #    - TensorBoard Logging alle 100 Steps
            #    - Speichere bestes Modell nach WER
            logging_steps=100,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            
            # 6. Optimizer:
            #    - Learning Rate für effektives Fine-Tuning
            learning_rate=1e-5,
            remove_unused_columns=False,  # Wichtig: Erlaube alle Dataset-Spalten
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
            
def cleanup_resources():
    """Bereinigt Ressourcen beim Beenden des Skripts"""
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

def main():
    try:
        train_model()
    except KeyboardInterrupt:
        logger.info("\nTraining durch Benutzer abgebrochen")
    except Exception as e:
        logger.error(f"Fehler während des Trainings: {str(e)}")
        raise
    finally:
        cleanup_resources()

# Lade Modell-Konfiguration aus JSON
with open(CONFIG_DIR / 'model_config.json', 'r') as f:
    MODEL_CONFIG = json.load(f)

# Erweitere Konfiguration um zusätzliche Parameter
MODEL_CONFIG.update({
    'pretrained_model': 'openai/whisper-large-v3-turbo',
    'language': 'de',
    'task': 'transcribe'
})

if __name__ == "__main__":
    logger.info("=== Starte Modell-Training ===")
    
    # Starte das Training
    main()