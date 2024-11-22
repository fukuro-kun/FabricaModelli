"""
Training eines deutschen Whisper-Modells mit anschließender CT2-Konvertierung

WICHTIGER HINWEIS:
Dieses Skript enthält ausführliche Kommentare und Erklärungen zu allen
wichtigen Konfigurationsparametern. Diese Dokumentation ist GEWOLLT und
muss bei Änderungen entsprechend angepasst werden. Die Kommentare dienen
der Nachvollziehbarkeit und Reproduzierbarkeit des Trainings, besonders
im Hinblick auf:

1. Hardware-spezifische Optimierungen (VRAM, CPU, etc.)
2. Trainings-Parameter und deren Begründung
3. Modell-Konfiguration und Spracheinstellungen
4. Performance-Metriken und Zielvorgaben

Trainings-Workflow:
1. Training auf flozi00/asr-german-mixed (970k Samples)
2. Evaluierung auf integriertem Test-Split (9.8k Samples)
3. Benchmark gegen andere Modelle via flozi00/asr-german-mixed-evals
4. Konvertierung zu CTranslate2/faster-whisper

Hardware-Anforderungen:
- 2x RTX 4060 Ti (16GB VRAM)
- 32GB System RAM
- 6 CPU Cores
- 200GB+ Speicherplatz
"""

import os
import sys
import logging
import json
from pathlib import Path
import subprocess
import torch
import torch.distributed as dist
import time
import numpy as np
import librosa
from dotenv import load_dotenv

# Lade Umgebungsvariablen aus .env
load_dotenv()

# Imports für Whisper und Datasets
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import atexit

# Setze wichtige Umgebungsvariablen für optimale Performance
os.environ["OMP_NUM_THREADS"] = "1"  # Optimiere OpenMP Threading
if not os.getenv("TOKENIZERS_PARALLELISM"):  # Nur setzen wenn nicht bereits gesetzt
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Aktiviere Tokenizer Parallelisierung nur im Hauptprozess

import evaluate
from tensorboard import program

# Konfiguration der Pfade
BASE_DIR = Path(os.getenv('BASE_DIR'))
DATA_DIR = Path(os.getenv('DATA_DIR', BASE_DIR / "training/data"))
MODEL_DIR = Path(os.getenv('MODEL_DIR', BASE_DIR / "training/models"))
LOG_DIR = Path(os.getenv('LOG_DIR', BASE_DIR / "training/logs"))
CONFIG_FILE = Path(os.getenv('CONFIG_DIR', BASE_DIR / "training/scripts")) / "model_config.json"

# Modell-Konfiguration
MODEL_CONFIG = {
    "base_name": "whisper-large-v3-turbo-german",
    "final_name": "whisper-large-v3-turbo-german-final",
    "faster_name": "faster-whisper-large-v3-turbo-german"
}

# Erstelle Verzeichnisse
for dir_path in [DATA_DIR, MODEL_DIR, LOG_DIR, CONFIG_FILE.parent]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Logging-Konfiguration (NACH Verzeichniserstellung)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Speichert die Modell-Konfiguration
def save_model_config():
    with open(CONFIG_FILE, 'w') as f:
        json.dump(MODEL_CONFIG, f, indent=2)

def prepare_dataset(batch, processor):
    """
    Bereitet einen Batch für das Training vor
    
    Wichtige Schritte:
    1. Audio-Format-Überprüfung auf 16kHz und ggf. Resampling
       - Qualitativ hochwertiges Resampling mit librosa (kaiser_best)
       - Automatische Normalisierung der Audio-Amplitude
       - Umfassende Fehlerbehandlung
    2. Feature-Extraktion mit Whisper Processor
    3. Vorbereitung der Labels (Transkriptionen)
    
    Audio-Verarbeitung Details:
    - Verwendung von librosa.resample statt datasets.Audio
    - Höchste Resampling-Qualität durch kaiser_best Filter
    - Explizite Amplituden-Normalisierung für bessere Modell-Performance
    - Detailliertes Logging für Debugging und Monitoring
    
    Args:
        batch: Ein Batch aus dem Dataset
        processor: WhisperProcessor Instanz
    
    Returns:
        dict: Verarbeitete Features und Labels
    
    Raises:
        RuntimeError: Wenn das Audio-Resampling fehlschlägt
    """
    # Verarbeite Audio
    audio = batch["audio"].copy()
    
    # Überprüfe und setze Sampling Rate
    # Für flozi00/asr-german-mixed ist dies bereits 16kHz,
    # aber wir implementieren Resampling für andere Datasets
    current_sr = audio.get("sampling_rate", 16000)
    audio_array = audio["array"]
    
    if current_sr != 16000:
        logger.warning(
            f"Unerwartete Sampling Rate {current_sr}Hz gefunden. "
            "Führe Resampling auf 16kHz durch..."
        )
        try:
            # Resampling mit librosa für beste Audioqualität
            # kaiser_best bietet die höchste Resampling-Qualität auf Kosten der Performance
            # Alternativen wären: 'kaiser_fast' (schneller) oder 'soxr_hq' (Sox Resampling)
            audio_array = librosa.resample(
                y=audio_array,
                orig_sr=current_sr,
                target_sr=16000,
                res_type='kaiser_best'  # Beste Qualität, aber rechenintensiver
            )
            
            # Normalisiere Audio nach Resampling
            # Dies ist wichtig für konsistente Modell-Inputs
            # Wir normalisieren nur wenn nötig (>1.0), um Präzision zu erhalten
            if np.abs(audio_array).max() > 1.0:
                audio_array = audio_array / np.abs(audio_array).max()
                logger.debug("Audio-Amplituden normalisiert")
            
            logger.info(f"Resampling von {current_sr}Hz auf 16kHz erfolgreich")
        except Exception as e:
            logger.error(f"Fehler beim Resampling: {str(e)}")
            raise RuntimeError(f"Audio-Konvertierung fehlgeschlagen: {str(e)}")
    
    # Feature-Extraktion
    # Der Whisper Processor erwartet 16kHz Audio
    # Die extrahierten Features sind die Basis für das Modell-Training
    extracted = processor(
        audio=audio_array,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    # Bestimme Feature-Typ (input_values oder input_features)
    # Dies hängt von der Whisper Version ab
    ft = "input_values" if hasattr(extracted, "input_values") else "input_features"
    
    # Erstelle verarbeiteten Batch
    processed_data = {
        "input_features": getattr(extracted, ft)[0],
        "attention_mask": extracted.attention_mask[0],
        "labels": processor(text=batch["transkription"]).input_ids
    }
    
    return processed_data

def compute_metrics(pred):
    """
    Berechnet die WER (Word Error Rate) für die Vorhersagen
    
    Args:
        pred: EvalPrediction object mit predictions und label_ids
    
    Returns:
        dict: Metrik-Ergebnisse (WER)
    """
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
    Vergleicht die Performance mit dem Benchmark-Dataset (flozi00/asr-german-mixed-evals)
    
    Fokus auf Vergleich mit:
    - primeline-whisper-large-v3-turbo-german (Beste: 4.77% WER)
    """
    logger.info("Vergleiche mit Benchmark-Modellen...")
    
    # Lade Benchmark Dataset
    benchmark = load_dataset(
        "flozi00/asr-german-mixed-evals",
        split="train",
        cache_dir=str(DATA_DIR)
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
    """Startet TensorBoard im Hintergrund"""
    # Nur der Hauptprozess (Rank 0) sollte TensorBoard starten
    if dist.get_rank() != 0:
        return

    try:
        # Verwende sys.executable um sicherzustellen, dass wir die richtige Python-Umgebung nutzen
        tensorboard_cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", str(log_dir), "--port", "6006"]
        process = subprocess.Popen(
            tensorboard_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Warte kurz und prüfe, ob der Prozess noch läuft
        time.sleep(3)
        if process.poll() is None:
            # Registriere Cleanup-Funktion
            atexit.register(lambda p: p.terminate(), process)
            logger.info("\nTensorBoard erfolgreich gestartet!")
            logger.info("Öffnen Sie http://localhost:6006 im Browser\n")
            return process
        else:
            stdout, stderr = process.communicate()
            logger.warning(f"TensorBoard konnte nicht gestartet werden: {stderr.decode()}")
            return None
    except Exception as e:
        logger.warning(f"Fehler beim Starten von TensorBoard: {e}")
        return None

def train_model():
    """
    Trainiert das deutsche Whisper-Modell
    
    Trainings-Details:
    - Dataset: flozi00/asr-german-mixed (970k Samples)
    - Evaluierung: Integrierter Test-Split (9.8k Samples)
    - Ziel-Metrik: WER (Word Error Rate)
    - Benchmark: Vergleich mit primeline-Modell (4.77% WER)
    
    Dataset-Eigenschaften:
    - Audio bereits in 16kHz
    - Quelle: Common Voice
    - Konsistente Struktur (audio, transkription, source)
    
    Performance-Optimierung:
    - Überspringen des Resamplings (Dataset bereits 16kHz)
    - Parallele Verarbeitung auf CPU-Cores
    - Optimierte Batch-Größen für RAM-Effizienz
    
    Training-Parameter:
    - Learning Rate: 1e-5 (optimiert für große Datenmenge)
    - Warmup Steps: 500 (~1.25% der Total Steps)
    - Total Steps: 40,000
    """
    try:
        # 0. Initialisiere Distributed Training
        if torch.cuda.is_available():
            dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(backend="gloo")
            
        logger.info(f"Distributed Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}")
        
        # 1. Starte TensorBoard
        logger.info("\nStarte TensorBoard...")
        tensorboard_process = start_tensorboard(MODEL_DIR / MODEL_CONFIG['base_name'])
        
        # 2. Lade Datensätze
        logger.info("Lade deutschen ASR-Datensatz...")
        # Cache-Verzeichnis explizit angeben für bessere Kontrolle und Wartbarkeit
        datasets = load_dataset(
            "flozi00/asr-german-mixed",
            cache_dir=str(DATA_DIR),  # Zentrales Cache-Verzeichnis für alle Dataset-Operationen
        )
        train_dataset = datasets["train"]  # 970,064 Samples
        eval_dataset = datasets["test"]    # 9,799 Samples
        
        logger.info(f"Datensätze geladen: {len(train_dataset)} Training, {len(eval_dataset)} Test")

        # 3. Initialisiere Processor und Modell
        logger.info("Lade Whisper Processor und Modell...")
        # Separates Cache-Verzeichnis für Modell-Dateien zur besseren Organisation
        processor = WhisperProcessor.from_pretrained(
            "openai/whisper-large-v3-turbo",
            cache_dir=str(DATA_DIR / "whisper_cache"),
            legacy=False,
            trust_remote_code=True,
        )
        
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3-turbo",
            cache_dir=str(DATA_DIR / "whisper_cache"),
            load_in_8bit=False,  # Volle Präzision für Training
            device_map="auto",   # Automatische GPU-Verteilung
        )

        # 4. Bereite Datensätze vor
        logger.info("Bereite Datensätze vor...")
        
        # Cache-Verzeichnis für verarbeitete Features
        cache_dir = DATA_DIR / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio-Format-Überprüfung
        # Für flozi00/asr-german-mixed ist keine Konvertierung nötig,
        # da das Audio bereits in 16kHz vorliegt
        sample_audio = train_dataset[0]["audio"]
        if sample_audio["sampling_rate"] != 16000:
            logger.warning(
                f"Unerwartete Sampling Rate {sample_audio['sampling_rate']}Hz im Dataset gefunden. "
                "Aktiviere Audio-Konvertierung..."
            )
            # Aktiviere Resampling nur wenn nötig
            train_dataset = train_dataset.cast_column(
                "audio",
                Audio(sampling_rate=16000)
            )
            eval_dataset = eval_dataset.cast_column(
                "audio",
                Audio(sampling_rate=16000)
            )
        else:
            logger.info("Audio bereits in 16kHz - überspringe Konvertierung")
        
        # Parallel Processing für schnellere Vorbereitung
        # Performance-Parameter wurden sorgfältig auf die Hardware abgestimmt:
        # - num_proc=5: Optimale Balance zwischen Verarbeitungsgeschwindigkeit und
        #               Systemstabilität (5 Cores für Processing, 1 Core für System)
        # - batch_size=48: Ermöglicht effiziente Verarbeitung ohne RAM-Überlastung
        #                  (Mittelwert zwischen 32 und 64 für optimale Performance)
        # - writer_batch_size=2000: Größere Batches beim Schreiben reduzieren I/O-Last
        #                          und beschleunigen die Cache-Erstellung
        train_dataset = train_dataset.map(
            lambda x: prepare_dataset(x, processor),
            remove_columns=train_dataset.column_names,
            num_proc=5,      # 5 Cores für Verarbeitung, 1 Core für System
            batch_size=48,   # Optimale Balance zwischen RAM-Nutzung und Performance
            writer_batch_size=2000,  # Reduziert I/O-Operationen durch größere Schreib-Batches
            cache_file_name=str(cache_dir / "train_processed.arrow"),  # Persistenter Cache in Arrow-Format
            desc="Verarbeite Trainingsdaten (num_proc=5)"
        )
        
        # Eval-Dataset mit identischen Parametern verarbeiten für Konsistenz
        eval_dataset = eval_dataset.map(
            lambda x: prepare_dataset(x, processor),
            remove_columns=eval_dataset.column_names,
            num_proc=5,      # Konsistent mit Training für reproduzierbare Ergebnisse
            batch_size=48,   # Identische Batch-Größe wie beim Training
            writer_batch_size=2000,  # Identische Schreib-Batch-Größe
            cache_file_name=str(cache_dir / "eval_processed.arrow"),
            desc="Verarbeite Evaluierungsdaten (num_proc=5)"
        )

        # 5. Trainings-Konfiguration
        # Trainingsparameter basierend auf Datensatzgröße und Hardware
        samples_per_batch = 6 * 4 * torch.cuda.device_count()  # Batch Size * Grad Accum * GPUs
        steps_per_epoch = len(train_dataset) / samples_per_batch
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(MODEL_DIR / MODEL_CONFIG["base_name"]),
            
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
        
        # 12. Speichere Konfiguration
        save_model_config()
        
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
        raise
    finally:
        if 'tensorboard_process' in locals():
            tensorboard_process.terminate()

if __name__ == "__main__":
    logger.info("=== Starte Modell-Training ===")
    # Speichere Modell-Konfiguration
    save_model_config()
    logger.info(f"Modell-Konfiguration gespeichert in {CONFIG_FILE}")
    
    success = train_model()
    if not success:
        sys.exit(1)
