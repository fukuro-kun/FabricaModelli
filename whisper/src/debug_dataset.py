from datasets import load_dataset
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Lade .env Datei
env_path = Path(__file__).parent.parent.parent / '.env'
if not env_path.exists():
    raise FileNotFoundError(f".env Datei nicht gefunden unter {env_path}")
load_dotenv(env_path)

# Cache-Verzeichnisse
BASE_DIR = Path(os.getenv('BASE_DIR'))
CACHE_DIR = Path(os.getenv('CACHE_DIR', BASE_DIR / 'cache'))
HF_CACHE_DIR = CACHE_DIR / "huggingface"
HF_DATASETS_CACHE = CACHE_DIR / "datasets"

# Konfiguriere Cache-Verzeichnisse
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HF_DATASETS_CACHE'] = str(HF_DATASETS_CACHE)
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR)

def inspect_dataset():
    # Lade das Dataset
    dataset = load_dataset(
        "flozi00/asr-german-mixed",
        split="train[:10]",
        cache_dir=HF_DATASETS_CACHE
    )
    
    print("\n=== Dataset Info ===")
    print(dataset)
    print("\n=== Features Info ===")
    print(dataset.features)
    
    print("\n=== Erstes Beispiel ===")
    example = dataset[0]
    # Formatiere das Beispiel schön für die Ausgabe
    formatted_example = json.dumps(
        {k: str(v) if k == "audio" else v for k, v in example.items()},
        indent=2,
        ensure_ascii=False
    )
    print(formatted_example)
    
    print("\n=== Audio Info des ersten Beispiels ===")
    audio = example["audio"]
    print(f"Audio Dict Keys: {audio.keys()}")
    print(f"Sampling Rate: {audio.get('sampling_rate', 'Not found')}")
    print(f"Array Shape: {audio['array'].shape if 'array' in audio else 'Not found'}")
    print(f"Array Type: {type(audio['array']) if 'array' in audio else 'Not found'}")

if __name__ == "__main__":
    inspect_dataset()
