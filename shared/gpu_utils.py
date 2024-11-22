"""
GPU-bezogene Utilities für FabricaModelli
"""

import os
import torch
import torch.distributed as dist
import logging

logger = logging.getLogger("FabricaModelli")

def setup_gpu_environment():
    """
    Konfiguriert die GPU-Umgebung für optimale Performance.
    """
    # CUDA-Umgebungsvariablen
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if torch.cuda.is_available():
        # Setze für reproduzierbare Ergebnisse
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Log GPU-Informationen
        gpu_count = torch.cuda.device_count()
        logger.info(f"Gefundene GPUs: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {gpu_name} ({memory:.1f}GB VRAM)")
    else:
        logger.warning("Keine CUDA-fähige GPU gefunden!")

def setup_distributed_training(port="12355"):
    """
    Richtet verteiltes Training ein.
    
    Args:
        port: Port für die Kommunikation zwischen Prozessen
    
    Returns:
        tuple: (local_rank, world_size)
    """
    if torch.cuda.device_count() > 1:
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["LOCAL_RANK"])
        else:
            world_size = torch.cuda.device_count()
            rank = 0
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = port
            
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size
            )
            
        logger.info(f"Verteiltes Training initialisiert: Rank {rank}/{world_size-1}")
        return rank, world_size
    else:
        logger.info("Einzelne GPU gefunden, kein verteiltes Training notwendig")
        return 0, 1

def cleanup_distributed():
    """
    Räumt verteiltes Training auf.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
