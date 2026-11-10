import os
import sys
import psutil
import torch
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    model_path: str = os.getenv("MODEL_PATH")
    affinity_cores: list[int] = [int(c.strip()) for c in os.getenv("CPU_AFFINITY_CORES", "0,1,2").split(",")]
    worker_threads: int = int(os.getenv("CPU_WORKER_THREADS", "3"))
    niceness: int = int(os.getenv("NICENESS_LEVEL", "10"))
    ionice_class: int = int(os.getenv("IONICE_CLASS", "3"))
    matryoshka_dim: int = int(os.getenv("MATRYOSHKA_DIM", "256"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    db_path: str = os.getenv("LANCEDB_PATH", "./data/db")
    ingest_path: str = os.getenv("DATA_INGEST_PATH", "./data/raw")

settings = Settings()

def apply_hardware_governor():
    torch.set_num_threads(settings.worker_threads)
    os.environ["OMP_NUM_THREADS"] = str(settings.worker_threads)
    os.environ["MKL_NUM_THREADS"] = str(settings.worker_threads)
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    p = psutil.Process()
    try:
        p.cpu_affinity(settings.affinity_cores)
    except Exception as e:
        sys.stderr.write(f"[WARN] Unable to set CPU affinity: {e}\n")

    try:
        os.nice(settings.niceness)
    except Exception as e:
        sys.stderr.write(f"[WARN] Unable to set nice level: {e}\n")

    try:
        p.ionice(ioclass=psutil.IOPRIO_CLASS_IDLE)
    except Exception as e:
        sys.stderr.write(f"[WARN] Unable to set ionice: {e}\n")
