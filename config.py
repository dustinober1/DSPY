"""
Configuration for DSPy Small-to-SOTA Model Demo
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "dspy_cache"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, CACHE_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for a language model"""
    name: str
    model_path: str
    max_tokens: int = 2048
    temperature: float = 0.0
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9


# Small model configurations (main demo models)
SMALL_MODELS = {
    "phi-2": ModelConfig(
        name="phi-2",
        model_path="microsoft/phi-2",
        max_tokens=2048,
    ),
    "tinyllama": ModelConfig(
        name="tinyllama-1.1b",
        model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_tokens=2048,
    ),
}

# Reference "SOTA" model (for comparison - using published benchmarks if not available locally)
LARGE_MODELS = {
    "llama-70b": ModelConfig(
        name="llama-2-70b-chat",
        model_path="meta-llama/Llama-2-70b-chat-hf",
        max_tokens=2048,
        tensor_parallel_size=4,  # Requires multi-GPU
    ),
    "mixtral-8x7b": ModelConfig(
        name="mixtral-8x7b-instruct",
        model_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_tokens=2048,
        tensor_parallel_size=2,
    ),
}

# Default models to use
DEFAULT_SMALL_MODEL = "phi-2"
DEFAULT_LARGE_MODEL = "llama-70b"  # May use benchmark results instead of actual inference

# Dataset configurations
DATASET_CONFIGS = {
    "gsm8k": {
        "train_size": 200,
        "dev_size": 100,
        "test_size": 100,
        "seed": 42,
    },
    "hotpotqa": {
        "train_size": 150,
        "dev_size": 75,
        "test_size": 75,
        "seed": 42,
        "setting": "distractor",  # fullwiki or distractor
    },
}

# DSPy optimization configurations
OPTIMIZER_CONFIGS = {
    "bootstrap_fewshot": {
        "max_bootstrapped_demos": 8,
        "max_labeled_demos": 8,
    },
    "bootstrap_random_search": {
        "max_bootstrapped_demos": 8,
        "max_labeled_demos": 8,
        "num_candidate_programs": 16,
        "num_threads": 4,
    },
    "mipro": {
        "num_candidates": 10,
        "init_temperature": 1.0,
    },
}

# Evaluation settings
EVAL_BATCH_SIZE = 32
EVAL_NUM_THREADS = 4

# vLLM server settings (for fast local inference)
VLLM_CONFIG = {
    "host": "localhost",
    "port": 8000,
    "trust_remote_code": True,
    "dtype": "auto",
    "max_model_len": 2048,
}

# Retrieval settings for HotPotQA
RETRIEVAL_CONFIG = {
    "top_k": 10,
    "method": "bm25",  # or "embedding"
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
}

# Hardware detection
def get_device_config():
    """Detect available hardware and configure accordingly"""
    import torch
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "device": "cuda",
            "num_gpus": num_gpus,
            "gpu_memory_gb": gpu_memory,
        }
    elif torch.backends.mps.is_available():
        return {
            "device": "mps",
            "num_gpus": 1,
            "gpu_memory_gb": "unknown",
        }
    else:
        return {
            "device": "cpu",
            "num_gpus": 0,
            "gpu_memory_gb": 0,
        }


# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API keys (optional - for fallback to API models)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # For gated models like Llama-2
