#!/usr/bin/env python3
"""
Download and prepare models for the DSPy demo.

This script helps download models from HuggingFace and verify they're ready to use.
For gated models (like Llama-2), you'll need to set HF_TOKEN environment variable.
"""
import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, login
from config import SMALL_MODELS, LARGE_MODELS, MODELS_DIR, HF_TOKEN


def download_model(model_path: str, cache_dir: Path, token: str = None):
    """Download a model from HuggingFace"""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_path}")
    print(f"{'='*60}")
    
    try:
        local_path = snapshot_download(
            repo_id=model_path,
            cache_dir=cache_dir,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Skip unnecessary files
        )
        print(f"✓ Successfully downloaded to: {local_path}")
        return local_path
    except Exception as e:
        print(f"✗ Failed to download {model_path}: {e}")
        if "gated" in str(e).lower() or "access" in str(e).lower():
            print("\nThis model requires authentication. Please:")
            print("1. Visit https://huggingface.co/" + model_path)
            print("2. Accept the model license")
            print("3. Set HF_TOKEN environment variable with your HuggingFace token")
        return None


def check_disk_space(required_gb: float = 50):
    """Check if enough disk space is available"""
    import shutil
    stat = shutil.disk_usage(MODELS_DIR)
    available_gb = stat.free / (1024**3)
    
    print(f"\nDisk space available: {available_gb:.1f} GB")
    if available_gb < required_gb:
        print(f"⚠ Warning: Less than {required_gb} GB available. Model downloads may fail.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Download models for DSPy demo")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mistral-7b", "llama-7b", "llama-70b", "mixtral-8x7b", "all-small", "all"],
        default=["all-small"],
        help="Which models to download (default: all-small)",
    )
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Skip disk space verification",
    )
    args = parser.parse_args()
    
    # Login to HuggingFace if token is available
    if HF_TOKEN:
        print("Logging in to HuggingFace...")
        login(token=HF_TOKEN)
        print("✓ Logged in successfully")
    else:
        print("⚠ No HF_TOKEN found. You may not be able to download gated models.")
    
    # Check disk space
    if not args.skip_disk_check:
        print("\nChecking disk space...")
        check_disk_space()
    
    # Determine which models to download
    models_to_download = {}
    
    if "all" in args.models:
        models_to_download.update(SMALL_MODELS)
        models_to_download.update(LARGE_MODELS)
    elif "all-small" in args.models:
        models_to_download.update(SMALL_MODELS)
    else:
        for model_key in args.models:
            if model_key in SMALL_MODELS:
                models_to_download[model_key] = SMALL_MODELS[model_key]
            elif model_key in LARGE_MODELS:
                models_to_download[model_key] = LARGE_MODELS[model_key]
    
    # Download each model
    print(f"\n{'='*60}")
    print(f"Will download {len(models_to_download)} model(s)")
    print(f"{'='*60}\n")
    
    for model_key, model_config in models_to_download.items():
        print(f"\nModel: {model_key}")
        print(f"Path: {model_config.model_path}")
        
        # Estimate size
        size_estimates = {
            "7b": "~14 GB",
            "13b": "~26 GB",
            "70b": "~140 GB",
            "8x7b": "~90 GB",
        }
        size_est = next((v for k, v in size_estimates.items() if k in model_key), "Unknown")
        print(f"Estimated size: {size_est}")
        
        download_model(
            model_config.model_path,
            MODELS_DIR,
            token=HF_TOKEN,
        )
    
    print("\n" + "="*60)
    print("Download process complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Verify models are working: python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(\"mistralai/Mistral-7B-Instruct-v0.2\")'")
    print("2. Set up data: python data/gsm8k_loader.py")
    print("3. Start experimenting with the notebooks!")


if __name__ == "__main__":
    main()
