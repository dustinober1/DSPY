#!/usr/bin/env python3
"""
Pull Ollama models for the DSPy demo.

Legacy HuggingFace aliases like `all-small` and `all` are still accepted for
backward compatibility with existing setup scripts.
"""
import argparse
import shutil
import subprocess
from config import DEFAULT_OLLAMA_MODEL


DEFAULT_OLLAMA_MODELS = [DEFAULT_OLLAMA_MODEL]
LEGACY_ALIASES = {
    "all-small": DEFAULT_OLLAMA_MODELS,
    "all": DEFAULT_OLLAMA_MODELS,
}


def resolve_models(requested_models):
    resolved = []
    for model in requested_models:
        if model in LEGACY_ALIASES:
            resolved.extend(LEGACY_ALIASES[model])
        else:
            resolved.append(model)

    # De-duplicate while preserving order.
    unique = []
    for model in resolved:
        if model not in unique:
            unique.append(model)
    return unique


def pull_model(model_name: str) -> bool:
    print(f"\n{'='*60}")
    print(f"Pulling Ollama model: {model_name}")
    print(f"{'='*60}")

    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"✓ Successfully pulled: {model_name}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"✗ Failed to pull {model_name}: exit code {exc.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Pull Ollama models for DSPy demo")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_OLLAMA_MODEL],
        help="Ollama model names to pull, or legacy aliases: all-small, all",
    )
    args = parser.parse_args()

    if shutil.which("ollama") is None:
        raise SystemExit(
            "Ollama CLI not found. Install Ollama first: https://ollama.com/download"
        )

    models_to_pull = resolve_models(args.models)

    print(f"\n{'='*60}")
    print(f"Will pull {len(models_to_pull)} model(s)")
    print(f"{'='*60}")

    failures = 0
    for model_name in models_to_pull:
        ok = pull_model(model_name)
        if not ok:
            failures += 1

    print("\n" + "="*60)
    if failures:
        print(f"Completed with {failures} failure(s).")
    else:
        print("Pull process complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Verify model list: ollama list")
    print("2. Set up data: python data/gsm8k_loader.py")
    print("3. Start experimenting with the notebooks!")


if __name__ == "__main__":
    main()
