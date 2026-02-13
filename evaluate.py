#!/usr/bin/env python3
"""
Command-line evaluation script for running experiments outside of notebooks.

Usage:
    python evaluate.py --task gsm8k --approach dspy --subset dev
    python evaluate.py --task hotpotqa --approach zero-shot --subset test
"""
import argparse
from pathlib import Path

import dspy
from config import (
    DATASET_CONFIGS,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_API_BASE,
    OLLAMA_MAX_TOKENS,
    OLLAMA_TEMPERATURE,
)
from data import (
    prepare_gsm8k_splits,
    prepare_hotpotqa_splits,
    gsm8k_metric,
    hotpotqa_metric,
)
from modules import get_module
from baselines import create_baseline
from optimizers import create_optimizer
from utils import Evaluator, plot_accuracy_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Run DSPy experiments")
    parser.add_argument(
        "--task",
        choices=["gsm8k", "hotpotqa", "both"],
        default="gsm8k",
        help="Task to evaluate",
    )
    parser.add_argument(
        "--approach",
        choices=["zero-shot", "few-shot", "dspy", "all"],
        default="all",
        help="Approach to use",
    )
    parser.add_argument(
        "--subset",
        choices=["train", "dev", "test"],
        default="dev",
        help="Dataset subset",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model name, with or without 'ollama/' prefix",
    )
    parser.add_argument(
        "--api-base",
        default=OLLAMA_API_BASE,
        help="Ollama API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (typically not needed for local Ollama)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=OLLAMA_MAX_TOKENS,
        help="Max tokens per generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=OLLAMA_TEMPERATURE,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Limit number of examples (for quick testing)",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to file",
    )
    parser.add_argument(
        "--load-optimized",
        type=Path,
        default=None,
        help="Path to load pre-optimized program",
    )
    return parser.parse_args()


def load_data(task, subset, num_examples=None):
    """Load dataset for a task"""
    if task == "gsm8k":
        config = DATASET_CONFIGS["gsm8k"]
        train, dev, test = prepare_gsm8k_splits(
            train_size=config["train_size"],
            dev_size=config["dev_size"],
            test_size=config["test_size"],
            seed=config["seed"],
        )
        splits = {"train": train, "dev": dev, "test": test}
        metric = gsm8k_metric
        
    elif task == "hotpotqa":
        config = DATASET_CONFIGS["hotpotqa"]
        train, dev, test = prepare_hotpotqa_splits(
            train_size=config["train_size"],
            dev_size=config["dev_size"],
            test_size=config["test_size"],
            setting=config["setting"],
            seed=config["seed"],
        )
        splits = {"train": train, "dev": dev, "test": test}
        metric = hotpotqa_metric
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    examples = splits[subset]
    if num_examples:
        examples = examples[:num_examples]
    
    return examples, splits["train"], metric


def run_evaluation(task, approach, examples, train_examples, metric, lm, load_path=None):
    """Run evaluation for a specific approach"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {approach.upper()} on {task.upper()}")
    print(f"{'='*80}\n")
    
    evaluator = Evaluator(metric_fn=metric, show_progress=True, verbose=False)
    
    if approach == "zero-shot":
        model = create_baseline("zero-shot", task, lm)
        model_name = f"Zero-Shot"
        
    elif approach == "few-shot":
        model = create_baseline("few-shot", task, lm, num_examples=3)
        model_name = f"Few-Shot"
        
    elif approach == "dspy":
        # Create module
        module = get_module(task, module_type="default")
        
        # Load or optimize
        if load_path and load_path.exists():
            print(f"Loading optimized program from {load_path}")
            optimizer = create_optimizer("bootstrap", metric, teacher_lm=lm)
            model = optimizer.load(load_path, type(module))
        else:
            print("Optimizing with DSPy (this may take a few minutes)...")
            optimizer = create_optimizer("bootstrap", metric, teacher_lm=lm)
            model = optimizer.compile(
                module,
                trainset=train_examples[:50],  # Use subset for speed
            )
        
        model_name = "DSPy-Optimized"
    
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    # Evaluate
    result = evaluator.evaluate(
        model=model,
        examples=examples,
        model_name=model_name,
        task=task,
    )
    
    return result


def main():
    args = parse_args()
    
    # Configure model
    print(f"Configuring model: {args.model}")
    model_name = args.model if "/" in args.model else f"ollama/{args.model}"
    lm = dspy.LM(
        model=model_name,
        api_base=args.api_base,
        api_key=args.api_key or None,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    dspy.settings.configure(lm=lm)
    
    # Determine tasks
    tasks = ["gsm8k", "hotpotqa"] if args.task == "both" else [args.task]
    
    # Determine approaches
    approaches = ["zero-shot", "few-shot", "dspy"] if args.approach == "all" else [args.approach]
    
    # Run experiments
    all_results = {}
    
    for task in tasks:
        print(f"\n{'#'*80}")
        print(f"# TASK: {task.upper()}")
        print(f"{'#'*80}")
        
        # Load data
        examples, train_examples, metric = load_data(
            task, args.subset, args.num_examples
        )
        
        print(f"\nDataset: {len(examples)} {args.subset} examples")
        
        task_results = {}
        
        for approach in approaches:
            result = run_evaluation(
                task, approach, examples, train_examples, metric, lm, args.load_optimized
            )
            task_results[f"{approach}"] = result.accuracy
        
        all_results[task] = task_results
    
    # Print summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}\n")
    
    for task, results in all_results.items():
        print(f"\n{task.upper()}:")
        for approach, accuracy in results.items():
            print(f"  {approach:20s}: {accuracy:>6.1%}")
    
    print(f"\n{'='*80}\n")
    
    # Save if requested
    if args.save_results:
        import json
        from config import RESULTS_DIR
        
        output_file = RESULTS_DIR / f"results_{args.task}_{args.subset}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
