"""
GSM8K Dataset Loader and Preprocessor

Load the GSM8K (Grade School Math 8K) dataset for math word problem solving.
Handles data splitting, answer extraction, and metric computation.
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
import dspy
from config import DATA_DIR, DATASET_CONFIGS


class GSM8KExample(dspy.Example):
    """A GSM8K example with question and answer"""
    pass


def extract_numeric_answer(answer_text: str) -> str:
    """
    Extract the final numeric answer from GSM8K answer format.
    
    GSM8K answers are in format: "reasoning steps\n#### ANSWER"
    We extract just the numeric value.
    """
    # Try to find the #### delimiter
    if "####" in answer_text:
        answer = answer_text.split("####")[-1].strip()
    else:
        answer = answer_text.strip()
    
    # Remove commas from numbers (e.g., "1,000" -> "1000")
    answer = answer.replace(",", "")
    
    # Extract just the number (handle negative numbers, decimals)
    match = re.search(r'-?\d+\.?\d*', answer)
    if match:
        answer = match.group(0)
    
    return answer


def extract_reasoning(answer_text: str) -> str:
    """Extract the reasoning steps from GSM8K answer"""
    if "####" in answer_text:
        return answer_text.split("####")[0].strip()
    return answer_text.strip()


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison"""
    answer = str(answer).strip().lower()
    answer = answer.replace(",", "")
    
    # Try to convert to float then back to handle 1.0 vs 1
    try:
        num = float(answer)
        # If it's a whole number, return as int
        if num.is_integer():
            return str(int(num))
        return str(num)
    except (ValueError, AttributeError):
        return answer


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Check if prediction exactly matches ground truth after normalization"""
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    return pred_norm == gt_norm


def load_gsm8k_split(
    split: str = "train",
    num_examples: Optional[int] = None,
    seed: int = 42,
) -> List[dspy.Example]:
    """
    Load GSM8K dataset split and convert to DSPy examples.
    
    Args:
        split: 'train' or 'test'
        num_examples: Number of examples to sample (None for all)
        seed: Random seed for sampling
        
    Returns:
        List of dspy.Example objects
    """
    print(f"Loading GSM8K {split} split...")
    
    # Load from HuggingFace
    dataset = load_dataset("gsm8k", "main", split=split)
    
    # Sample if requested
    if num_examples is not None and num_examples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(num_examples))
    
    # Convert to DSPy examples
    examples = []
    for item in dataset:
        question = item["question"]
        full_answer = item["answer"]
        
        # Extract numeric answer and reasoning
        numeric_answer = extract_numeric_answer(full_answer)
        reasoning_steps = extract_reasoning(full_answer)
        
        example = dspy.Example(
            question=question,
            answer=numeric_answer,
            reasoning=reasoning_steps,
        ).with_inputs("question")
        examples.append(example)
    
    print(f"Loaded {len(examples)} examples from GSM8K {split}")
    return examples


def prepare_gsm8k_splits(
    train_size: int = 200,
    dev_size: int = 100,
    test_size: int = 100,
    seed: int = 42,
    save_dir: Optional[Path] = None,
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """
    Prepare train/dev/test splits for GSM8K.
    
    Since GSM8K only has train/test splits, we'll:
    - Use a subset of train for our training
    - Split some train examples for dev
    - Use official test for our test set
    
    Args:
        train_size: Number of training examples
        dev_size: Number of dev examples
        test_size: Number of test examples
        seed: Random seed
        save_dir: Directory to save prepared splits (optional)
        
    Returns:
        (train_examples, dev_examples, test_examples)
    """
    # Load all training data
    all_train = load_gsm8k_split("train", num_examples=train_size + dev_size, seed=seed)
    
    # Split into train and dev
    train_examples = all_train[:train_size]
    dev_examples = all_train[train_size:train_size + dev_size]
    
    # Load test set
    test_examples = load_gsm8k_split("test", num_examples=test_size, seed=seed)
    
    print(f"\nPrepared GSM8K splits:")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Dev:   {len(dev_examples)} examples")
    print(f"  Test:  {len(test_examples)} examples")
    
    # Save if requested
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, examples in [
            ("train", train_examples),
            ("dev", dev_examples),
            ("test", test_examples),
        ]:
            filepath = save_dir / f"gsm8k_{split_name}.json"
            with open(filepath, "w") as f:
                data = [
                    {
                        "question": ex.question,
                        "answer": ex.answer,
                        "reasoning": ex.reasoning,
                    }
                    for ex in examples
                ]
                json.dump(data, f, indent=2)
            print(f"Saved {split_name} to {filepath}")
    
    return train_examples, dev_examples, test_examples


def gsm8k_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    Metric function for DSPy optimization.
    Returns 1.0 if exact match, 0.0 otherwise.
    """
    # Extract answer from prediction
    pred_answer = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
    
    # Normalize and compare
    return 1.0 if compute_exact_match(pred_answer, example.answer) else 0.0


def evaluate_gsm8k(
    examples: List[dspy.Example],
    predictions: List[str],
) -> Dict[str, float]:
    """
    Evaluate GSM8K predictions.
    
    Returns:
        Dictionary with metrics: accuracy, num_correct, num_total
    """
    assert len(examples) == len(predictions), "Mismatch in number of examples and predictions"
    
    correct = 0
    for example, prediction in zip(examples, predictions):
        if compute_exact_match(prediction, example.answer):
            correct += 1
    
    accuracy = correct / len(examples) if examples else 0.0
    
    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": len(examples),
    }


def show_example(example: dspy.Example, prediction: Optional[str] = None):
    """Pretty print a GSM8K example"""
    print("=" * 80)
    print("QUESTION:")
    print(example.question)
    print("\nCORRECT ANSWER:")
    print(example.answer)
    if prediction:
        print("\nPREDICTION:")
        print(prediction)
        is_correct = compute_exact_match(prediction, example.answer)
        print(f"\n{'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    print("=" * 80)


if __name__ == "__main__":
    # Example usage
    config = DATASET_CONFIGS["gsm8k"]
    
    train, dev, test = prepare_gsm8k_splits(
        train_size=config["train_size"],
        dev_size=config["dev_size"],
        test_size=config["test_size"],
        seed=config["seed"],
        save_dir=DATA_DIR / "gsm8k",
    )
    
    # Show a sample
    print("\n" + "="*80)
    print("SAMPLE EXAMPLE:")
    show_example(train[0])
    
    print("\n" + "="*80)
    print("Test answer extraction:")
    test_answer = "First calc: 5 + 3 = 8\nThen: 8 * 2 = 16\n#### 16"
    extracted = extract_numeric_answer(test_answer)
    print(f"Full answer: {test_answer}")
    print(f"Extracted: {extracted}")
