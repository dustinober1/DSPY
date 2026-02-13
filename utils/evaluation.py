"""
Evaluation utilities for running experiments and collecting metrics.
"""
import time
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, asdict
import dspy
from tqdm import tqdm


@dataclass
class EvaluationResult:
    """Results from evaluating a model"""
    model_name: str
    task: str
    num_examples: int
    num_correct: int
    accuracy: float
    avg_time_per_example: float
    total_time: float
    additional_metrics: Dict[str, Any] = None
    
    def to_dict(self):
        """Convert to dictionary"""
        result = asdict(self)
        if result['additional_metrics'] is None:
            result['additional_metrics'] = {}
        return result


class Evaluator:
    """
    Evaluation harness for running models on datasets.
    """
    
    def __init__(
        self,
        metric_fn: Callable,
        show_progress: bool = True,
        verbose: bool = False,
    ):
        self.metric_fn = metric_fn
        self.show_progress = show_progress
        self.verbose = verbose
    
    def evaluate(
        self,
        model: Any,
        examples: List[Any],
        model_name: str = "model",
        task: str = "task",
    ) -> EvaluationResult:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: Model/module to evaluate (should be callable)
            examples: List of examples to evaluate on
            model_name: Name of model for tracking
            task: Task name
            
        Returns:
            EvaluationResult object
        """
        correct = 0
        predictions = []
        start_time = time.time()
        
        iterator = tqdm(examples, desc=f"Evaluating {model_name}") if self.show_progress else examples
        
        for example in iterator:
            try:
                # Run model
                if hasattr(model, 'forward'):
                    # DSPy module
                    if hasattr(example, 'context'):
                        # HotPotQA-style
                        prediction = model.forward(
                            question=example.question,
                            context=example.context,
                        )
                    else:
                        # GSM8K-style
                        prediction = model.forward(question=example.question)
                else:
                    # Other callable
                    if hasattr(example, 'context'):
                        prediction = model(question=example.question, context=example.context)
                    else:
                        prediction = model(question=example.question)
                
                # Evaluate
                score = self.metric_fn(example, prediction)
                if score > 0.5:  # Consider correct if score > 0.5
                    correct += 1
                
                predictions.append(prediction)
                
                if self.verbose:
                    print(f"\nExample: {example.question[:50]}...")
                    pred_answer = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
                    print(f"Prediction: {pred_answer}")
                    print(f"Correct: {example.answer}")
                    print(f"Score: {score}")
                
            except Exception as e:
                print(f"Error evaluating example: {e}")
                predictions.append(None)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(examples) if examples else 0
        accuracy = correct / len(examples) if examples else 0
        
        result = EvaluationResult(
            model_name=model_name,
            task=task,
            num_examples=len(examples),
            num_correct=correct,
            accuracy=accuracy,
            avg_time_per_example=avg_time,
            total_time=total_time,
            additional_metrics={},
        )
        
        if self.verbose or self.show_progress:
            print(f"\n{'='*60}")
            print(f"Results for {model_name}:")
            print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(examples)})")
            print(f"  Avg time: {avg_time:.2f}s per example")
            print(f"  Total time: {total_time:.1f}s")
            print(f"{'='*60}")
        
        return result
    
    def compare_models(
        self,
        models: Dict[str, Any],
        examples: List[Any],
        task: str = "task",
    ) -> Dict[str, EvaluationResult]:
        """
        Compare multiple models on the same dataset.
        
        Args:
            models: Dict mapping model names to models
            examples: List of examples
            task: Task name
            
        Returns:
            Dict mapping model names to EvaluationResults
        """
        results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {model_name}")
            print(f"{'='*60}")
            
            result = self.evaluate(
                model=model,
                examples=examples,
                model_name=model_name,
                task=task,
            )
            results[model_name] = result
        
        # Print summary
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<30} {'Accuracy':<15} {'Avg Time (s)':<15}")
        print("-" * 60)
        for name, result in results.items():
            print(f"{name:<30} {result.accuracy:>8.1%}      {result.avg_time_per_example:>8.2f}")
        print(f"{'='*60}\n")
        
        return results


def estimate_cost(
    num_examples: int,
    avg_tokens_per_example: int,
    model_name: str = "gpt-3.5-turbo",
) -> Dict[str, float]:
    """
    Estimate API cost for running experiments.
    
    Args:
        num_examples: Number of examples to evaluate
        avg_tokens_per_example: Average tokens per example (input + output)
        model_name: Model name for pricing
        
    Returns:
        Dict with cost estimates
    """
    # Pricing (as of 2024 - approximate)
    pricing = {
        "gpt-3.5-turbo": {"input": 0.0015 / 1000, "output": 0.002 / 1000},
        "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
        "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
    }
    
    if model_name not in pricing:
        model_name = "gpt-3.5-turbo"  # Default fallback
    
    # Estimate (assume 60/40 split input/output)
    input_tokens = num_examples * avg_tokens_per_example * 0.6
    output_tokens = num_examples * avg_tokens_per_example * 0.4
    
    cost = (
        input_tokens * pricing[model_name]["input"] +
        output_tokens * pricing[model_name]["output"]
    )
    
    return {
        "total_cost": cost,
        "cost_per_example": cost / num_examples if num_examples > 0 else 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model_name,
    }


def analyze_errors(
    examples: List[Any],
    predictions: List[Any],
    correct_fn: Callable,
    categorize_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Analyze errors in predictions.
    
    Args:
        examples: List of ground truth examples
        predictions: List of predictions
        correct_fn: Function to check if prediction is correct
        categorize_fn: Optional function to categorize errors
        
    Returns:
        Dict with error analysis
    """
    errors = []
    correct = []
    
    for example, prediction in zip(examples, predictions):
        is_correct = correct_fn(example, prediction)
        
        if is_correct:
            correct.append((example, prediction))
        else:
            error_info = {
                "example": example,
                "prediction": prediction,
                "category": "unknown",
            }
            
            if categorize_fn:
                error_info["category"] = categorize_fn(example, prediction)
            
            errors.append(error_info)
    
    # Count error categories
    error_categories = {}
    for error in errors:
        cat = error["category"]
        error_categories[cat] = error_categories.get(cat, 0) + 1
    
    return {
        "num_errors": len(errors),
        "num_correct": len(correct),
        "accuracy": len(correct) / len(examples) if examples else 0,
        "error_categories": error_categories,
        "sample_errors": errors[:5],  # First 5 errors
        "sample_correct": correct[:5],  # First 5 correct
    }


__all__ = [
    "EvaluationResult",
    "Evaluator",
    "estimate_cost",
    "analyze_errors",
]
