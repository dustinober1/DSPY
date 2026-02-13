"""
HotPotQA Dataset Loader and Preprocessor

Load the HotPotQA dataset for multi-hop question answering.
Handles retrieval corpus creation, answer extraction, and metrics.
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import re
import string
from datasets import load_dataset
import dspy
from rank_bm25 import BM25Okapi
from config import DATA_DIR, DATASET_CONFIGS


class HotPotQAExample(dspy.Example):
    """A HotPotQA example with question, context, and answer"""
    def __init__(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
        supporting_facts: Optional[List[Tuple[str, int]]] = None,
    ):
        super().__init__(
            question=question,
            answer=answer,
            context=context or "",
            supporting_facts=supporting_facts or [],
        )


def normalize_answer(s: str) -> str:
    """
    Normalize answer for evaluation (following official HotPotQA evaluation).
    Lower case and remove punctuation, articles and extra whitespace.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match score (0 or 1)"""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score between prediction and ground truth.
    Following official HotPotQA evaluation.
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    
    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common_tokens.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def load_hotpotqa_split(
    split: str = "train",
    setting: str = "distractor",
    num_examples: Optional[int] = None,
    seed: int = 42,
) -> List[HotPotQAExample]:
    """
    Load HotPotQA dataset split.
    
    Args:
        split: 'train' or 'validation'
        setting: 'distractor' (10 paragraphs, 2 gold) or 'fullwiki' (full wikipedia)
        num_examples: Number of examples to sample
        seed: Random seed
        
    Returns:
        List of HotPotQAExample objects
    """
    print(f"Loading HotPotQA {split} split ({setting} setting)...")
    
    # Load from HuggingFace
    dataset = load_dataset("hotpot_qa", setting, split=split)
    
    # Sample if requested
    if num_examples is not None and num_examples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(num_examples))
    
    # Convert to DSPy examples
    examples = []
    for item in dataset:
        question = item["question"]
        answer = item["answer"]
        
        # Build context from provided paragraphs
        context_parts = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            paragraph = " ".join(sentences)
            context_parts.append(f"{title}: {paragraph}")
        context = "\n\n".join(context_parts)
        
        # Extract supporting facts (title, sentence_id pairs)
        supporting_facts = list(zip(
            item["supporting_facts"]["title"],
            item["supporting_facts"]["sent_id"],
        ))
        
        example = HotPotQAExample(
            question=question,
            answer=answer,
            context=context,
            supporting_facts=supporting_facts,
        )
        examples.append(example)
    
    print(f"Loaded {len(examples)} examples from HotPotQA {split}")
    return examples


def prepare_hotpotqa_splits(
    train_size: int = 150,
    dev_size: int = 75,
    test_size: int = 75,
    setting: str = "distractor",
    seed: int = 42,
    save_dir: Optional[Path] = None,
) -> Tuple[List[HotPotQAExample], List[HotPotQAExample], List[HotPotQAExample]]:
    """
    Prepare train/dev/test splits for HotPotQA.
    
    Args:
        train_size: Number of training examples
        dev_size: Number of dev examples (from validation split)
        test_size: Number of test examples (from validation split)
        setting: 'distractor' or 'fullwiki'
        seed: Random seed
        save_dir: Directory to save prepared splits
        
    Returns:
        (train_examples, dev_examples, test_examples)
    """
    # Load training examples
    train_examples = load_hotpotqa_split(
        "train",
        setting=setting,
        num_examples=train_size,
        seed=seed,
    )
    
    # Load validation and split into dev/test
    all_val = load_hotpotqa_split(
        "validation",
        setting=setting,
        num_examples=dev_size + test_size,
        seed=seed,
    )
    dev_examples = all_val[:dev_size]
    test_examples = all_val[dev_size:dev_size + test_size]
    
    print(f"\nPrepared HotPotQA splits:")
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
            filepath = save_dir / f"hotpotqa_{split_name}.json"
            with open(filepath, "w") as f:
                data = [
                    {
                        "question": ex.question,
                        "answer": ex.answer,
                        "context": ex.context,
                        "supporting_facts": ex.supporting_facts,
                    }
                    for ex in examples
                ]
                json.dump(data, f, indent=2)
            print(f"Saved {split_name} to {filepath}")
    
    return train_examples, dev_examples, test_examples


def create_bm25_retriever(corpus: List[HotPotQAExample]) -> BM25Okapi:
    """
    Create a BM25 retriever from a corpus of examples.
    
    Args:
        corpus: List of HotPotQAExample objects with context
        
    Returns:
        BM25Okapi retriever
    """
    # Tokenize contexts
    tokenized_corpus = [
        example.context.split()
        for example in corpus
    ]
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    return bm25


def retrieve_context(
    query: str,
    bm25: BM25Okapi,
    corpus: List[HotPotQAExample],
    top_k: int = 3,
) -> str:
    """
    Retrieve relevant context for a query using BM25.
    
    Args:
        query: Question to retrieve context for
        bm25: BM25 retriever
        corpus: Original corpus
        top_k: Number of contexts to retrieve
        
    Returns:
        Combined context string
    """
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    
    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    # Combine contexts
    contexts = [corpus[i].context for i in top_indices]
    return "\n\n".join(contexts)


def hotpotqa_metric(example: HotPotQAExample, prediction: dspy.Prediction, trace=None) -> float:
    """
    Metric function for DSPy optimization.
    Returns F1 score.
    """
    pred_answer = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
    return compute_f1(pred_answer, example.answer)


def evaluate_hotpotqa(
    examples: List[HotPotQAExample],
    predictions: List[str],
) -> Dict[str, float]:
    """
    Evaluate HotPotQA predictions.
    
    Returns:
        Dictionary with metrics: em (exact match), f1, num_total
    """
    assert len(examples) == len(predictions), "Mismatch in examples and predictions"
    
    em_scores = []
    f1_scores = []
    
    for example, prediction in zip(examples, predictions):
        em = compute_exact_match(prediction, example.answer)
        f1 = compute_f1(prediction, example.answer)
        em_scores.append(em)
        f1_scores.append(f1)
    
    return {
        "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "num_total": len(examples),
    }


def show_example(example: HotPotQAExample, prediction: Optional[str] = None):
    """Pretty print a HotPotQA example"""
    print("=" * 80)
    print("QUESTION:")
    print(example.question)
    print("\nCONTEXT (truncated):")
    print(example.context[:500] + "..." if len(example.context) > 500 else example.context)
    print("\nCORRECT ANSWER:")
    print(example.answer)
    if prediction:
        print("\nPREDICTION:")
        print(prediction)
        em = compute_exact_match(prediction, example.answer)
        f1 = compute_f1(prediction, example.answer)
        print(f"\nEM: {em:.2f}, F1: {f1:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    # Example usage
    config = DATASET_CONFIGS["hotpotqa"]
    
    train, dev, test = prepare_hotpotqa_splits(
        train_size=config["train_size"],
        dev_size=config["dev_size"],
        test_size=config["test_size"],
        setting=config["setting"],
        seed=config["seed"],
        save_dir=DATA_DIR / "hotpotqa",
    )
    
    # Show a sample
    print("\n" + "="*80)
    print("SAMPLE EXAMPLE:")
    show_example(train[0])
    
    print("\n" + "="*80)
    print("Test metrics:")
    print(f"F1('The President', 'The President'): {compute_f1('The President', 'The President')}")
    print(f"F1('The President', 'president'): {compute_f1('The President', 'president')}")
    print(f"F1('Barack Obama', 'Obama'): {compute_f1('Barack Obama', 'Obama'):.2f}")
