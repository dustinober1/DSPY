"""
Baseline implementations for comparison.

Zero-shot and manual few-shot approaches without DSPy optimization.
"""
import dspy
from typing import List, Dict, Any, Optional
from data import GSM8KExample, HotPotQAExample


# ==================== Zero-Shot Baselines ====================

class ZeroShotMathSolver:
    """Zero-shot math solver with minimal prompting"""
    
    def __init__(self, lm: dspy.LM):
        self.lm = lm
    
    def __call__(self, question: str) -> str:
        """Solve a math problem with zero-shot prompting"""
        prompt = f"""Solve this math problem and provide the final numerical answer.

Question: {question}

Let's solve this step by step and provide the final answer as a number.

Answer:"""
        
        with dspy.context(lm=self.lm):
            response = self.lm(prompt)
        
        # Extract answer from response (simple heuristic)
        # Look for numbers in the response
        import re
        numbers = re.findall(r'-?\d+\.?\d*', str(response))
        return numbers[-1] if numbers else "0"


class ZeroShotQA:
    """Zero-shot QA with minimal prompting"""
    
    def __init__(self, lm: dspy.LM):
        self.lm = lm
    
    def __call__(self, question: str, context: str) -> str:
        """Answer a question given context"""
        prompt = f"""Answer the following question based on the context provided.

Context: {context}

Question: {question}

Answer:"""
        
        with dspy.context(lm=self.lm):
            response = self.lm(prompt)
        
        # Return the response (first line typically)
        return str(response).strip().split('\n')[0]


# ==================== Manual Few-Shot Baselines ====================

# Hand-crafted examples for GSM8K
GSM8K_MANUAL_EXAMPLES = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "reasoning": "Janet's ducks lay 16 eggs per day. She eats 3 eggs for breakfast. She uses 4 eggs for muffins. So she has 16 - 3 - 4 = 9 eggs left to sell. She sells each egg for $2. So she makes 9 × $2 = $18 per day.",
        "answer": "18"
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "reasoning": "The robe takes 2 bolts of blue fiber. It takes half that much white fiber, which is 2 ÷ 2 = 1 bolt of white fiber. In total, it takes 2 + 1 = 3 bolts.",
        "answer": "3"
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "reasoning": "Josh bought the house for $80,000. He spent $50,000 on repairs. So his total cost is $80,000 + $50,000 = $130,000. The house value increased by 150%, so the new value is $80,000 + ($80,000 × 1.5) = $80,000 + $120,000 = $200,000. His profit is $200,000 - $130,000 = $70,000.",
        "answer": "70000"
    },
]

# Hand-crafted examples for HotPotQA
HOTPOTQA_MANUAL_EXAMPLES = [
    {
        "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        "context": "Kiss and Tell is a 1945 American comedy film starring then-17-year-old Shirley Temple as Corliss Archer. Shirley Temple Black was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938. As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.",
        "reasoning": "The film Kiss and Tell starred Shirley Temple as Corliss Archer. Shirley Temple later became a diplomat and held several government positions. She served as United States ambassador to Ghana and to Czechoslovakia.",
        "answer": "United States ambassador"
    },
    {
        "question": "Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?",
        "context": "Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg about the civil rights leader. The Saimaa Gesture is a 1981 film by Finnish director Aki Kaurismäki. It is a documentary about three Finnish rock groups.",
        "reasoning": "We need to determine which documentary is about Finnish rock groups. Adam Clayton Powell is a documentary about a civil rights leader. The Saimaa Gesture is a documentary about three Finnish rock groups.",
        "answer": "The Saimaa Gesture"
    },
]


class ManualFewShotMathSolver:
    """Math solver with hand-crafted few-shot examples"""
    
    def __init__(self, lm: dspy.LM, num_examples: int = 3):
        self.lm = lm
        self.examples = GSM8K_MANUAL_EXAMPLES[:num_examples]
    
    def __call__(self, question: str) -> str:
        """Solve a math problem with few-shot prompting"""
        # Build prompt with examples
        prompt = "Solve these math problems step by step:\n\n"
        
        for i, ex in enumerate(self.examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Reasoning: {ex['reasoning']}\n"
            prompt += f"Answer: {ex['answer']}\n\n"
        
        prompt += f"Now solve this problem:\n"
        prompt += f"Question: {question}\n"
        prompt += f"Reasoning:"
        
        with dspy.context(lm=self.lm):
            response = self.lm(prompt)
        
        # Extract answer
        import re
        numbers = re.findall(r'-?\d+\.?\d*', str(response))
        return numbers[-1] if numbers else "0"


class ManualFewShotQA:
    """QA with hand-crafted few-shot examples"""
    
    def __init__(self, lm: dspy.LM, num_examples: int = 2):
        self.lm = lm
        self.examples = HOTPOTQA_MANUAL_EXAMPLES[:num_examples]
    
    def __call__(self, question: str, context: str) -> str:
        """Answer a question with few-shot prompting"""
        # Build prompt with examples
        prompt = "Answer questions based on the given context:\n\n"
        
        for i, ex in enumerate(self.examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Context: {ex['context']}\n"
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Answer: {ex['answer']}\n\n"
        
        prompt += f"Now answer this question:\n"
        prompt += f"Context: {context}\n"
        prompt += f"Question: {question}\n"
        prompt += f"Answer:"
        
        with dspy.context(lm=self.lm):
            response = self.lm(prompt)
        
        return str(response).strip().split('\n')[0]


# ==================== Baseline Runners ====================

def run_baseline(
    baseline,
    examples: List[Any],
    task: str = "gsm8k",
) -> List[str]:
    """
    Run a baseline on a list of examples.
    
    Args:
        baseline: Baseline model (zero-shot or few-shot)
        examples: List of examples to evaluate
        task: Task type ("gsm8k" or "hotpotqa")
        
    Returns:
        List of predicted answers
    """
    predictions = []
    
    for example in examples:
        if task == "gsm8k":
            pred = baseline(question=example.question)
        elif task == "hotpotqa":
            pred = baseline(question=example.question, context=example.context)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        predictions.append(pred)
    
    return predictions


def create_baseline(
    baseline_type: str,
    task: str,
    lm: dspy.LM,
    **kwargs,
):
    """
    Factory function to create baselines.
    
    Args:
        baseline_type: "zero-shot" or "few-shot"
        task: "gsm8k" or "hotpotqa"
        lm: Language model
        **kwargs: Additional arguments
        
    Returns:
        Baseline instance
    """
    if baseline_type == "zero-shot":
        if task == "gsm8k":
            return ZeroShotMathSolver(lm)
        elif task == "hotpotqa":
            return ZeroShotQA(lm)
    
    elif baseline_type == "few-shot":
        if task == "gsm8k":
            return ManualFewShotMathSolver(lm, **kwargs)
        elif task == "hotpotqa":
            return ManualFewShotQA(lm, **kwargs)
    
    raise ValueError(f"Unknown baseline type: {baseline_type} for task: {task}")


__all__ = [
    "ZeroShotMathSolver",
    "ZeroShotQA",
    "ManualFewShotMathSolver",
    "ManualFewShotQA",
    "run_baseline",
    "create_baseline",
]
