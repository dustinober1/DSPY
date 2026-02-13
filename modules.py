"""
DSPy Modules for GSM8K and HotPotQA tasks.

Modules are composable building blocks that use signatures to implement
reasoning patterns. They can be optimized by DSPy teleprompters.
"""
import dspy
from typing import Optional, List
from signatures import (
    SolveMath,
    GenerateMathReasoning,
    ExtractAnswer,
    AnswerQuestion,
    GenerateSearchQuery,
    AnswerFromContext,
    MultiHopReasoning,
)


# ==================== GSM8K Modules ====================

class MathSolver(dspy.Module):
    """
    Solve math problems using chain-of-thought reasoning.
    
    This is the main module for GSM8K. It generates step-by-step
    reasoning and extracts the final numerical answer.
    """
    
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought(SolveMath)
    
    def forward(self, question: str):
        result = self.solve(question=question)
        return dspy.Prediction(
            reasoning=result.reasoning,
            answer=result.answer,
        )


class MathSolverWithReflection(dspy.Module):
    """
    Math solver that reflects on its answer before finalizing.
    
    Uses a two-step process:
    1. Generate initial solution
    2. Reflect and potentially revise
    """
    
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought(SolveMath)
        # Could add reflection step here
    
    def forward(self, question: str):
        # First attempt
        result = self.solve(question=question)
        
        # TODO: Add reflection/revision step
        # For now, just return first attempt
        return dspy.Prediction(
            reasoning=result.reasoning,
            answer=result.answer,
        )


class ProgramMathSolver(dspy.Module):
    """
    Decomposed math solver that separates reasoning from answer extraction.
    
    This can be useful for optimization as each step can be tuned separately.
    """
    
    def __init__(self):
        super().__init__()
        self.generate_reasoning = dspy.ChainOfThought(GenerateMathReasoning)
        self.extract_answer = dspy.Predict(ExtractAnswer)
    
    def forward(self, question: str):
        # Generate reasoning
        reasoning_result = self.generate_reasoning(question=question)
        
        # Extract answer
        answer_result = self.extract_answer(
            question=question,
            reasoning=reasoning_result.reasoning,
        )
        
        return dspy.Prediction(
            reasoning=reasoning_result.reasoning,
            answer=answer_result.answer,
        )


# ==================== HotPotQA Modules ====================

class SimpleQA(dspy.Module):
    """
    Simple QA module that answers questions given context.
    
    Assumes context is already provided (no retrieval).
    """
    
    def __init__(self):
        super().__init__()
        self.answer = dspy.ChainOfThought(AnswerQuestion)
    
    def forward(self, question: str, context: str):
        result = self.answer(question=question, context=context)
        return dspy.Prediction(
            reasoning=result.reasoning,
            answer=result.answer,
        )


class MultiHopQA(dspy.Module):
    """
    Multi-hop QA module that explicitly performs multi-hop reasoning.
    
    This is useful for HotPotQA which requires combining information
    from multiple sources.
    """
    
    def __init__(self):
        super().__init__()
        self.reason = dspy.ChainOfThought(MultiHopReasoning)
    
    def forward(self, question: str, context: str):
        result = self.reason(question=question, context=context)
        return dspy.Prediction(
            hop1=result.hop1,
            hop2=result.hop2,
            answer=result.answer,
        )


class RAGMultiHopQA(dspy.Module):
    """
    RAG (Retrieval-Augmented Generation) module for multi-hop QA.
    
    Combines retrieval with reasoning. In practice, you'd pass a
    retriever to this module.
    """
    
    def __init__(self, retriever=None, num_passages: int = 3):
        super().__init__()
        self.retriever = retriever
        self.num_passages = num_passages
        self.answer = dspy.ChainOfThought(AnswerQuestion)
    
    def forward(self, question: str):
        # Retrieve context (if retriever is provided)
        if self.retriever:
            context = self.retriever(question, k=self.num_passages)
        else:
            # Fallback: empty context
            context = ""
        
        # Answer with retrieved context
        result = self.answer(question=question, context=context)
        
        return dspy.Prediction(
            context=context,
            reasoning=result.reasoning,
            answer=result.answer,
        )


class ReActQA(dspy.Module):
    """
    ReAct-style QA module that interleaves reasoning and retrieval.
    
    This is a simplified version - a full ReAct implementation would
    actually execute retrieval actions iteratively.
    """
    
    def __init__(self, retriever=None):
        super().__init__()
        self.retriever = retriever
        self.generate_query = dspy.Predict(GenerateSearchQuery)
        self.answer = dspy.ChainOfThought(AnswerQuestion)
    
    def forward(self, question: str):
        # Generate search query
        query_result = self.generate_query(question=question)
        search_query = query_result.query
        
        # Retrieve (if retriever available)
        if self.retriever:
            context = self.retriever(search_query, k=3)
        else:
            context = ""
        
        # Answer with retrieved context
        answer_result = self.answer(question=question, context=context)
        
        return dspy.Prediction(
            search_query=search_query,
            context=context,
            reasoning=answer_result.reasoning,
            answer=answer_result.answer,
        )


# ==================== Module Factory ====================

def get_module(task: str, module_type: str = "default", **kwargs):
    """
    Factory function to get a module by task and type.
    
    Args:
        task: "gsm8k" or "hotpotqa"
        module_type: Type of module (e.g., "default", "reflection", "react")
        **kwargs: Additional arguments for module initialization
        
    Returns:
        Initialized DSPy module
    """
    if task == "gsm8k":
        if module_type == "default":
            return MathSolver()
        elif module_type == "reflection":
            return MathSolverWithReflection()
        elif module_type == "program":
            return ProgramMathSolver()
        else:
            raise ValueError(f"Unknown GSM8K module type: {module_type}")
    
    elif task == "hotpotqa":
        if module_type == "default" or module_type == "simple":
            return SimpleQA()
        elif module_type == "multihop":
            return MultiHopQA()
        elif module_type == "rag":
            return RAGMultiHopQA(**kwargs)
        elif module_type == "react":
            return ReActQA(**kwargs)
        else:
            raise ValueError(f"Unknown HotPotQA module type: {module_type}")
    
    else:
        raise ValueError(f"Unknown task: {task}")
