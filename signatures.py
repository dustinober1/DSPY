"""
DSPy Signatures for GSM8K and HotPotQA tasks.

Signatures are declarative specifications of input/output behavior.
DSPy uses these to automatically generate and optimize prompts.
"""
import dspy


# ==================== GSM8K Signatures ====================

class SolveMath(dspy.Signature):
    """Solve a grade school math word problem step by step, showing your reasoning."""
    
    question = dspy.InputField(desc="a math word problem")
    reasoning = dspy.OutputField(desc="step-by-step solution showing all work")
    answer = dspy.OutputField(desc="the final numerical answer")


class GenerateMathReasoning(dspy.Signature):
    """Generate step-by-step reasoning for solving a math problem."""
    
    question = dspy.InputField(desc="a math word problem")
    reasoning = dspy.OutputField(desc="detailed step-by-step reasoning")


class ExtractAnswer(dspy.Signature):
    """Extract the final numerical answer from a reasoning trace."""
    
    question = dspy.InputField(desc="the original question")
    reasoning = dspy.InputField(desc="step-by-step reasoning")
    answer = dspy.OutputField(desc="final numerical answer only")


# ==================== HotPotQA Signatures ====================

class AnswerQuestion(dspy.Signature):
    """Answer a question based on the provided context."""
    
    context = dspy.InputField(desc="relevant background information")
    question = dspy.InputField(desc="a question to answer")
    reasoning = dspy.OutputField(desc="step-by-step reasoning")
    answer = dspy.OutputField(desc="short factual answer")


class GenerateSearchQuery(dspy.Signature):
    """Generate a search query to find information needed to answer a question."""
    
    question = dspy.InputField(desc="a question that needs to be answered")
    query = dspy.OutputField(desc="search query to find relevant information")


class AnswerFromContext(dspy.Signature):
    """Answer a question using only the provided context, with reasoning."""
    
    context = dspy.InputField(desc="passages of text that may contain the answer")
    question = dspy.InputField(desc="a question to answer")
    answer = dspy.OutputField(desc="answer derived from the context")


class MultiHopReasoning(dspy.Signature):
    """Perform multi-hop reasoning to answer a complex question."""
    
    context = dspy.InputField(desc="background information from multiple sources")
    question = dspy.InputField(desc="a question requiring multi-hop reasoning")
    hop1 = dspy.OutputField(desc="first reasoning step / fact")
    hop2 = dspy.OutputField(desc="second reasoning step / fact")
    answer = dspy.OutputField(desc="final answer synthesizing both hops")


# ==================== General Reasoning ====================

class ChainOfThought(dspy.Signature):
    """Think step-by-step to solve a problem."""
    
    question = dspy.InputField()
    reasoning = dspy.OutputField(desc="step-by-step thought process")
    answer = dspy.OutputField()


class Reflect(dspy.Signature):
    """Reflect on whether an answer is correct and why."""
    
    question = dspy.InputField()
    answer = dspy.InputField()
    reflection = dspy.OutputField(desc="critical analysis of the answer")
    is_correct = dspy.OutputField(desc="yes or no")
