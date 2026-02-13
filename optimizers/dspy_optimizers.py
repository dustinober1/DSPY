"""
DSPy Optimization Pipeline

Wrapper around DSPy optimizers (teleprompters) for easy experimentation.
Supports BootstrapFewShot, BootstrapFewShotWithRandomSearch, and MIPRO.
"""
import dspy
from pathlib import Path
from typing import List, Callable, Optional, Any, Dict
import json
import pickle
from config import OPTIMIZER_CONFIGS, CACHE_DIR


# ==================== Optimizer Wrappers ====================

class OptimizerWrapper:
    """Base class for optimizer wrappers"""
    
    def __init__(
        self,
        metric: Callable,
        teacher_lm: Optional[dspy.LM] = None,
        student_lm: Optional[dspy.LM] = None,
    ):
        self.metric = metric
        self.teacher_lm = teacher_lm
        self.student_lm = student_lm
        self.optimizer = None
        self.compiled_program = None
    
    def compile(
        self,
        module: dspy.Module,
        trainset: List[Any],
        valset: Optional[List[Any]] = None,
        **kwargs,
    ) -> dspy.Module:
        """Compile/optimize a module. Returns optimized module."""
        raise NotImplementedError
    
    def save(self, path: Path):
        """Save compiled program"""
        if self.compiled_program is None:
            raise ValueError("No compiled program to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save using DSPy's save mechanism
        self.compiled_program.save(str(path))
        print(f"Saved compiled program to {path}")
    
    def load(self, path: Path, module_class: type) -> dspy.Module:
        """Load a compiled program"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved program at {path}")
        
        # Load using DSPy's load mechanism
        module = module_class()
        module.load(str(path))
        self.compiled_program = module
        return module


class BootstrapFewShotOptimizer(OptimizerWrapper):
    """
    BootstrapFewShot optimizer.
    
    Generates demonstrations by running the teacher model on training examples
    and uses them to create few-shot prompts for the student model.
    """
    
    def __init__(
        self,
        metric: Callable,
        teacher_lm: Optional[dspy.LM] = None,
        max_bootstrapped_demos: int = 8,
        max_labeled_demos: int = 8,
    ):
        super().__init__(metric, teacher_lm)
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
    
    def compile(
        self,
        module: dspy.Module,
        trainset: List[Any],
        valset: Optional[List[Any]] = None,
        **kwargs,
    ) -> dspy.Module:
        """Compile using BootstrapFewShot"""
        print(f"\n{'='*60}")
        print("Running BootstrapFewShot optimization...")
        print(f"{'='*60}")
        print(f"Training examples: {len(trainset)}")
        print(f"Max bootstrapped demos: {self.max_bootstrapped_demos}")
        print(f"Max labeled demos: {self.max_labeled_demos}")
        
        # Set teacher LM context if provided
        if self.teacher_lm:
            dspy.settings.configure(lm=self.teacher_lm)
        
        # Create optimizer
        self.optimizer = dspy.BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
        )
        
        # Compile
        self.compiled_program = self.optimizer.compile(
            module,
            trainset=trainset,
            **kwargs,
        )
        
        print(f"✓ BootstrapFewShot optimization complete")
        return self.compiled_program


class BootstrapRandomSearchOptimizer(OptimizerWrapper):
    """
    BootstrapFewShotWithRandomSearch optimizer.
    
    Extends BootstrapFewShot by searching over different combinations
    of demonstrations to find the best set.
    """
    
    def __init__(
        self,
        metric: Callable,
        teacher_lm: Optional[dspy.LM] = None,
        max_bootstrapped_demos: int = 8,
        max_labeled_demos: int = 8,
        num_candidate_programs: int = 16,
        num_threads: int = 4,
    ):
        super().__init__(metric, teacher_lm)
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.num_candidate_programs = num_candidate_programs
        self.num_threads = num_threads
    
    def compile(
        self,
        module: dspy.Module,
        trainset: List[Any],
        valset: Optional[List[Any]] = None,
        **kwargs,
    ) -> dspy.Module:
        """Compile using BootstrapFewShotWithRandomSearch"""
        print(f"\n{'='*60}")
        print("Running BootstrapFewShotWithRandomSearch optimization...")
        print(f"{'='*60}")
        print(f"Training examples: {len(trainset)}")
        print(f"Validation examples: {len(valset) if valset else 'N/A'}")
        print(f"Candidate programs: {self.num_candidate_programs}")
        
        # Require validation set for random search
        if not valset:
            raise ValueError("BootstrapRandomSearch requires a validation set")
        
        # Set teacher LM context if provided
        if self.teacher_lm:
            dspy.settings.configure(lm=self.teacher_lm)
        
        # Create optimizer
        self.optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=self.metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
            num_candidate_programs=self.num_candidate_programs,
            num_threads=self.num_threads,
        )
        
        # Compile
        self.compiled_program = self.optimizer.compile(
            module,
            trainset=trainset,
            valset=valset,
            **kwargs,
        )
        
        print(f"✓ BootstrapRandomSearch optimization complete")
        return self.compiled_program


class MIPROOptimizer(OptimizerWrapper):
    """
    MIPRO (Multi-prompt Instruction Proposal Optimizer).
    
    State-of-the-art optimizer that jointly optimizes instructions
    and few-shot demonstrations.
    """
    
    def __init__(
        self,
        metric: Callable,
        teacher_lm: Optional[dspy.LM] = None,
        num_candidates: int = 10,
        init_temperature: float = 1.0,
    ):
        super().__init__(metric, teacher_lm)
        self.num_candidates = num_candidates
        self.init_temperature = init_temperature
    
    def compile(
        self,
        module: dspy.Module,
        trainset: List[Any],
        valset: Optional[List[Any]] = None,
        **kwargs,
    ) -> dspy.Module:
        """Compile using MIPRO"""
        print(f"\n{'='*60}")
        print("Running MIPRO optimization...")
        print(f"{'='*60}")
        print(f"Training examples: {len(trainset)}")
        print(f"Validation examples: {len(valset) if valset else 'N/A'}")
        print(f"Num candidates: {self.num_candidates}")
        print("⚠ Note: MIPRO is computationally expensive and may take a while")
        
        # Require validation set
        if not valset:
            raise ValueError("MIPRO requires a validation set")
        
        # Set teacher LM context if provided
        if self.teacher_lm:
            dspy.settings.configure(lm=self.teacher_lm)
        
        # Create optimizer
        try:
            self.optimizer = dspy.MIPRO(
                metric=self.metric,
                num_candidates=self.num_candidates,
                init_temperature=self.init_temperature,
            )
        except AttributeError:
            print("⚠ MIPRO not available in this DSPy version, falling back to BootstrapRandomSearch")
            # Fallback to BootstrapRandomSearch
            fallback = BootstrapRandomSearchOptimizer(
                metric=self.metric,
                teacher_lm=self.teacher_lm,
            )
            return fallback.compile(module, trainset, valset, **kwargs)
        
        # Compile
        self.compiled_program = self.optimizer.compile(
            module,
            trainset=trainset,
            valset=valset,
            **kwargs,
        )
        
        print(f"✓ MIPRO optimization complete")
        return self.compiled_program


# ==================== Optimizer Factory ====================

def create_optimizer(
    optimizer_type: str,
    metric: Callable,
    teacher_lm: Optional[dspy.LM] = None,
    **kwargs,
) -> OptimizerWrapper:
    """
    Factory function to create optimizers.
    
    Args:
        optimizer_type: "bootstrap", "random_search", or "mipro"
        metric: Metric function for optimization
        teacher_lm: Teacher language model
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        OptimizerWrapper instance
    """
    # Get default configs
    if optimizer_type == "bootstrap":
        config = OPTIMIZER_CONFIGS.get("bootstrap_fewshot", {})
        config.update(kwargs)
        return BootstrapFewShotOptimizer(metric, teacher_lm, **config)
    
    elif optimizer_type == "random_search":
        config = OPTIMIZER_CONFIGS.get("bootstrap_random_search", {})
        config.update(kwargs)
        return BootstrapRandomSearchOptimizer(metric, teacher_lm, **config)
    
    elif optimizer_type == "mipro":
        config = OPTIMIZER_CONFIGS.get("mipro", {})
        config.update(kwargs)
        return MIPROOptimizer(metric, teacher_lm, **config)
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


# ==================== Utilities ====================

def inspect_optimized_program(program: dspy.Module) -> Dict[str, Any]:
    """
    Inspect an optimized DSPy program to see what was learned.
    
    Returns:
        Dictionary with learned prompts, demonstrations, etc.
    """
    inspection = {
        "demonstrations": [],
        "instructions": [],
        "predictors": [],
    }
    
    # Iterate through predictors in the module
    for name, predictor in program.named_predictors():
        predictor_info = {
            "name": name,
            "signature": str(predictor.signature),
        }
        
        # Extract demonstrations if available
        if hasattr(predictor, 'demos') and predictor.demos:
            predictor_info["num_demos"] = len(predictor.demos)
            predictor_info["demos"] = [
                {k: str(v) for k, v in demo.items()}
                for demo in predictor.demos[:3]  # Show first 3
            ]
        
        inspection["predictors"].append(predictor_info)
    
    return inspection


def print_inspection(inspection: Dict[str, Any]):
    """Pretty print program inspection"""
    print("\n" + "="*80)
    print("OPTIMIZED PROGRAM INSPECTION")
    print("="*80)
    
    for i, pred_info in enumerate(inspection["predictors"], 1):
        print(f"\nPredictor {i}: {pred_info['name']}")
        print(f"Signature: {pred_info['signature']}")
        
        if 'num_demos' in pred_info:
            print(f"Number of demonstrations: {pred_info['num_demos']}")
            print("\nSample demonstrations:")
            for j, demo in enumerate(pred_info.get('demos', []), 1):
                print(f"\n  Demo {j}:")
                for k, v in demo.items():
                    # Truncate long values
                    v_str = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                    print(f"    {k}: {v_str}")
    
    print("\n" + "="*80)


__all__ = [
    "OptimizerWrapper",
    "BootstrapFewShotOptimizer",
    "BootstrapRandomSearchOptimizer",
    "MIPROOptimizer",
    "create_optimizer",
    "inspect_optimized_program",
    "print_inspection",
]
