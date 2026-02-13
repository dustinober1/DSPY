# DSPy Small-to-SOTA Model Demo

Demonstrate how DSPy optimization improves local-model reasoning performance on GSM8K and HotPotQA.

## 🎯 Project Goal

Show how **local Ollama models + DSPy optimization** can approach strong benchmark performance at lower infrastructure cost on:
- **GSM8K**: Grade school math word problems (multi-step reasoning)
- **HotPotQA**: Multi-hop question answering (information synthesis)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Optional: set Ollama/OpenAI overrides
# OLLAMA_MODEL=lfm2.5-thinking:latest
# OLLAMA_API_BASE=http://localhost:11434
# OPENAI_API_KEY=your_openai_key
```

### 3. Download Models (Optional)

For local inference, pull Ollama models:

```bash
# Pull the default Ollama model from config/.env
python setup_models.py

# Or pull a specific model
python setup_models.py --models lfm2.5-thinking:latest
```

**Alternative**: You can use API models (OpenAI/Anthropic) by updating notebook model configuration.

### 4. Prepare Data

```bash
# Download and prepare GSM8K dataset
python data/gsm8k_loader.py

# Download and prepare HotPotQA dataset
python data/hotpotqa_loader.py
```

### 5. Run Experiments

Open and run the Jupyter notebooks:

```bash
jupyter notebook
```

Then navigate to:
- `notebooks/gsm8k_demo.ipynb` - Math word problems experiment
- `notebooks/hotpotqa_demo.ipynb` - Multi-hop QA experiment  
- `notebooks/comparison.ipynb` - Comparative analysis across both tasks

## 📊 What's Included

### Core Components

```
DSPY/
├── config.py                 # Configuration and model settings
├── signatures.py             # DSPy task signatures
├── modules.py                # DSPy reasoning modules
│
├── data/                     # Dataset loaders
│   ├── gsm8k_loader.py      # GSM8K data pipeline
│   └── hotpotqa_loader.py   # HotPotQA data pipeline
│
├── baselines/                # Baseline implementations
│   └── baseline_models.py   # Zero-shot and manual few-shot
│
├── optimizers/               # DSPy optimization
│   └── dspy_optimizers.py   # BootstrapFewShot, MIPRO, etc.
│
├── utils/                    # Utilities
│   ├── evaluation.py        # Evaluation harness
│   └── visualization.py     # Plotting functions
│
└── notebooks/                # Interactive demos
    ├── gsm8k_demo.ipynb     # Math problems demo
    ├── hotpotqa_demo.ipynb  # Multi-hop QA demo
    └── comparison.ipynb     # Cross-task analysis
```

### Key Features

✅ **Multiple Baselines**: Zero-shot, manual few-shot, DSPy-optimized  
✅ **Multiple Optimizers**: BootstrapFewShot, BootstrapRandomSearch, MIPRO  
✅ **Two Challenging Tasks**: Math reasoning (GSM8K) + Multi-hop QA (HotPotQA)  
✅ **Local Model Support**: Run entirely on local hardware with Ollama  
✅ **Comprehensive Analysis**: Error analysis, cost metrics, visualization  
✅ **Reproducible**: Seed control, caching, save/load optimized programs  

## 🧪 Expected Results

Based on the optimization approach:

### GSM8K (Math Word Problems)

| Approach | Expected Accuracy | Notes |
|----------|------------------|-------|
| Zero-Shot (Local Ollama Model) | ~20-30% | Minimal prompting |
| Few-Shot Manual (Local Ollama Model) | ~35-45% | 3-5 hand-crafted examples |
| **DSPy Optimized** (Local Ollama Model) | **~55-70%** | Automatic optimization |
| Reference (Llama-70B) | ~80% | Published benchmark |

**Goal Achievement**: ~70-90% of large model performance

### HotPotQA (Multi-Hop QA)

| Approach | Expected F1 | Notes |
|----------|-------------|-------|
| Zero-Shot (Local Ollama Model) | ~15-25% | No context guidance |
| Few-Shot Manual (Local Ollama Model) | ~30-40% | Manual examples |
| **DSPy Optimized** (Local Ollama Model) | **~45-60%** | With retrieval + reasoning |
| Reference (Llama-70B) | ~70% | Published benchmark |

**Goal Achievement**: ~65-85% of large model performance

## 💡 Key Insights

### Why Does This Work?

1. **Prompting Matters**: Small models often fail due to poor prompts, not lack of capability
2. **Demonstrations Help**: Good few-shot examples dramatically improve reasoning
3. **Automatic > Manual**: DSPy finds better demonstrations than hand-crafting
4. **Reasoning Traces**: Chain-of-thought prompting helps smaller models show their work
5. **Task-Specific Optimization**: Each task benefits from different optimization strategies

### When to Use DSPy Optimization

✅ **Good Fit**:
- Reasoning tasks (math, logic, multi-step problems)
- Tasks with clear evaluation metrics
- When you have 50-200 training examples
- Need to reduce API costs
- Want to run models locally

❌ **Not Ideal**:
- Pure knowledge retrieval (no reasoning needed)
- Fuzzy/subjective evaluation
- No training data available
- Tasks requiring knowledge beyond training cutoff

## 🔧 Advanced Usage

### Using Ollama Models

The project defaults to local Ollama inference:

```bash
# Pull and start Ollama (in separate terminal if needed)
ollama pull lfm2.5-thinking:latest
# If Ollama app/service is not already running:
ollama serve

# Example DSPy LM config
small_lm = dspy.LM(
    model="ollama/lfm2.5-thinking:latest",
    api_base="http://localhost:11434",
    api_key=None,
    max_tokens=1024,
    temperature=0.0,
)
```

### Trying Different Optimizers

```python
# BootstrapFewShot (fast, good baseline)
optimizer = create_optimizer("bootstrap", metric=gsm8k_metric, teacher_lm=teacher)

# BootstrapRandomSearch (better, slower)
optimizer = create_optimizer("random_search", metric=gsm8k_metric, teacher_lm=teacher)

# MIPRO (state-of-the-art, slowest)
optimizer = create_optimizer("mipro", metric=gsm8k_metric, teacher_lm=teacher)
```

### Saving/Loading Optimized Programs

```python
# Save after optimization
optimizer.save(CACHE_DIR / "my_optimized_program.json")

# Load later (skip re-optimization)
optimized_program = optimizer.load(
    CACHE_DIR / "my_optimized_program.json",
    module_class=MathSolver
)
```

## 📝 Hardware Requirements

### Minimum (Testing Only)

- CPU: 4+ cores
- RAM: 16 GB
- Storage: 50 GB
- GPU: Optional (CPU fallback available)

**Note**: CPU-only inference is very slow. Use API models for quick testing.

### Recommended (Local Inference)

- CPU: 8+ cores
- RAM: 32 GB
- Storage: 100 GB (for models)
- GPU: 16+ GB VRAM (e.g., RTX 4080, A10, T4)

### Ideal (Full Experimentation)

- CPU: 16+ cores  
- RAM: 64 GB
- Storage: 200 GB SSD
- GPU: 24+ GB VRAM (e.g., RTX 4090, A100, L40)

## 🐛 Troubleshooting

### Out of Memory
- Reduce batch sizes in config.py
- Use a smaller local Ollama model
- Reduce optimizer candidate counts / trainset size during iteration

### Slow Inference
- Use a smaller Ollama model for iteration
- Reduce `max_tokens` in model config
- Reduce evaluation subset size for fast feedback

### Import Errors
- Ensure you're in the project root directory
- Activate virtual environment
- Reinstall: `pip install -r requirements.txt --upgrade`

### GPU Not Detected
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## 📚 Learn More

### DSPy Resources
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [DSPy Paper](https://arxiv.org/abs/2310.03714)

### Datasets
- [GSM8K Paper](https://arxiv.org/abs/2110.14168)
- [HotPotQA Paper](https://arxiv.org/abs/1809.09600)

### Related Work
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [ReAct: Reasoning + Acting](https://arxiv.org/abs/2210.03629)

## 🤝 Contributing

This is a demonstration project. To extend it:

1. Add new tasks (e.g., MMLU, HellaSwag)
2. Try different model architectures
3. Implement custom optimizers
4. Add retrieval methods for RAG
5. Create web UI with Gradio/Streamlit

## 📄 License

MIT License - Feel free to use for learning and experimentation.

## 🙏 Acknowledgments

- **DSPy Team** at Stanford for the incredible framework
- **Ollama** for local model serving
- **HuggingFace Datasets** for benchmark data access

---

**Questions?** Open an issue or check the notebooks for detailed walkthroughs.

**Happy optimizing!** 🚀
