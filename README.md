# DSPy Small-to-SOTA Model Demo

Demonstrate how DSPy optimization can make small local models (2.7B parameters) achieve performance comparable to large SOTA models (70B+) on reasoning tasks.

## 🎯 Project Goal

Show that **Phi-2 + DSPy optimization** can reach **≥90% of Llama-70B performance** at **<5% of the cost** on:
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

# Edit .env and add your tokens (required for gated models like Llama-2)
# HF_TOKEN=your_huggingface_token_here
```

### 3. Download Models (Optional)

For local inference, download models:

```bash
# Download small models (Phi-2 recommended)
python setup_models.py --models all-small

# Or download all models including large ones (requires significant disk space)
python setup_models.py --models all
```

**Alternative**: You can use API models (OpenAI/Anthropic) instead by updating the configuration in the notebooks.

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
✅ **Local Model Support**: Run entirely on local hardware with vLLM  
✅ **Comprehensive Analysis**: Error analysis, cost metrics, visualization  
✅ **Reproducible**: Seed control, caching, save/load optimized programs  

## 🧪 Expected Results

Based on the optimization approach:

### GSM8K (Math Word Problems)

| Approach | Expected Accuracy | Notes |
|----------|------------------|-------|
| Zero-Shot (Phi-2) | ~20-30% | Minimal prompting |
| Few-Shot Manual (Phi-2) | ~35-45% | 3-5 hand-crafted examples |
| **DSPy Optimized** (Phi-2) | **~55-70%** | Automatic optimization |
| Reference (Llama-70B) | ~80% | Published benchmark |

**Goal Achievement**: ~70-90% of large model performance

### HotPotQA (Multi-Hop QA)

| Approach | Expected F1 | Notes |
|----------|-------------|-------|
| Zero-Shot (Phi-2) | ~15-25% | No context guidance |
| Few-Shot Manual (Phi-2) | ~30-40% | Manual examples |
| **DSPy Optimized** (Phi-2) | **~45-60%** | With retrieval + reasoning |
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

### Using vLLM for Fast Inference

vLLM dramatically speeds up local model inference:

```bash
# Start vLLM server (in separate terminal)
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/phi-2 \
    --dtype auto \
    --max-model-len 2048

# Then update notebook to use vLLM client
small_lm = dspy.HFClientVLLM(
    model="microsoft/phi-2",
    port=8000,
    url="http://localhost"
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
- Use smaller models (Phi-2 instead of Llama-7B)
- Enable CPU offloading: `device_map="auto"`

### Slow Inference
- Use vLLM server instead of direct HuggingFace
- Reduce `max_tokens` in model config
- Use API models for initial testing

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
- **HuggingFace** for model hosting and datasets library
- **Mistral AI** and **Meta** for open-source models

---

**Questions?** Open an issue or check the notebooks for detailed walkthroughs.

**Happy optimizing!** 🚀
