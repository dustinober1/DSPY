# Getting Started Guide

Welcome to the DSPy Small-to-SOTA Demo! This guide will help you get up and running quickly.

## Prerequisites

- Python 3.8 or higher
- 16GB+ RAM (32GB+ recommended)
- ~50GB disk space for models (if using local models)
- GPU with 16GB+ VRAM recommended (can use CPU or API as fallback)

## Installation

### Option 1: Quick Setup (Recommended)

Run the automated setup script:

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Set up directory structure
- Optionally download datasets and models

### Option 2: Manual Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/gsm8k data/hotpotqa models results dspy_cache

# Copy environment template
cp .env.example .env
```

## Configuration

### 1. Set Up API Keys (if using API models)

Edit `.env` file:

```bash
# For gated models like Llama-2
HF_TOKEN=your_huggingface_token

# For API fallback (optional)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

Get HuggingFace token from: https://huggingface.co/settings/tokens

### 2. Choose Your Model Backend

You have three options:

#### Option A: Local Models (Best for cost, requires GPU)

```bash
# Download Mistral-7B (recommended)
python setup_models.py --models mistral-7b

# Or download all small models
python setup_models.py --models all-small
```

Then in notebooks, use:
```python
small_lm = dspy.HFModel(
    model=SMALL_MODELS['mistral-7b'].model_path,
    max_tokens=512
)
```

#### Option B: vLLM Server (Fastest local inference)

Start vLLM server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --dtype auto \
    --max-model-len 2048
```

In notebooks:
```python
small_lm = dspy.HFClientVLLM(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    port=8000
)
```

#### Option C: API Models (Easiest, costs money)

In notebooks:
```python
# Use GPT-3.5 as "small" model
small_lm = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=512)

# Use GPT-4 as "large" model for comparison
large_lm = dspy.OpenAI(model='gpt-4', max_tokens=512)
```

## Running Experiments

### Jupyter Notebooks (Recommended for Learning)

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/gsm8k_demo.ipynb
# 2. notebooks/hotpotqa_demo.ipynb  (if created)
# 3. notebooks/comparison.ipynb     (if created)
```

Each notebook includes:
- Explanations of concepts
- Interactive code cells
- Visualizations
- Error analysis

### Command Line (For Automation)

```bash
# Evaluate zero-shot on GSM8K
python evaluate.py --task gsm8k --approach zero-shot --subset dev

# Evaluate all approaches on both tasks
python evaluate.py --task both --approach all --subset dev

# Quick test with limited examples
python evaluate.py --task gsm8k --approach all --subset dev --num-examples 10

# Save results to file
python evaluate.py --task gsm8k --approach all --subset test --save-results
```

## Common Workflows

### Workflow 1: Quick Demo (30 minutes)

Perfect for understanding DSPy basics:

```bash
# 1. Use API models for speed
# Edit notebooks to use: dspy.OpenAI(model='gpt-3.5-turbo')

# 2. Run GSM8K notebook with small dataset
# In notebook, use: eval_subset = dev_examples[:20]

# 3. Get results and see improvement
```

### Workflow 2: Full Local Demo (2-3 hours)

Complete local demonstration:

```bash
# 1. Download models
python setup_models.py --models all-small

# 2. Download datasets
python data/gsm8k_loader.py
python data/hotpotqa_loader.py

# 3. Run full notebooks
jupyter notebook

# 4. Compare results across tasks
```

### Workflow 3: Production Evaluation (1 day)

Comprehensive benchmarking:

```bash
# 1. Set up vLLM for speed
# Start vLLM server (see Option B above)

# 2. Run full evaluations
python evaluate.py --task gsm8k --approach all --subset test --save-results
python evaluate.py --task hotpotqa --approach all --subset test --save-results

# 3. Analyze results in comparison notebook
```

## Understanding the Results

### What to Expect

**GSM8K (Math Problems)**
- Zero-shot: ~25-35% accuracy (baseline)
- Few-shot: ~40-50% accuracy (manual examples help)
- DSPy optimized: ~60-75% accuracy (automatic optimization wins!)
- Target (Llama-70B): ~80% accuracy

**Improvement**: Small model + DSPy reaches ~75-94% of large model performance

**HotPotQA (Multi-Hop QA)**
- Zero-shot: ~20-30% F1
- Few-shot: ~35-45% F1
- DSPy optimized: ~50-65% F1
- Target (Llama-70B): ~70% F1

**Improvement**: Small model + DSPy reaches ~71-93% of large model performance

### Key Metrics to Watch

1. **Accuracy/F1**: Primary performance metric
2. **Improvement over baseline**: Shows DSPy's impact
3. **Cost per example**: API costs or compute time
4. **Gap to large model**: How close we get to SOTA performance

## Troubleshooting

### Issue: Out of Memory

**Solutions**:
```python
# 1. Reduce batch size in evaluation
evaluator = Evaluator(..., batch_size=1)

# 2. Use smaller model
small_lm = dspy.HFModel(
    model="microsoft/phi-2",  # Smaller than Mistral-7B
    max_tokens=256
)

# 3. Enable CPU offloading
small_lm = dspy.HFModel(
    model=model_path,
    device_map="auto",
    load_in_8bit=True  # Quantization
)

# 4. Fall back to API
small_lm = dspy.OpenAI(model='gpt-3.5-turbo')
```

### Issue: Slow Inference

**Solutions**:
- Use vLLM server (10-20x faster)
- Reduce max_tokens configuration
- Use smaller eval_subset for testing
- Consider paid API for experiments

### Issue: Import Errors

```bash
# Make sure you're in project root
cd /path/to/DSPY

# Make sure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Issue: CUDA Not Available

```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Or use CPU (slower)
# Models will auto-fallback to CPU
```

### Issue: Dataset Download Fails

```bash
# Manual download
python -c "from datasets import load_dataset; load_dataset('gsm8k', 'main')"

# Or use cached offline mode
export HF_DATASETS_OFFLINE=1
```

## Next Steps

After getting basic results:

1. **Experiment with optimizers**: Try BootstrapRandomSearch and MIPRO
2. **Tune hyperparameters**: Adjust max_demos, temperature, etc.
3. **Try different models**: Compare Mistral, Llama, Phi models
4. **Test on more data**: Use full dev/test sets
5. **Error analysis**: Inspect where models fail
6. **Custom tasks**: Adapt to your own use cases

## Getting Help

- Check the main [README.md](README.md) for overview
- Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if it exists
- Inspect notebook markdown cells for explanations
- Read DSPy docs: https://dspy-docs.vercel.app/
- Review code comments in Python files

## Tips for Best Results

1. **Start small**: Test with 10-20 examples before full runs
2. **Use caching**: DSPy caches LM calls - clear between experiments
3. **Save optimized programs**: Avoid re-running expensive optimization
4. **Monitor costs**: Track API usage if using paid models
5. **Compare fairly**: Use same examples across all approaches
6. **Analyze errors**: Understanding failures helps improve prompts
7. **Be patient**: Optimization can take 5-15 minutes

## Resource Usage Guide

| Configuration | Time | Cost | Accuracy |
|--------------|------|------|----------|
| API + Small dataset | 30 min | $5-10 | Demo quality |
| Local + Medium dataset | 2-3 hours | Free | Good results |
| vLLM + Full dataset | 4-6 hours | Free | Best results |
| Production setup | 1 day | Free/Low | Publication quality |

Happy experimenting! 🚀
