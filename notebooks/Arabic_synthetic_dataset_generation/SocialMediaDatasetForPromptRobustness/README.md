# Social Media Dataset - Prompt Robustness Analysis

This directory contains notebooks and utilities for conducting prompt robustness analysis on the Arabic social media dataset.

## Overview

For the journal submission, we're evaluating how different polishing prompts affect the detection of AI-generated text. We:
- Selected 100 random posts (with seed=42 for reproducibility)
- Created 5 diverse polishing prompts using different prompt engineering techniques
- Generate polished versions with both ALLaM and Llama 3.1 models

## Files

### Python Modules

1. **`prompts.py`** - Contains 5 polishing prompts with different approaches:
   - `prompt_1_direct`: Direct instruction (Arabic, baseline similar to original)
   - `prompt_2_cot`: Chain-of-Thought prompting (Arabic)
   - `prompt_3_role_based`: Role-based prompting with detailed persona (Arabic)
   - `prompt_4_constrained`: Explicit constraints with examples style (Arabic)
   - `prompt_5_english`: English instruction for cross-lingual robustness testing

2. **`sample_posts.py`** - Handles sampling of 100 posts with fixed seed:
   - Fixed random seed: 42
   - Samples 100 posts from the 3,500 available
   - Saves reference file with sampled posts and their indices

### Notebooks

1. **`llama.ipynb`** - Llama 3.1 70B generation notebook
   - Uses HuggingFace model loader from jrcai_corekit
   - Batch processing (batch_size=2)
   - Generates posts with all 5 prompts
   - Output: `generated_arabic_datasets/llama/arabic_social_media_dataset_prompt_robustness/`

2. **`allam.ipynb`** - ALLaM generation notebook
   - Uses HuggingFace model loader (similar to Llama)
   - Same batch processing approach
   - Uses same 100 sampled posts for consistency
   - Output: `generated_arabic_datasets/allam/arabic_social_media_dataset_prompt_robustness/`

## Output Structure

For each model, the following files will be generated:

```
generated_arabic_datasets/{model}/arabic_social_media_dataset_prompt_robustness/
├── sampled_posts_reference.json          # Reference file with sampled posts
├── prompt_1_direct_posts_generation.jsonl
├── prompt_2_cot_posts_generation.jsonl
├── prompt_3_role_based_posts_generation.jsonl
├── prompt_4_constrained_posts_generation.jsonl
└── prompt_5_english_posts_generation.jsonl
```

Each JSONL file contains entries with:
- `original_post`: The original human-written post
- `prompt_name`: Which prompt was used
- `generated_post`: The AI-generated polished version
- `original_index`: Index in the original dataset

## Key Changes from Original Notebooks

### Structure Improvements
1. ✅ Proper output directory structure with clear naming
2. ✅ Modular design with `prompts.py` and `sample_posts.py`
3. ✅ Better documentation and code organization
4. ✅ Resume capability - can restart from where it stopped
5. ✅ Reproducible sampling with fixed seed

### ALLaM Model Loading
- Changed from API-based to local HuggingFace loading (similar to Llama)
- Uses `LLMLoader` with appropriate initializer
- Path to model: `/hdd/shared_models/allam/` (update as needed)
- Note: You need to download ALLaM model locally first

### Prompt Engineering
Created 5 diverse prompts using different techniques:
1. **Direct**: Straightforward instructions
2. **CoT**: Step-by-step reasoning approach
3. **Role-based**: Persona with detailed guidelines
4. **Constrained**: Explicit rules with ✓/✗ format
5. **English**: Cross-lingual testing

## Usage

### Step 1: Download ALLaM Model
Before running the ALLaM notebook, download the ALLaM model from HuggingFace to your local directory.

### Step 2: Run Llama Notebook
```bash
jupyter notebook llama.ipynb
```
Or submit to SLURM:
```bash
sbatch --wrap="uv run papermill llama.ipynb llama.ipynb --log-output" \
       --mem=32G \
       --cpus-per-task=32 \
       --gres=gpu:1 \
       --partition=RTX3090 \
       --output=llama_prompt_robustness_%j.out \
       --error=llama_prompt_robustness_%j.err
```

### Step 3: Run ALLaM Notebook
```bash
jupyter notebook allam.ipynb
```
Or submit to SLURM similarly.

## Next Steps

After generation, you can:
1. Train detectors on original polishing method
2. Test on posts generated with different prompts
3. Analyze cross-prompt generalization performance
4. Compare results across different prompt engineering techniques

## Notes

- Both notebooks use the **same 100 sampled posts** (seed=42) for fair comparison
- The sampling is done in `sample_posts.py` which is imported by both notebooks
- Output format is JSONL with all metadata for analysis
- Each prompt generates exactly 100 posts (one per sampled input)
