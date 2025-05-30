# Detection of Machine-Generated Arabic Text in the Era of Large Language Models

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-red)]([link-to-be-added](https://arxiv.org/abs/2505.23276))
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Abstracts-blue)](https://huggingface.co/datasets/MagedSaeed/arabic-generated-abstracts)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Social%20Media-green)](https://huggingface.co/datasets/MagedSaeed/arabic-generated-social-media-posts)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

This repository contains the official implementation and datasets for the research paper **"Detection of Machine-Generated Arabic Text in the Era of Large Language Models"** by Maged S. Al-Shaibani and Moataz Ahmed.

## 📋 Overview

This work represents the **first comprehensive investigation and stylometric analysis** of Arabic machine-generated text detection across multiple llms and generation methods, addressing the critical challenge of distinguishing between human-written and AI-generated Arabic content across multiple domains and generation strategies.

### 🎯 Contributions

- **Multi-dimensional stylometric analysis** of human vs. machine-generated Arabic text
- **Multi-prompt generation framework** across 4 LLMs (ALLaM, Jais, Llama 3.1, GPT-4)
- **High-performance detection systems** achieving up to 99.9% F1-score
- **Cross-domain evaluation** (academic abstracts + social media)
- **Cross-model generalization** studies

## 🏗️ Repository Structure

```
arabic_datasets/
    ├── arabic_filtered_papers.json
    └── social_media_dataset.json
generated_arabic_datasets/
    ├── allam/
        ├── arabic_abstracts_dataset/
            ├── by_polishing_abstracts_abstracts_generation_filtered.jsonl
            ├── by_polishing_abstracts_abstracts_generation.jsonl
            ├── from_title_abstracts_generation_filtered.jsonl
            ├── from_title_abstracts_generation.jsonl
            ├── from_title_and_content_abstracts_generation_filtered.jsonl
            └── from_title_and_content_abstracts_generation.jsonl
        ├── arabic_social_media_dataset/
            ├── by_polishing_posts_generation_filtered.jsonl
            └── by_polishing_posts_generation.jsonl
        └── arasum/
            └── generated_articles_from_polishing.jsonl
    ├── claude/
        ├── arabic_abstracts_dataset/
            ├── by_polishing_abstracts_abstracts_generation.jsonl
            ├── from_title_abstracts_generation.jsonl
            └── from_title_and_content_abstracts_generation.jsonl
        └── arasum/
            └── generated_articles_from_polishing.jsonl
    ├── jais-batched/
        ├── arabic_abstracts_dataset/
            ├── by_polishing_abstracts_abstracts_generation_filtered.jsonl
            ├── by_polishing_abstracts_abstracts_generation.jsonl
            ├── from_title_abstracts_generation_filtered.jsonl
            ├── from_title_abstracts_generation.jsonl
            ├── from_title_and_content_abstracts_generation_filtered.jsonl
            └── from_title_and_content_abstracts_generation.jsonl
        ├── arabic_social_media_dataset/
            ├── by_polishing_posts_generation_filtered.jsonl
            └── by_polishing_posts_generation.jsonl
        └── arasum/
            └── generated_articles_from_polishing.jsonl
    ├── llama-batched/
        ├── arabic_abstracts_dataset/
            ├── by_polishing_abstracts_abstracts_generation_filtered.jsonl
            ├── by_polishing_abstracts_abstracts_generation.jsonl
            ├── from_title_abstracts_generation_filtered.jsonl
            ├── from_title_abstracts_generation.jsonl
            ├── from_title_and_content_abstracts_generation_filtered.jsonl
            └── from_title_and_content_abstracts_generation.jsonl
        ├── arabic_social_media_dataset/
            ├── by_polishing_posts_generation_filtered.jsonl
            └── by_polishing_posts_generation.jsonl
        └── arasum/
            └── generated_articles_from_polishing.jsonl
    └── openai/
        ├── arabic_abstracts_dataset/
            ├── by_polishing_abstracts_abstracts_generation_filtered.jsonl
            ├── by_polishing_abstracts_abstracts_generation.jsonl
            ├── from_title_abstracts_generation_filtered.jsonl
            ├── from_title_abstracts_generation.jsonl
            ├── from_title_and_content_abstracts_generation_filtered.jsonl
            └── from_title_and_content_abstracts_generation.jsonl
        ├── arabic_social_media_dataset/
            ├── by_polishing_posts_generation_filtered.jsonl
            └── by_polishing_posts_generation.jsonl
        └── arasum/
            └── generated_articles_from_polishing.jsonl
hf_export/
    ├── abstracts_dataset.py
    └── social_media_dataset.py
models/
    ├── __init__.py
    ├── data.py
    ├── models.py
    └── train.py
notebooks/
    ├── Arabic_experiments/
        ├── ArabicAbstractsDataset/
            ├── Arabic_abstracts_dataset_preparation.ipynb
            ├── continual_training_of_arasum_detector_on_arabic_abstracts.ipynb
            ├── llms_multi_class_arabic_detector.ipynb
            ├── train_on_one_model_test_on_others.ipynb
            ├── train_on_one_prompt_test_on_others.ipynb
            └── zero_shot_on_arabic_abstracts_dataset.ipynb
        ├── ArabicSocialMediaDataset/
            ├── llms_multi_class_arabic_detector.ipynb
            ├── prepare_arabic_social_media_dataset.ipynb
            └── train_on_one_model_test_on_others.ipynb
        └── AraSum/
            ├── AllamWithAraSumTestingOnly.ipynb
            ├── arabic_detector_trained_on_all_llms.ipynb
            ├── arabic_detector_trained_on_allam.ipynb
            └── arasum_abstracts_detector.ipynb
    ├── Arabic_synthetic_dataset_generation/
        ├── AbstractsDataset/
            ├── allam.ipynb
            ├── analysis_on_the_generated_abstracts.ipynb
            ├── claude.ipynb
            ├── jais.ipynb
            ├── llama.ipynb
            ├── openai.ipynb
            └── top_frequent_words_analysis.ipynb
        ├── AraSum/
            ├── allam.ipynb
            ├── claude.ipynb
            ├── jais.ipynb
            ├── llama.ipynb
            └── openai.ipynb
        └── SocialMediaDataset/
            ├── allam.ipynb
            ├── analysis_on_the_generated_posts.ipynb
            ├── jais.ipynb
            ├── llama.ipynb
            ├── openai.ipynb
            └── top_frequent_words_analysis.ipynb
    └── exploration/
        ├── explore_arabic_content_detection_dataset.ipynb
        └── explore_arbicQA_dataset.ipynb
.gitattributes
.gitignore
LICENSE
README.md
requirements.txt
```

## 🔬 Research Methodology

### Text Generation Strategies

**Academic Abstracts (3 methods):**
- **Title-only generation**: Free-form generation from paper titles
- **Title+Content generation**: Content-aware generation using paper content
- **Abstract polishing**: Refinement of existing human abstracts

**Social Media Posts (1 method):**
- **Post polishing**: Refinement preserving dialectal expressions

### Models Evaluated

| Model | Size | Focus | Source |
|-------|------|-------|--------|
| **ALLaM** | 7B | Arabic-focused | Open |
| **Jais** | 70B | Arabic-focused | Open |
| **Llama 3.1** | 70B | General | Open |
| **GPT-4** | - | General | Closed |

### Detection Approaches

- **Binary detection**: Human vs. Machine-generated
- **Multi-class detection**: Identify specific LLM
- **Cross-model generalization**: Train on one model, test on others

## 📊 Spotlight Findings

### Stylometric Insights
- **Reduced vocabulary diversity** in AI-generated text
- **Distinctive word frequency patterns** with steeper drop-offs
- **Model-specific linguistic signatures** enabling identification
- **Domain-specific characteristics** varying between contexts

### Detection Performance

**Academic Abstracts:**
- Binary detection: **99.5-99.9% F1-score**
- Cross-model: **86.4-99.9% F1-score range**
- Multi-class: **94.1-98.2% F1-score per model**

**Social Media:**
- More challenging due to informal nature
- **Cross-domain generalization** issues confirmed
- **Model-specific detectability** variations observed

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt
```

### Installation

Make sure to also take a look at the corekit repo (https://github.com/MagedSaeed/llms-corekit) as it is needed in some scripts and notebooks.

```bash
git clone https://github.com/MagedSaeed/arabic-text-detection.git
cd arabic-text-detection

# Install dependencies
pip install -r requirements.txt

# Download datasets (requires Git LFS)
git lfs pull
```

The code was written with practices that support self-explanatory purpuses. You can browse the scripts and notebooks and run them providing the necessary API keys if required.

## 📁 Datasets

### Academic Abstracts
- **Source**: [Algerian Scientific Journals Platform](https://asjp.cerist.dz/)
- **Size**: 8,388 samples across 3 generation methods
- **Period**: 2010-2022 (pre-AI era)
- **Available**: [🤗 HuggingFace Hub](https://huggingface.co/datasets/MagedSaeed/arabic-generated-abstracts)

### Social Media Posts
- **Source**: BRAD (Book Reviews) + HARD (Hotel Reviews)
- **Size**: 3,318 samples (polishing method)
- **Available**: [🤗 HuggingFace Hub](https://huggingface.co/datasets/MagedSaeed/arabic-generated-social-media-posts)


## 🔗 Related Work

- **Abstracts Data Collection code**: [arabic-dataset](https://github.com/MagedSaeed/arabs-dataset)
- **LLMs corekit**: [LLMs-corekit](https://github.com/MagedSaeed/llms-corekit)

## 📚 Citation

```bibtex
coming soon
```

## 🏢 Institutional Support

This research is supported by:
- **SDAIA-KFUPM Joint Research Center for Artificial Intelligence**

## 👥 Authors

- **Maged S. Al-Shaibani**
- **Moataz Ahmed** - Corresponding Author (moataz.ahmed@kfupm.edu.sa)

*SDAIA-KFUPM Joint Research Center for Artificial Intelligence*  
*King Fahd University of Petroleum and Minerals, Saudi Arabia*


## ⚖️ Ethical Considerations

This research is intended to:
- **Improve detection** of machine-generated content
- **Enhance academic integrity** in Arabic contexts
- **Advance Arabic NLP** research capabilities
- **Support information verification** systems

Please use this work responsibly and in accordance with ethical AI principles.

---
