papermill with sbatch:

```bash
sbatch --wrap="uv run papermill notebooks/Arabic_experiments/ArabicSocialMediaDataset/text_lengths_experiments/allam_cross_model_detection.ipynb notebooks/Arabic_experiments/ArabicSocialMediaDataset/text_lengths_experiments/allam_cross_model_detection.ipynb --log-output" \
        --mem=32G \
        --cpus-per-task=32 \
        --gres=gpu:1 \
        --partition=RTX3090 \
        --output=notebooks/Arabic_experiments/ArabicSocialMediaDataset/text_lengths_experiments/allam_cross_model_detection_papermill_job_%j.out \
        --error=notebooks/Arabic_experiments/ArabicSocialMediaDataset/text_lengths_experiments/allam_cross_model_detection_papermill_job_%j.err
```