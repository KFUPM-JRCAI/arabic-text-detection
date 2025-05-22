import os
import datasets
import jsonlines
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()


models = ["allam", "jais-batched", "llama-batched", "openai"]
base_path = "generated_arabic_datasets/{model}/arabic_social_media_dataset/"

dataset = []

for model in models:
    for file in os.listdir(base_path.format(model=model)):
        if file.endswith("_filtered.jsonl"):
            with jsonlines.open(base_path.format(model=model) + file, "r") as reader:
                for obj in tqdm(reader):
                    if obj["original_post"] not in [
                        d["original_post"] for d in dataset
                    ]:
                        dataset.append(
                            {
                                "original_post": obj["original_post"],
                                f"{model.removesuffix('-batched')}_generated_post": obj[
                                    "generated_post"
                                ],
                            }
                        )
                    else:
                        for d in dataset:
                            if d["original_post"] == obj["original_post"]:
                                d.update(
                                    {
                                        f"{model.removesuffix('-batched')}_generated_post": obj[
                                            "generated_post"
                                        ],
                                    }
                                )
                                break


# assertions
assert len(dataset) == len(
    open(
        base_path.format(model=models[0])  # all models have the same number of samples
        + "by_polishing_posts_generation_filtered.jsonl"
    ).readlines()
), (
    f"by_polishing_posts: {len(dataset)} != {len(open(base_path.format(model=models[0]) + 'by_polishing_posts_generation_filtered.jsonl').readlines())}"
)


hf_dataset = datasets.Dataset.from_list(dataset)

# Print dataset info
print("Dataset structure:")
print(f" {len(hf_dataset)} samples")
print(f" Features: {list(hf_dataset.features.keys())}")

# Push to Hugging Face Hub

repo_name = "MagedSaeed/arabic-generated-social-media-posts"
print(f"Pushing dataset to {repo_name}...")
hf_dataset.push_to_hub(
    repo_name,
    private=False,  # Set to True if you want a private dataset
    token=os.environ["HF_TOKEN"],  # Will use the token from login() or environment
)
print("Dataset successfully pushed to Hugging Face Hub!")
print(f"You can view it at: https://huggingface.co/datasets/{repo_name}")
