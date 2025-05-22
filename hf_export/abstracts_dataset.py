import os
import datasets
import jsonlines
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()


models = ["allam", "jais-batched", "llama-batched", "openai"]
base_path = "generated_arabic_datasets/{model}/arabic_abstracts_dataset/"

dataset = {}

for model in models:
    for file in os.listdir(base_path.format(model=model)):
        if file.endswith("_filtered.jsonl"):
            generation_method = file.removesuffix(
                "_abstracts_generation_filtered.jsonl"
            )
            generation_method = generation_method.removesuffix(
                "_abstracts"
            )  # in the case of by_polishing_abstracts
            if generation_method not in dataset:
                dataset[generation_method] = []
            with jsonlines.open(base_path.format(model=model) + file, "r") as reader:
                for obj in tqdm(reader):
                    if obj["original_abstract"] not in [
                        d["original_abstract"] for d in dataset[generation_method]
                    ]:
                        dataset[generation_method].append(
                            {
                                "original_abstract": obj["original_abstract"],
                                f"{model.removesuffix('-batched')}_generated_abstract": obj[
                                    "generated_abstract"
                                ],
                            }
                        )
                    else:
                        for d in dataset[generation_method]:
                            if d["original_abstract"] == obj["original_abstract"]:
                                d.update(
                                    {
                                        f"{model.removesuffix('-batched')}_generated_abstract": obj[
                                            "generated_abstract"
                                        ],
                                    }
                                )
                                break


# assertions
for method in ["by_polishing_abstracts", "from_title", "from_title_and_content"]:
    assert len(dataset[method.removesuffix("_abstracts")]) == len(
        open(
            base_path.format(
                model=models[0]
            )  # all models have the same number of samples
            + f"{method}_abstracts_generation_filtered.jsonl"
        ).readlines()
    ), (
        f"{method}: {len(dataset[method.removesuffix('_abstracts')])} != {len(open(base_path.format(model=models[0]) + f'{method}_abstracts_generation_filtered.jsonl').readlines())}"
    )


# Convert to Hugging Face Dataset format
dataset_dict = {}
for method_name, data_list in dataset.items():
    # Convert list of dicts to Dataset
    hf_dataset = datasets.Dataset.from_list(data_list)
    dataset_dict[method_name] = hf_dataset

# Create a DatasetDict
final_dataset = datasets.DatasetDict(dataset_dict)

# Print dataset info
print("Dataset structure:")
for split_name, split_dataset in final_dataset.items():
    print(f"  {split_name}: {len(split_dataset)} samples")
    print(f"    Features: {list(split_dataset.features.keys())}")

# Push to Hugging Face Hub
repo_name = "MagedSaeed/arabic-generated-abstracts"

print(f"Pushing dataset to {repo_name}...")
final_dataset.push_to_hub(
    repo_name,
    private=False,  # Set to True if you want a private dataset
    token=os.environ["HF_TOKEN"],  # Will use the token from login() or environment
)
print("Dataset successfully pushed to Hugging Face Hub!")
print(f"You can view it at: https://huggingface.co/datasets/{repo_name}")
