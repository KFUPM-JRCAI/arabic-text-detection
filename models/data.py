import os
import sys
import jsonlines

# Set seed for Python's random module
import random

# Set seed for NumPy
import numpy as np

# Set seed for PyTorch
import torch

sys.path.append(os.getcwd())  # add the root folder to path
import datasets
from transformers import BertTokenizer, AutoTokenizer
import requests
import json
import zipfile
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule
from tqdm.auto import tqdm

from pyarabic import araby


# Set a global seed
GLOBAL_SEED = 42

random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# Make CuDNN deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def collate_fn(self, batch):
        texts, labels = zip(*batch)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_len,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, attention_mask, labels


class ArabicAIDetectorDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        max_len=512,
        data_dir="downloaded_data_files/arabic_ai_detector",
        # model_name="aubmindlab/bert-base-arabertv02",
        model_name="xlm-roberta-base",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.urls = {
            "train": [
                "https://github.com/Hamed1Hamed/Arabic_AI_Detector/raw/main/Dataset/LargeTraining.csv",
                "https://github.com/Hamed1Hamed/Arabic_AI_Detector/raw/main/Dataset/customTraining.csv",
            ],
            "val": [
                "https://github.com/Hamed1Hamed/Arabic_AI_Detector/raw/main/Dataset/LargeValidation.csv",
                "https://raw.githubusercontent.com/Hamed1Hamed/Arabic_AI_Detector/main/Dataset/customValidation.csv",
            ],
            "test": [
                "https://raw.githubusercontent.com/Hamed1Hamed/Arabic_AI_Detector/main/Dataset/LargeTesting.csv",
                "https://raw.githubusercontent.com/Hamed1Hamed/Arabic_AI_Detector/main/Dataset/customTesting.csv",
                # "https://raw.githubusercontent.com/Hamed1Hamed/Arabic_AI_Detector/main/Dataset/AIRABICTesting.csv",
            ],
        }

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for split, urls in self.urls.items():
            for url in urls:
                filename = os.path.join(
                    self.data_dir, f"{split}_{os.path.basename(url)}"
                )
                if not os.path.exists(filename):
                    print(f"Downloading: {url}")
                    response = requests.get(url)
                    with open(filename, "wb") as f:
                        f.write(response.content)
                else:
                    print(f"File already exists: {filename}")

    def setup(self, stage=None):
        self.prepare_data()
        train_data, val_data, test_data = [], [], []
        for split, urls in self.urls.items():
            for url in urls:
                filename = os.path.join(
                    self.data_dir,
                    f"{split}_{os.path.basename(url)}",
                )
                df = pd.read_csv(filename)
                data = list(zip(df["text"], df["label"]))
                if split == "train":
                    train_data.extend(data)
                elif split == "val":
                    val_data.extend(data)
                else:  # test
                    test_data.extend(data)

        # Invert the labels (0 -> 1, 1 -> 0)
        train_texts, train_labels = zip(
            *[(araby.strip_diacritics(text), 1 - label) for text, label in train_data]
        )
        val_texts, val_labels = zip(
            *[(araby.strip_diacritics(text), 1 - label) for text, label in val_data]
        )
        test_texts, test_labels = zip(
            *[(araby.strip_diacritics(text), 1 - label) for text, label in test_data]
        )

        print(len(train_texts), len(val_texts), len(test_texts))

        self.train_dataset = TextDataset(
            train_texts,
            train_labels,
            self.tokenizer,
            self.max_len,
        )
        self.val_dataset = TextDataset(
            val_texts,
            val_labels,
            self.tokenizer,
            self.max_len,
        )
        self.test_dataset = TextDataset(
            test_texts,
            test_labels,
            self.tokenizer,
            self.max_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=self.test_dataset.collate_fn,
        )


class AraSumDataModule(LightningDataModule):
    def __init__(
        self,
        max_len=512,
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.15,
        generated_by=["allam"],  # Now accepts a list of sources
    ):
        super().__init__()
        self.file_paths = [
            f"generated_arabic_datasets/{source}/arasum/generated_articles_from_polishing.jsonl"
            for source in generated_by
        ]
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def prepare_data(self):
        self.data = []
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")

            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    item = json.loads(line)
                    item["source"] = file_path  # Store the source for later reference
                    self.data.append(item)

    def setup(self, stage=None):
        self.prepare_data()
        texts, labels = [], []

        for item in self.data:
            original_article = item["original_article"]

            # Check if this is a dict type? this is expected in some cases from Allam's generated datasets
            # if "allam" in item["source"]:
            if isinstance(item["generated_article"], dict):
                generated_article = item["generated_article"]["choices"][0]["message"][
                    "content"
                ]
            else:
                generated_article = item["generated_article"]

            if original_article.strip() not in texts:
                texts.append(original_article.strip())
                labels.append(0)  # 0 for human-written

            texts.append(generated_article)
            labels.append(1)  # 1 for AI-generated

        # Shuffle and split the data
        combined = list(zip(texts, labels))
        random.seed(GLOBAL_SEED)
        random.shuffle(combined)
        texts, labels = zip(*combined)

        train_end = int(len(texts) * self.train_ratio)
        val_end = int(len(texts) * (self.train_ratio + self.val_ratio))

        self.train_dataset = TextDataset(
            texts[:train_end], labels[:train_end], self.tokenizer, self.max_len
        )
        self.val_dataset = TextDataset(
            texts[train_end:val_end],
            labels[train_end:val_end],
            self.tokenizer,
            self.max_len,
        )
        self.test_dataset = TextDataset(
            texts[val_end:], labels[val_end:], self.tokenizer, self.max_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.test_dataset.collate_fn,
        )


class ArabicAbstractsDataModule(LightningDataModule):
    AVAILABLE_MODELS = ["allam", "jais-batched", "llama-batched", "openai"]
    GENERATION_TYPES = [
        "by_polishing_abstracts_abstracts_generation_filtered",
        "from_title_abstracts_generation_filtered",
        "from_title_and_content_abstracts_generation_filtered",
    ]

    # Class mapping for multi-label classification
    MODEL_TO_LABEL = {
        "human": 0,
        "allam": 1,
        "jais-batched": 2,
        "llama-batched": 3,
        "openai": 4,
    }

    def __init__(
        self,
        max_len: int = 512,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        models=None,  # list
        generation_types=None,  # list
        multi_label: bool = False,
        balance_ai_with_human=False,
    ):
        super().__init__()
        self.generated_base_path = "generated_arabic_datasets"
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.multi_label = multi_label
        self.balance_ai_with_human = balance_ai_with_human

        # Set default values if None
        self.models = models if models is not None else self.AVAILABLE_MODELS
        self.generation_types = (
            generation_types if generation_types is not None else self.GENERATION_TYPES
        )

        # Validate inputs
        self._validate_inputs()

        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def _validate_inputs(self):
        """Validate the model and generation type inputs."""
        invalid_models = [m for m in self.models if m not in self.AVAILABLE_MODELS]
        if invalid_models:
            raise ValueError(
                f"Invalid models: {invalid_models}. Available models: {self.AVAILABLE_MODELS}"
            )

        invalid_types = [
            t for t in self.generation_types if t not in self.GENERATION_TYPES
        ]
        if invalid_types:
            raise ValueError(
                f"Invalid generation types: {invalid_types}. Available types: {self.GENERATION_TYPES}"
            )

    def _get_label(self, model_name: str) -> int:
        """Get the appropriate label based on the model name and classification mode."""
        if self.multi_label:
            return self.MODEL_TO_LABEL[model_name]
        else:
            return 0 if model_name == "human" else 1

    def _load_data(self):
        """Load original and generated abstracts from all specified models and generation types."""
        texts = []
        labels = []
        processed_originals = set()  # To track processed original abstracts

        for model in self.models:
            model_path = os.path.join(
                self.generated_base_path, model, "arabic_abstracts_dataset"
            )

            for gen_type in self.generation_types:
                file_path = os.path.join(model_path, f"{gen_type}.jsonl")

                if not os.path.exists(file_path):
                    print(f"Warning: File not found: {file_path}")
                    continue

                try:
                    with jsonlines.open(file_path) as reader:
                        for item in reader:
                            # Add original abstract if not processed before
                            if item["original_abstract"] not in processed_originals:
                                # if item["original_abstract"] == "":
                                #     continue
                                texts.append(item["original_abstract"])
                                labels.append(self._get_label("human"))
                                processed_originals.add(item["original_abstract"])

                            # Add generated abstract
                            # if item["generated_abstract"] == "":
                            #     continue
                            if self.balance_ai_with_human:
                                if (
                                    len([_ for _ in labels if _ == 1]) >= 3000
                                ):  # we have around 3k human abstracts
                                    continue
                            texts.append(item["generated_abstract"])
                            labels.append(self._get_label(model))
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")

        return texts, labels

    def prepare_data(self):
        """Load all data pairs."""
        self.texts, self.labels = self._load_data()
        # perform shuffling
        # Convert to list of pairs, shuffle, and unzip
        pairs = list(zip(self.texts, self.labels))
        random.seed(GLOBAL_SEED)
        random.shuffle(pairs)
        self.texts, self.labels = zip(*pairs)

        # Convert back to lists if needed (zip creates tuples)
        self.texts = list(self.texts)
        self.labels = list(self.labels)

    def setup(self, stage=None):
        self.prepare_data()

        assert all([i for i in self.texts]), "some texts are empty!"

        # Split the data
        train_end = int(len(self.texts) * self.train_ratio)
        val_end = int(len(self.texts) * (self.train_ratio + self.val_ratio))

        self.train_dataset = TextDataset(
            self.texts[:train_end],
            self.labels[:train_end],
            self.tokenizer,
            self.max_len,
        )
        self.val_dataset = TextDataset(
            self.texts[train_end:val_end],
            self.labels[train_end:val_end],
            self.tokenizer,
            self.max_len,
        )
        self.test_dataset = TextDataset(
            self.texts[val_end:], self.labels[val_end:], self.tokenizer, self.max_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            # shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.test_dataset.collate_fn,
        )


class ArabicSocialMediaDataModule(LightningDataModule):
    AVAILABLE_MODELS = ["allam", "jais-batched", "llama-batched", "openai"]
    GENERATION_TYPES = ["by_polishing_posts_generation_filtered"]

    # Class mapping for multi-label classification
    MODEL_TO_LABEL = {
        "human": 0,
        "allam": 1,
        "jais-batched": 2,
        "llama-batched": 3,
        "openai": 4,
    }

    def __init__(
        self,
        max_len: int = 512,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        models=None,  # list
        generation_types=None,  # list
        multi_label: bool = False,
    ):
        super().__init__()
        self.generated_base_path = "generated_arabic_datasets"
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.multi_label = multi_label

        # Set default values if None
        self.models = models if models is not None else self.AVAILABLE_MODELS
        self.generation_types = (
            generation_types if generation_types is not None else self.GENERATION_TYPES
        )

        # Validate inputs
        self._validate_inputs()

        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def _validate_inputs(self):
        """Validate the model and generation type inputs."""
        invalid_models = [m for m in self.models if m not in self.AVAILABLE_MODELS]
        if invalid_models:
            raise ValueError(
                f"Invalid models: {invalid_models}. Available models: {self.AVAILABLE_MODELS}"
            )

        invalid_types = [
            t for t in self.generation_types if t not in self.GENERATION_TYPES
        ]
        if invalid_types:
            raise ValueError(
                f"Invalid generation types: {invalid_types}. Available types: {self.GENERATION_TYPES}"
            )

    def _get_label(self, model_name: str) -> int:
        """Get the appropriate label based on the model name and classification mode."""
        if self.multi_label:
            return self.MODEL_TO_LABEL[model_name]
        else:
            return 0 if model_name == "human" else 1

    def _load_data(self):
        """Load original and generated abstracts from all specified models and generation types."""
        texts = []
        labels = []
        processed_originals = set()  # To track processed original abstracts

        for model in self.models:
            model_path = os.path.join(
                self.generated_base_path, model, "arabic_social_media_dataset"
            )

            for gen_type in self.generation_types:
                file_path = os.path.join(model_path, f"{gen_type}.jsonl")

                if not os.path.exists(file_path):
                    print(f"Warning: File not found: {file_path}")
                    continue

                try:
                    with jsonlines.open(file_path) as reader:
                        for item in reader:
                            # Add original post if not processed before
                            if item["original_post"] not in processed_originals:
                                # if item["original_post"] == "":
                                #     continue
                                texts.append(item["original_post"])
                                labels.append(self._get_label("human"))
                                processed_originals.add(item["original_post"])

                            # Add generated post
                            if item["generated_post"] == "":
                                continue
                            texts.append(item["generated_post"])
                            labels.append(self._get_label(model))
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")

        return texts, labels

    def prepare_data(self):
        """Load all data pairs."""
        self.texts, self.labels = self._load_data()
        # perform shuffling
        # Convert to list of pairs, shuffle, and unzip
        pairs = list(zip(self.texts, self.labels))
        random.seed(GLOBAL_SEED)
        random.shuffle(pairs)
        self.texts, self.labels = zip(*pairs)

        # Convert back to lists if needed (zip creates tuples)
        self.texts = list(self.texts)
        self.labels = list(self.labels)

    def setup(self, stage=None):
        self.prepare_data()

        assert all([i for i in self.texts]), "some texts are empty!"

        # Split the data
        train_end = int(len(self.texts) * self.train_ratio)
        val_end = int(len(self.texts) * (self.train_ratio + self.val_ratio))

        self.train_dataset = TextDataset(
            self.texts[:train_end],
            self.labels[:train_end],
            self.tokenizer,
            self.max_len,
        )
        self.val_dataset = TextDataset(
            self.texts[train_end:val_end],
            self.labels[train_end:val_end],
            self.tokenizer,
            self.max_len,
        )
        self.test_dataset = TextDataset(
            self.texts[val_end:], self.labels[val_end:], self.tokenizer, self.max_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            # shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.test_dataset.collate_fn,
        )


class DataModuleFromDataModules(LightningDataModule):
    def __init__(self, datamodules, batch_size=32):
        """
        Merge multiple text DataModules that use custom TextDataset and collate_fn.

        Args:
            datamodules: List of LightningDataModule instances to merge
            batch_size: Batch size for the merged dataset
            seed: Random seed for shuffling
        """
        super().__init__()
        if not datamodules:
            raise ValueError("At least one datamodule must be provided")

        self.datamodules = datamodules
        self.batch_size = batch_size

        # Use the tokenizer from the first datamodule
        self.tokenizer = datamodules[0].tokenizer
        self.max_len = datamodules[0].max_len

    def prepare_data(self):
        for dm in self.datamodules:
            dm.prepare_data()

    def setup(self, stage=None):
        # Call setup on all datamodules
        for dm in self.datamodules:
            dm.setup(stage)

        if stage == "fit" or stage is None:
            # Create custom TextDataset instances with combined data
            train_pairs, val_pairs = [], []

            for dm in self.datamodules:
                # Combine texts and labels into pairs
                train_pairs.extend(
                    list(zip(dm.train_dataset.texts, dm.train_dataset.labels))
                )
                val_pairs.extend(list(zip(dm.val_dataset.texts, dm.val_dataset.labels)))

            # Shuffle the pairs using GLOBAL_SEED
            random.seed(GLOBAL_SEED)
            random.shuffle(train_pairs)
            random.shuffle(val_pairs)

            # Unzip the pairs back into texts and labels
            train_texts, train_labels = zip(*train_pairs)
            val_texts, val_labels = zip(*val_pairs)

            # Create new TextDataset instances with shuffled data
            self.train_dataset = TextDataset(
                list(train_texts),
                list(train_labels),
                self.tokenizer,
                self.max_len,
            )

            self.val_dataset = TextDataset(
                list(val_texts),
                list(val_labels),
                self.tokenizer,
                self.max_len,
            )

        if stage == "test" or stage is None:
            test_pairs = []
            for dm in self.datamodules:
                test_pairs.extend(
                    list(zip(dm.test_dataset.texts, dm.test_dataset.labels))
                )

            # Shuffle test data
            random.seed(GLOBAL_SEED)
            random.shuffle(test_pairs)

            # Unzip the pairs
            test_texts, test_labels = zip(*test_pairs)

            self.test_dataset = TextDataset(
                list(test_texts),
                list(test_labels),
                self.tokenizer,
                self.max_len,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
        )
