from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import requests
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig

from ainfinity.app.schemas.base import DatalogSchema


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders"""

    def __init__(self, config: DictConfig):
        """
        Initialize dataset loader with configuration

        Args:
            config: Dataset configuration from Hydra
        """
        self.config = config

    @abstractmethod
    def load(self) -> DatasetDict:
        """
        Load dataset and return DatasetDict

        Returns:
            DatasetDict with train/validation splits
        """

    @abstractmethod
    def get_text_column(self) -> str:
        """
        Get the name of the text column for this dataset

        Returns:
            Column name containing text data
        """


class DatalogLoader(DatasetLoader):
    """Loader for datalog datasets from URLs"""

    def _parse_example(self, example: Dict) -> Optional[Dict]:
        """
        Parse and normalize datalog example based on type.

        Args:
            example: Raw example from JSONL file

        Returns:
            Normalized example with 'messages' field, or None if invalid
        """
        example_type = example.get("type")

        if example_type == DatalogSchema.CONVERSATION.value:
            # Conversation format: already has messages field
            if "messages" in example:
                return {"messages": example["messages"]}
            else:
                print("Warning: conversation type missing 'messages' field, skipping")
                return None

        elif example_type == DatalogSchema.TEXT_GENERATION.value:
            # Text generation: convert to conversation format
            # Input: text + context, Output: generated_text
            if "text" not in example or "generated_text" not in example:
                print(f"Warning: text_generation missing required fields, skipping: {example.keys()}")
                return None

            # Build conversation-style messages
            user_message = example["text"]
            if "context" in example:
                user_message = f"{example['context']}\n\n{user_message}"

            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": example["generated_text"]},
            ]
            return {"messages": messages}

        else:
            print(f"Warning: unknown type '{example_type}', skipping")
            return None

    def load(self) -> DatasetDict:
        """
        Load datalog dataset from URLs according to config.
        Always returns a DatasetDict with 'train' and 'validation' splits.

        Supports:
        - schema_type: conversation, text_generation, etc.
        - urls: list of URLs to download JSONL files
        - shuffle: whether to shuffle the combined dataset
        - train_eval_split: ratio for train/validation split (default 0.8)
        - train_max_samples: max samples for training set
        - validation_max_samples: max samples for validation set
        """
        import json

        print(f"Loading datalog dataset from {len(self.config.urls)} URLs...")

        # Download and parse JSONL files from URLs
        all_examples: List[Dict] = []
        for url in self.config.urls:
            print(f"Downloading from: {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Parse JSONL
            for line in response.text.strip().split("\n"):
                if line.strip():
                    raw_example = json.loads(line)
                    # Parse and normalize example based on type
                    normalized = self._parse_example(raw_example)
                    if normalized is not None:
                        all_examples.append(normalized)

        print(f"Loaded {len(all_examples)} examples total")

        if len(all_examples) == 0:
            raise ValueError("No valid examples found in dataset")

        # Create dataset from examples
        dataset = Dataset.from_list(all_examples)

        # Shuffle if requested
        if self.config.get("shuffle", True):
            dataset = dataset.shuffle(seed=42)

        # Split into train/validation
        train_eval_split = self.config.get("train_eval_split", 0.8)
        split_dataset = dataset.train_test_split(train_size=train_eval_split, seed=42)

        train_dataset = split_dataset["train"]
        validation_dataset = split_dataset["test"]

        # Limit samples if requested
        train_max_samples = self.config.get("train_max_samples", None)
        if train_max_samples is not None and train_max_samples < len(train_dataset):
            train_dataset = train_dataset.select(range(train_max_samples))

        validation_max_samples = self.config.get("validation_max_samples", None)
        if validation_max_samples is not None and validation_max_samples < len(validation_dataset):
            validation_dataset = validation_dataset.select(range(validation_max_samples))

        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(validation_dataset)}")

        return DatasetDict({"train": train_dataset, "validation": validation_dataset})

    def get_text_column(self) -> str:
        """Datalog uses 'messages' field for normalized conversation format"""
        return "messages"


class HuggingFaceLoader(DatasetLoader):
    """Loader for HuggingFace datasets"""

    def load(self) -> DatasetDict:
        """
        Load HuggingFace dataset according to config.
        Always returns a DatasetDict.

        Supports:
        - empty split configs (validation:)
        - shuffle, max_samples
        - revision
        """
        dataset = load_dataset(
            self.config.name,
            split=None,
            revision=self.config.get("revision", None),
        )

        # Prepare final splits according to config
        dataset_splits = {}
        split_cfg = self.config.get("split", {})

        for split_key, s_cfg in split_cfg.items():
            if s_cfg is not None:
                print(f"split_key={split_key}, s_cfg={s_cfg}")
                split_name = s_cfg.get("name", split_key)
                subset = dataset[split_name]

                # Shuffle if requested
                if s_cfg.get("shuffle", False):
                    subset = subset.shuffle(seed=42)

                # Limit number of samples if requested
                max_samples = s_cfg.get("max_samples", None)
                if max_samples is not None:
                    subset = subset.select(range(max_samples))

                dataset_splits[split_key] = subset

        dataset_dict = DatasetDict(dataset_splits)

        return dataset_dict

    def get_text_column(self) -> str:
        """Get text column name from config"""
        return self.config.text_col


class DatasetLoaderFactory:
    """Factory for creating appropriate dataset loaders"""

    @staticmethod
    def create_loader(config: DictConfig) -> DatasetLoader:
        """
        Create appropriate dataset loader based on source type

        Args:
            config: Dataset configuration from Hydra

        Returns:
            DatasetLoader instance (DatalogLoader or HuggingFaceLoader)

        Raises:
            ValueError: If source type is not supported
        """
        source = config.get("source", "huggingface")

        if source == "datalog":
            print("Creating datalog dataset loader...")
            return DatalogLoader(config)
        elif source == "huggingface":
            print("Creating HuggingFace dataset loader...")
            return HuggingFaceLoader(config)
        else:
            raise ValueError(f"Unsupported dataset source: {source}")


def load_dataset_wrapper(dataset_args: DictConfig) -> Tuple[DatasetDict, str]:
    """
    Unified dataset loader that routes to appropriate loader based on source.
    Returns (dataset, text_column_name)

    This is the recommended way to load datasets - uses the factory pattern.
    """
    loader = DatasetLoaderFactory.create_loader(dataset_args)
    dataset = loader.load()
    text_col = loader.get_text_column()
    return dataset, text_col
