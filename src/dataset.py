import datasets
import pandas as pd
import lightning as pl


from pathlib import Path
from datasets import ClassLabel, Dataset, Value, Features
from torch.utils.data import DataLoader

from src import utilities
from transformers import AutoTokenizer
from src.utilities import split_data, tokenize_function, cast_to_float


class TweetsDataModule(pl.LightningDataModule):

    def __init__(
        self,
        raw_data_path: str,
        processed_data_dir: str,
        batch_size: int,
        num_workers: int,
        cleaning_steps: list[str],
        max_length: int,
        model_checkpoint: str,
        columns_to_rename: dict,
        columns_to_drop: list,
        class_names: list,
    ) -> None:

        super().__init__()

        self.raw_data_path = Path(raw_data_path)
        self.processed_data_dir = Path(processed_data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)

        self.columns_to_rename = columns_to_rename
        self.columns_to_drop = columns_to_drop
        self.cleaning_steps = cleaning_steps
        self.class_names = class_names

    def prepare_data(self) -> None:

        ds = pd.read_excel(self.raw_data_path)
        ds = Dataset.from_pandas(ds)

        ds = ds.rename_columns(self.columns_to_rename)
        ds = ds.remove_columns(column_names=self.columns_to_drop)
        ds = ds.cast_column(
            column="labels", feature=ClassLabel(names=self.class_names)
        )

        # Apply cleaning steps to tweets.
        if self.cleaning_steps:
            for transform in self.cleaning_steps:
                transform = getattr(utilities, transform)
                # Have to keep in memory because using cache was leading to data loss.
                ds = ds.map(transform, batched=True, keep_in_memory=True)

        # Have to keep in memory because using cache was leading to data loss.
        ds = ds.map(
            tokenize_function,
            batched=True,
            fn_kwargs={"tokenizer": self.tokenizer, "max_length": self.max_length},
            keep_in_memory=True
        )
        ds = split_data(ds)
        ds.set_format("torch")

        ds.save_to_disk(self.processed_data_dir)

    def setup(self, stage: str) -> None:
        self.train_ds = datasets.load_from_disk(self.processed_data_dir / "train")
        self.val_ds = datasets.load_from_disk(self.processed_data_dir / "val")
        self.test_ds = datasets.load_from_disk(self.processed_data_dir / "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )
