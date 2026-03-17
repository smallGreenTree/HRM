from __future__ import annotations

import json
import os

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel, model_validator

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/synthetic-arithmetic"

    train_size: int = 50000
    test_size: int = 5000

    num_terms: int = 10
    train_modulus: int = 5
    test_modulus: int | None = None
    noise_range: int = 100

    seed: int = 0

    @model_validator(mode="after")
    def validate_config(self) -> "DataProcessConfig":
        if self.num_terms < 2:
            raise ValueError("num_terms must be at least 2.")
        if self.train_modulus < 2:
            raise ValueError("train_modulus must be at least 2.")
        if self.test_modulus is not None and self.test_modulus < 2:
            raise ValueError("test_modulus must be at least 2.")
        if self.noise_range < 1:
            raise ValueError("noise_range must be at least 1.")
        if self.train_size < 1 or self.test_size < 1:
            raise ValueError("train_size and test_size must be positive.")
        return self


QUERY_TOKEN_ID = 1


def encode_signal(value: np.ndarray | int, max_modulus: int) -> np.ndarray | int:
    return np.asarray(value) + 2


def encode_noise(value: np.ndarray | int, max_modulus: int) -> np.ndarray | int:
    return np.asarray(value) + 2 + max_modulus


def build_split(num_examples: int, modulus: int, max_modulus: int, config: DataProcessConfig, rng: np.random.Generator):
    seq_len = 2 * (config.num_terms - 1) + 1
    inputs = np.zeros((num_examples, seq_len), dtype=np.int32)
    labels = np.zeros((num_examples, seq_len), dtype=np.int32)

    for i in range(num_examples):
        start = int(rng.integers(0, modulus))
        delta = int(rng.integers(1, modulus))
        signals = (start + np.arange(config.num_terms, dtype=np.int32) * delta) % modulus
        noise = rng.integers(0, config.noise_range, size=config.num_terms - 1, dtype=np.int32)

        pos = 0
        for t in range(config.num_terms - 1):
            inputs[i, pos] = int(encode_signal(signals[t], max_modulus))
            pos += 1
            inputs[i, pos] = int(encode_noise(noise[t], max_modulus))
            pos += 1

        inputs[i, pos] = QUERY_TOKEN_ID
        labels[i, pos] = int(encode_signal(signals[-1], max_modulus))

    puzzle_indices = np.arange(num_examples + 1, dtype=np.int32)
    group_indices = np.arange(num_examples + 1, dtype=np.int32)
    puzzle_identifiers = np.zeros(num_examples, dtype=np.int32)

    return {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": puzzle_identifiers,
        "puzzle_indices": puzzle_indices,
        "group_indices": group_indices,
    }


def save_split(split_name: str, data: dict[str, np.ndarray], metadata: PuzzleDatasetMetadata, output_dir: str):
    save_dir = os.path.join(output_dir, split_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    for key, value in data.items():
        np.save(os.path.join(save_dir, f"all__{key}.npy"), value)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    test_modulus = config.test_modulus if config.test_modulus is not None else config.train_modulus
    max_modulus = max(config.train_modulus, test_modulus)
    vocab_size = 2 + max_modulus + config.noise_range
    seq_len = 2 * (config.num_terms - 1) + 1

    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=vocab_size,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=0,
        mean_puzzle_examples=1,
        sets=["all"],
    )

    train_rng = np.random.default_rng(config.seed)
    test_rng = np.random.default_rng(config.seed + 1)

    train_data = build_split(config.train_size, config.train_modulus, max_modulus, config, train_rng)
    test_data = build_split(config.test_size, test_modulus, max_modulus, config, test_rng)

    metadata.total_groups = int(train_data["group_indices"].size - 1)
    save_split("train", train_data, metadata, config.output_dir)

    metadata.total_groups = int(test_data["group_indices"].size - 1)
    save_split("test", test_data, metadata, config.output_dir)

    identifiers = [
        "<blank>",
        "<query>",
        *[f"S{i}" for i in range(max_modulus)],
        *[f"N{i}" for i in range(config.noise_range)],
    ]
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(identifiers, f)


if __name__ == "__main__":
    cli()
