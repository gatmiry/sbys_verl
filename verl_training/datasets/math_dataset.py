# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os

import sys
from typing import List, Dict

import datasets
from verl.utils.hdfs_io import copy, makedirs

DATASET_NAME = "math_dataset"
PROCESSED_DATASET_SAVE_PATH = os.path.expanduser(f"~/data/{DATASET_NAME}")
VALIDATION_SIZE = 128

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
from utils import load_system_prompt

SYSTEM_PROMPT_SIMPLE = "full_solution_simple"

def format_prompt(problem: str) -> List[Dict[str, str]]:
    """Format a problem into chat messages (system + user)."""
    system_prompt = load_system_prompt(SYSTEM_PROMPT_SIMPLE)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Problem: {problem}"},
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="The local path to the raw dataset, if it exists.",
    )
    parser.add_argument(
        "--local_save_dir",
        default=PROCESSED_DATASET_SAVE_PATH,
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the {data_source} dataset...", flush=True)
    if local_dataset_path is not None:
        dataset = datasets.load_from_disk(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source, split="train")

    import random
    indices = list(range(len(dataset)))
    random.Random(42).shuffle(indices)
    validation_indices = indices[:VALIDATION_SIZE]
    training_indices = indices[VALIDATION_SIZE:]
    train_dataset = dataset.select(training_indices)
    test_dataset = dataset.select(validation_indices)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")
            answer = example.pop("solution")
            prompt = format_prompt(question)
            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "problem": question,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print(
            "Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead."
        )
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
