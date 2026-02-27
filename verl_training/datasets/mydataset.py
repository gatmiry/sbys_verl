

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
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
from typing import List, Dict

import datasets
from verl.utils.hdfs_io import copy, makedirs

SYSTEM_PROMPT_WITH_HINT = "full_solution_with_hint"
SYSTEM_PROMPT_SIMPLE = "full_solution_simple"
VALIDATION_SIZE = 128
DATASET_NAME = "pope_dataset_filtered"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "sbys_datasets", "pope_dataset_filtered")
#PROCESSED_DATASET_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "parquet_datasets", DATASET_NAME)
PROCESSED_DATASET_SAVE_PATH = os.path.expanduser(f"~/data/{DATASET_NAME}")

### loading load_system_prompt from utils.py
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
from utils import load_system_prompt
########################################################

def format_prompt(problem: str) -> List[Dict[str, str]]:
    """Format a problem into chat messages (system + user)."""
    system_prompt = load_system_prompt(SYSTEM_PROMPT_SIMPLE)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Problem: {problem}"},
    ]

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument(
    #    "--local_dir",
    #    default=None,
    #    help="The save directory for the preprocessed dataset.",
    #)
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
    
    data_source = "omni_math"

    if local_dataset_path is not None:
        dataset = datasets.load_from_disk(local_dataset_path, "main")
    else:
        print(f'im loading dataset from {DATASET_PATH}')
        dataset = datasets.load_from_disk(DATASET_PATH)

    import random
    indices = list(range(len(dataset)))
    random.Random(42).shuffle(indices)
    validation_indices = indices[:VALIDATION_SIZE]
    training_indices = indices[VALIDATION_SIZE:]
    train_dataset = dataset.select(training_indices)
    test_dataset = dataset.select(validation_indices)

    #instruction_following = (
    #    'Let\'s think step by step and output the final answer after "####".'
    #)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("problem")
            solution = example.pop("answer")
            sbys_solution = example.pop("sbys_solution")
            prompt = format_prompt(question_raw)
            reward_style = "rule"
            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "math",
                "reward_model": {"style": reward_style, "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "sbys_solution": sbys_solution,
                    "problem": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_save_dir
    #if local_save_dir is None:
    #    local_save_dir = PROCESSED_DATASET_SAVE_PATH


    # Expand ~ to home directory
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    print(f"[mydataset] Saving dataset to {local_save_dir}")
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    print(f"[mydataset] Dataset saved to {local_save_dir}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)




