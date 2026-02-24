# veRL Training Example

This example demonstrates distributed reinforcement learning training using the veRL framework on the bolt-ray platform. It trains a DeepSeek-LLM-7B-Chat model on GSM8K and MATH datasets using the GRPO (Group Relative Policy Optimization) algorithm.

## Training Overview

- **Model**: DeepSeek-LLM-7B-Chat (7 billion parameters)
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Datasets**: GSM8K (grade school math) + MATH (competition math)
- **Parallelism**: FSDP (Fully Sharded Data Parallel) with tensor model parallelism
- **Inference Engine**: vLLM for efficient rollout generation

## Directory Structure

```
verl/
├── config.yml                          # bolt-ray cluster configuration
├── setup.sh                            # Dependency installation script
├── main.py                             # Main entry point (adopted from run_deepseek7b_llm_math.sh script in veRL)
├── README.md                           # This file
└── verl_training/                      # Training package
    └── datasets/
        ├── gsm8k.py                    # GSM8K dataset preprocessing
        └── math_dataset.py             # MATH dataset preprocessing
```

## Usage

### Submitting the Training Job

Submit the training job using the `bolt` command:

```bash
bolt rayjob submit --config config.yml --tar .
```

For running on B200 GPUs:
```bash
bolt rayjob submit --config config-b200.yml --tar .
```

### Checkpointing

Checkpoint configuration is located in `main.py`. The checkpoint directory is created in line 79-80, and used in line 139. In this example, the checkpoint located in `bolt.ARTIFACT_DIR`
