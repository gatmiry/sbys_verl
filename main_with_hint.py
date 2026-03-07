"""
veRL Training Entry Point for bolt-ray — HintDataset variant

Uses HintDataset (hint_dataset.py) for dynamic hint-level adjustment
via on_batch_end. The dataset preprocessor (similardataset.py) outputs
sbys_solution as a top-level column and is_validation in extra_info.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import apple_bolt as bolt
import ray
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(process)d:%(name)s:%(lineno)s:%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


################################################################################
# Khashayar's ADD-ONs
################################################################################
DATASET_NAME = "similar_dataset"
PROCESSED_DATASET_SAVE_PATH = os.path.expanduser(f"~/data/{DATASET_NAME}")
TOTAL_EPOCHS = 600
    
import argparse
import random
from typing import Dict, Any
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))



################################################################################
# END OF Khashayar's ADD-ONs
################################################################################

# Load cluster configuration from config.yml
def load_cluster_config():
    """Load cluster configuration from config.yml."""
    config_path = Path(__file__).parent / "config-b200.yml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    worker_config = config["ray_cluster_spec"]["roles"]["worker"]["resources"]

    num_nodes = worker_config["max_node"]
    task_type = worker_config["task_type"]
    gpus_per_node = int(task_type.replace("gpu", ""))

    return num_nodes, gpus_per_node


# Global variables for nodes number and gpus per node
num_nodes, gpus_per_node = load_cluster_config()


def wait_for_resources_with_placement_group():
    """Wait for GPU resources using Ray placement groups."""
    logger.info(
        f"Creating placement group for {num_nodes} nodes with {gpus_per_node} GPUs each..."
    )

    bundles = [{"GPU": gpus_per_node} for _ in range(num_nodes)]

    pg = ray.util.placement_group(
        bundles=bundles,
        strategy="SPREAD",
        name="verl_training_pg",
    )

    logger.info(f"Placement group created: {pg}")
    logger.info("Waiting for placement group to be ready...")

    try:
        ready = pg.wait(timeout_seconds=1800)
        if not ready:
            raise RuntimeError(
                f"Timeout waiting for placement group with {num_nodes} nodes "
                f"and {gpus_per_node} GPUs per node."
            )
        logger.info("Placement group ready! All GPU resources are available.")

        logger.info("Freeing reserved resources for veRL to use...")
        ray.util.remove_placement_group(pg)

        logger.info(f"Cluster resources: {ray.cluster_resources()}")
        logger.info(f"Available resources: {ray.available_resources()}")

        return

    except Exception as e:
        logger.error(f"Failed to create placement group: {e}")
        ray.util.remove_placement_group(pg)
        raise


def download_datasets():
    """Load and preprocess the similar dataset using similardataset.py."""
    logger.info("Starting similar dataset preprocessing...")

    @ray.remote
    def load_similar_dataset():
        print("Preprocessing similar dataset...")
        logger.info("Preprocessing similar dataset...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(
            [sys.executable, "-m", "verl_training.datasets.similardataset",
             "--local_save_dir", PROCESSED_DATASET_SAVE_PATH],
            capture_output=True,
            text=True,
            cwd=script_dir,
        )
        if result.returncode != 0:
            logger.error(f"Similar dataset prep failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
            print(f"Similar dataset prep failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
            raise RuntimeError(f"Similar dataset prep failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        logger.info("Similar dataset preprocessed successfully")
        print("Similar dataset preprocessed successfully")
        return result.stdout

    try:
        task = load_similar_dataset.remote()
        output = ray.get([task])
        logger.info("Dataset loaded successfully")
        logger.debug(f"Similar dataset output: {output}")
    except Exception as e:
        logger.error(f"Dataset load failed: {e}")
        raise


def run_training():
    """Execute the veRL training pipeline with HintDataset."""
    logger.info("Starting veRL training with HintDataset...")

    checkpoint_dir = os.path.join(
        bolt.ARTIFACT_DIR_PARENT, "qwen3-4b-instruct-2507-hint-checkpoint"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default=PROCESSED_DATASET_SAVE_PATH)
    parser.add_argument("--total_epochs", default=TOTAL_EPOCHS)
    args, _ = parser.parse_known_args()
    local_save_dir = args.local_save_dir
    total_epochs = args.total_epochs

    train_files = f"['{local_save_dir}/train.parquet']"
    test_files = f"['{local_save_dir}/test.parquet']"

    hint_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hint_dataset.py")

    training_args = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        # Algorithm configuration
        "algorithm.adv_estimator=grpo",
        # Data configuration
        f"data.train_files={train_files}",
        f"data.val_files={test_files}",
        "data.train_batch_size=256",
        "data.max_prompt_length=4096",
        "data.max_response_length=24576",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        # Custom dataset class — HintDataset with on_batch_end hook
        f"data.custom_cls.path={hint_dataset_path}",
        "data.custom_cls.name=HintDataset",
        # Model and actor configuration
        "actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.ppo_mini_batch_size=128",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        # KL divergence loss configuration
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        # Memory optimization
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        # Rollout configuration (vLLM)
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "actor_rollout_ref.rollout.max_num_batched_tokens=32768",
        "actor_rollout_ref.rollout.n=4",
        "actor_rollout_ref.rollout.val_kwargs.n=5",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        # Reference model configuration
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.ref.fsdp_config.param_offload=False",
        # Reward model disabled (using rule-based compute_score instead)
        "reward_model.enable=False",
        # Custom reward function for rule-based validation
        f"custom_reward_function.path={os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verl_training', 'reward_fn.py')}",
        "custom_reward_function.name=compute_score",
        # Algorithm settings
        "algorithm.use_kl_in_reward=False",
        # Trainer configuration
        "trainer.critic_warmup=0",
        'trainer.logger=["console", "wandb"]',
        "trainer.project_name=verl_grpo_hint_guided",
        "trainer.experiment_name=qwen3_4b_instruct_grpo_hint_guided",
        f"trainer.default_local_dir={checkpoint_dir}",
        f"trainer.n_gpus_per_node={gpus_per_node}",
        f"trainer.nnodes={num_nodes}",
        "trainer.save_freq=20",
        "trainer.test_freq=5",
        "trainer.log_val_generations=10",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.rollout_data_dir={os.path.join(bolt.ARTIFACT_DIR, 'training_batches')}",
        f"trainer.validation_data_dir={os.path.join(bolt.ARTIFACT_DIR, 'validation_batches')}",
    ]

    logger.info(f"Training command: {' '.join(training_args)}")

    try:
        result = subprocess.run(
            training_args,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"Training failed with return code {result.returncode}")
            logger.error(f"STDOUT:\n{result.stdout[-5000:] if result.stdout else 'None'}")
            logger.error(f"STDERR:\n{result.stderr[-5000:] if result.stderr else 'None'}")
            raise RuntimeError(f"verl training failed. STDERR (last 2000 chars):\n{result.stderr[-2000:] if result.stderr else 'None'}")
        logger.info("Training completed successfully!")
        return result.returncode
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Training subprocess error: {e}")
        raise


def main():
    """Main execution flow."""
    logger.info("=" * 80)
    logger.info("veRL Training with HintDataset - bolt-ray Edition")
    logger.info("=" * 80)

    logger.info("Connecting to Ray cluster...")
    ray.init()

    logger.info(f"Ray cluster resources: {ray.cluster_resources()}")
    logger.info(f"Ray available resources: {ray.available_resources()}")

    wait_for_resources_with_placement_group()

    # Step 1: Preprocess dataset
    download_datasets()

    # Step 2: Run training
    run_training()

    logger.info("=" * 80)
    logger.info("All tasks completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
