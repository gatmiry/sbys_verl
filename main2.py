"""
veRL Training Entry Point for bolt-ray
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
PROCESSED_DATASET_FOLDER = os.path.expanduser("~/data")
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
    # TODO: Could explore native veRL PG support to directly re-use the PG created here.
    logger.info(
        f"Creating placement group for {num_nodes} nodes with {gpus_per_node} GPUs each..."
    )

    # Create bundles - one bundle per node
    bundles = [{"GPU": gpus_per_node} for _ in range(num_nodes)]

    pg = ray.util.placement_group(
        bundles=bundles,
        strategy="SPREAD",
        name="verl_training_pg",  # Named for easier debugging
    )

    logger.info(f"Placement group created: {pg}")
    logger.info("Waiting for placement group to be ready...")

    # Wait for placement group to be ready
    try:
        ready = pg.wait(timeout_seconds=1800)  # 30 minutes
        if not ready:
            raise RuntimeError(
                f"Timeout waiting for placement group with {num_nodes} nodes "
                f"and {gpus_per_node} GPUs per node."
            )
        logger.info("Placement group ready! All GPU resources are available.")

        logger.info("Freeing reserved resources for veRL to use...")
        ray.util.remove_placement_group(pg)

        # Log the placement group details
        logger.info(f"Cluster resources: {ray.cluster_resources()}")
        logger.info(f"Available resources: {ray.available_resources()}")

        return

    except Exception as e:
        logger.error(f"Failed to create placement group: {e}")
        ray.util.remove_placement_group(pg)
        raise


def download_datasets():
    """Download and preprocess GSM8K and MATH datasets in parallel."""
    logger.info("Starting dataset download and preprocessing...")

    # Create data directories
    #home_dir = Path.home()
    #(home_dir / "data/gsm8k").mkdir(parents=True, exist_ok=True)
    #(home_dir / "data/math").mkdir(parents=True, exist_ok=True)

    # Run both dataset downloads in parallel using Ray tasks
    @ray.remote
    def download_gsm8k():
        logger.info("Downloading GSM8K dataset...")
        result = subprocess.run(
            [sys.executable, "-m", "verl_training.datasets.gsm8k"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"GSM8K download failed: {result.stderr}")
            raise RuntimeError("GSM8K dataset download failed")
        logger.info("GSM8K dataset downloaded successfully")
        return result.stdout

    @ray.remote
    def download_math():
        logger.info("Downloading MATH dataset...")
        result = subprocess.run(
            [sys.executable, "-m", "verl_training.datasets.math_dataset"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"MATH download failed: {result.stderr}")
            raise RuntimeError("MATH dataset download failed")
        logger.info("MATH dataset downloaded successfully")
        return result.stdout

    @ray.remote
    def download_omni_math():
        print(f"Downloading OMNI-MATH dataset...")
        logger.info("Downloading OMNI-MATH dataset...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(
            [sys.executable, "-m", "verl_training.datasets.mydataset"],
            capture_output=True,
            text=True,
            cwd=script_dir,
        )
        if result.returncode != 0:
            logger.error(f"OMNI-MATH download failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
            print(f"OMNI-MATH download failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
            raise RuntimeError(f"OMNI-MATH dataset download failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        logger.info("Hint helped dataset downloaded successfully")
        print(f"Hint helped dataset downloaded successfully")
        return result.stdout

    # Execute downloads in parallel
    try:
        hint_task = download_omni_math.remote()
        math_task = download_math.remote()
        # Wait for both to complete
        outputs = ray.get([hint_task, math_task])
        logger.info("All datasets downloaded and preprocessed successfully")
        logger.debug(f"Dataset outputs: {outputs}")
    except Exception as e:
        logger.error(f"Dataset download failed: {e}")
        raise


def run_training():
    """Execute the veRL training pipeline."""
    logger.info("Starting veRL training...")

    # Create checkpoint directory
    checkpoint_dir = os.path.join(
        bolt.ARTIFACT_DIR_PARENT, "qwen3-4b-instruct-2507-jadid-checkpoint"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset paths
    #home_dir = Path.home()
    #gsm8k_train = home_dir / "data/gsm8k/train.parquet"
    #gsm8k_test = home_dir / "data/gsm8k/test.parquet"
    #math_train = home_dir / "data/math/train.parquet"
    #math_test = home_dir / "data/math/test.parquet"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default=PROCESSED_DATASET_FOLDER)
    parser.add_argument("--total_epochs", default=TOTAL_EPOCHS)
    args, _ = parser.parse_known_args()
    data_folder = os.path.expanduser(args.data_folder)
    total_epochs = args.total_epochs

    dataset_names = ["pope_dataset_filtered", "math_dataset"]
    train_file_list = [os.path.join(data_folder, name, "train.parquet") for name in dataset_names]
    test_file_list = [os.path.join(data_folder, name, "test.parquet") for name in dataset_names]

    logger.info(f"Train files: {train_file_list}")
    logger.info(f"Test files: {test_file_list}")

    train_files = "[" + ", ".join(f"'{f}'" for f in train_file_list) + "]"
    test_files = "[" + ", ".join(f"'{f}'" for f in test_file_list) + "]"
    # Training command with all hyperparameters
    # This mirrors the original run_deepseek7b_llm_math.sh script in veRL
    training_args = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        # Algorithm configuration
        "algorithm.adv_estimator=grpo",
        # Data configuration
        f"data.train_files={train_files}",
        f"data.val_files={test_files}",
        "data.train_batch_size=1024",
        "data.max_prompt_length=4096",
        "data.max_response_length=24576",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        # Model and actor configuration
        "actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.ppo_mini_batch_size=256",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        # KL divergence loss configuration
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        # Memory optimization
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        # Rollout configuration (vLLM)
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.7",
        "actor_rollout_ref.rollout.max_num_batched_tokens=32768",
        "actor_rollout_ref.rollout.n=5",
        # Reference model configuration
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
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
        "trainer.project_name=verl_grpo_pope_dataset_guided_hinting",
        "trainer.experiment_name=qwen3_4b_instruct_grpo_pope_dataset_guided_hinting",
        f"trainer.default_local_dir={checkpoint_dir}",
        f"trainer.n_gpus_per_node={gpus_per_node}",
        f"trainer.nnodes={num_nodes}",
        "trainer.save_freq=20",
        "trainer.test_freq=5",
        "trainer.log_val_generations=10",
        f"trainer.total_epochs={total_epochs}",
    ]

    logger.info(f"Training command: {' '.join(training_args)}")

    # Run training (blocking call)
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
    logger.info("veRL Training - bolt-ray Edition")
    logger.info("=" * 80)

    # Initialize Ray connection
    # In bolt-ray, Ray cluster is already running, so we just connect to it
    logger.info("Connecting to Ray cluster...")
    ray.init()

    logger.info(f"Ray cluster resources: {ray.cluster_resources()}")
    logger.info(f"Ray available resources: {ray.available_resources()}")

    # Wait for GPU workers using placement groups
    wait_for_resources_with_placement_group()

    # Step 1: Download datasets
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
