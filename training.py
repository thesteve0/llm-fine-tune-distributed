import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig
from aim.hugging_face import AimCallback

def setup_distributed():
    """Setup distributed training environment for Accelerate integration"""
    # Get distributed training parameters from environment
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "23456")

    # Set distributed environment variables for Accelerate/Transformers
    if world_size > 1:
        print(f"Setting up distributed training environment for Accelerate...")
        print(f"Rank {rank}/{world_size}, local_rank {local_rank}")
        print(f"Master: {master_addr}:{master_port}")

        # Ensure environment variables are properly set for Accelerate
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)

        print("Distributed environment configured - letting Accelerate handle initialization")
    else:
        print("Single-node training mode")

    return world_size, rank, local_rank

def cleanup_distributed():
    """Clean up distributed training"""
    # Accelerate handles cleanup automatically
    print("Distributed training cleanup - handled by Accelerate")

def main():
    # Initialize distributed training
    world_size, rank, local_rank = setup_distributed()

    # Configuration from environment variables
    model_name = "HuggingFaceTB/SmolLM3-3B"
    dataset_path = "data/qa_dataset.parquet"
    epochs = int(os.getenv("EPOCHS", "4"))
    batch_size = int(os.getenv("BATCH_SIZE", "8"))  # Reduced for distributed training
    learning_rate = float(os.getenv("LEARNING_RATE", "5e-5"))
    data_dir = os.getenv("DATA_DIR", "/tmp/data")
    output_dir = os.getenv("OUTPUT_DIR", "/tmp/models")

    # Create output directories (only on master node)
    if rank == 0:
        os.makedirs(f"{output_dir}/best_model", exist_ok=True)

    print(f"Configuration:")
    print(f"- Model: {model_name}")
    print(f"- Dataset: {dataset_path}")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Data dir: {data_dir}")
    print(f"- Output dir: {output_dir}")

    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available - distributed training requires GPUs")
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load Model and Tokenizer
        print("Loading model and tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp/model_cache")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("Tokenizer loaded successfully")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir="/tmp/model_cache",
            attn_implementation="flash_attention_2",
        )

        model = model.to(device)
        print(f"Model loaded on device: {device}")

        # Monitor VRAM usage after model loading
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"VRAM after model loading: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        # Freeze all layers except the last ones for memory efficiency
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last 2 layers and output layer
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layers = model.model.layers
                for param in layers[-2:].parameters():
                    param.requires_grad = True
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                layers = model.transformer.h
                for param in layers[-2:].parameters():
                    param.requires_grad = True
            elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
                layers = model.gpt_neox.layers
                for param in layers[-2:].parameters():
                    param.requires_grad = True
            else:
                for param in model.parameters():
                    param.requires_grad = True

            # Unfreeze output embedding layer
            if hasattr(model, 'embed_out'):
                for param in model.embed_out.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'lm_head'):
                for param in model.lm_head.parameters():
                    param.requires_grad = True

        except Exception as e:
            for param in model.parameters():
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    try:
        # Load and Prepare Dataset
        print(f"Loading Q&A dataset from: {dataset_path}")

        dataset = load_dataset("parquet", data_files=dataset_path)
        full_dataset = dataset["train"]

        print(f"Total dataset size: {len(full_dataset):,} Q&A pairs")

        split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']

        print(f"Training samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(val_dataset):,}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    # Define wilderness survival expert system prompt
    WILDERNESS_EXPERT_SYSTEM_PROMPT = """You are a wilderness survival and practical skills expert. Your mission is to provide comprehensive, detailed guidance on essential survival and practical skills. Give thorough, step-by-step instructions with explanations of why each step matters.

Your expertise covers:
- Wilderness Survival Basics: Rule of 3s (3 minutes without air, 3 hours without shelter in harsh conditions, 3 days without water, 3 weeks without food), emergency signaling techniques, essential knots, identifying poisonous plants and safe alternatives
- Basic First Aid: Treatment for cuts, burns, sprains, shock, and emergency care procedures
- Simple Car Maintenance: Checking fluids (oil, coolant, brake, transmission), tire inspection and pressure, lights and electrical systems
- Basic Cooking Techniques: Food safety, preparation methods, cooking over open fires, food preservation
- Common Measurement Conversions: Imperial to metric, cooking measurements, distance and weight conversions
- Essential Knots: Bowline, clove hitch, trucker's hitch, figure-eight, sheet bend, and their practical applications

Always provide detailed explanations, safety warnings when relevant, and multiple approaches when possible. Your responses should be comprehensive enough to help someone learn and apply these skills safely and effectively. Aim for thorough, educational responses rather than brief answers."""

    # Preprocessing function to format the Q&A data for TRL SFTTrainer
    def format_prompt(example):
        question = example['full-question']
        answer = example['answer']

        messages = [
            {"role": "system", "content": WILDERNESS_EXPERT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]

        return {"messages": messages}

    try:
        # Apply the formatting to train and validation datasets for TRL
        print("Processing datasets for TRL SFTTrainer...")
        processed_train = train_dataset.map(format_prompt)
        processed_val = val_dataset.map(format_prompt)

        print(f"Training samples: {len(processed_train):,}")
        print(f"Validation samples: {len(processed_val):,}")

    except Exception as e:
        print(f"Error processing datasets: {e}")
        raise

    # Custom trainer callback to track training history
    class TrainingHistoryCallback(TrainerCallback):
        def __init__(self):
            self.history = []

        def on_log(self, args, state, control, model=None, logs=None, **kwargs):
            if logs:
                self.history.append(logs)

    # Custom callback to track additional metrics like perplexity
    class PerplexityCallback(TrainerCallback):
        def on_log(self, args, state, control, model=None, logs=None, **kwargs):
            """Calculate and log perplexity from loss values"""
            if logs:
                import math
                # Add perplexity for training loss
                if 'loss' in logs:
                    logs['perplexity'] = math.exp(logs['loss'])
                # Add perplexity for evaluation loss
                if 'eval_loss' in logs:
                    logs['eval_perplexity'] = math.exp(logs['eval_loss'])

    history_callback = TrainingHistoryCallback()
    perplexity_callback = PerplexityCallback()

    # Initialize Aim callback for experiment tracking
    aim_repo = os.getenv("AIM_REPO", "/aim")
    aim_callback = AimCallback(repo=aim_repo, experiment='smollm3-wilderness-finetuning-distributed')

    try:
        # Set Up SFTTrainer with TRL for chat model fine-tuning
        import transformers
        import trl

        # Configure distributed training arguments
        distributed_args = {}
        if world_size > 1:
            distributed_args.update({
                "ddp_find_unused_parameters": False,
                "ddp_bucket_cap_mb": 50,  # Larger buckets for fewer network syncs
                "local_rank": local_rank,
            })
            print("Using NCCL backend for DDP training")

        training_args = SFTConfig(
            output_dir=f"{output_dir}/checkpoints",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Balance between sync frequency and training steps
            learning_rate=learning_rate * world_size if world_size > 1 else learning_rate,  # Scale learning rate
            max_grad_norm=1.0,
            num_train_epochs=epochs,
            logging_steps=2,  # Log metrics every 2 steps for more frequent updates
            logging_first_step=True,  # Log immediately so Aim shows data right away
            save_steps=500,
            bf16=True,
            eval_strategy="steps",
            eval_steps=10,  # Evaluate every 10 steps (was 100, which never ran)
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            dataloader_drop_last=True,
            max_seq_length=1024,
            packing=False,
            # Explicitly configure distributed training for Accelerate
            ddp_backend="nccl" if world_size > 1 else None,
            **distributed_args
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_train,
            eval_dataset=processed_val,
            callbacks=[history_callback, perplexity_callback, aim_callback],
        )

        # Start Fine-Tuning
        print("Starting fine-tuning on Q&A dataset with SmolLM3-3B...")

        training_result = trainer.train()
        print("Fine-tuning complete!")

    except Exception as e:
        print(f"Error during training: {e}")
        raise

    try:
        # Save the best model and training artifacts (only on rank 0 in distributed training)
        if rank == 0:
            trainer.save_model(f"{output_dir}/best_model")
            tokenizer.save_pretrained(f"{output_dir}/best_model")
            print(f"Best model saved to {output_dir}/best_model")

            # Save training history as JSON
            with open(f"{output_dir}/training_history.json", "w") as f:
                json.dump(history_callback.history, f, indent=2)

            # Save training summary
            training_summary = {
                "model_name": model_name,
                "dataset_path": dataset_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "training_samples": len(train_dataset),
                "validation_samples": len(val_dataset),
                "final_train_loss": training_result.training_loss if hasattr(training_result, 'training_loss') else None,
                "world_size": world_size,
                "distributed_training": world_size > 1
            }

            with open(f"{output_dir}/training_summary.json", "w") as f:
                json.dump(training_summary, f, indent=2)

    except Exception as e:
        print(f"Error saving artifacts: {e}")
        raise

    # Cleanup distributed training
    cleanup_distributed()

    if rank == 0:
        print(f"\nDistributed Q&A fine-tuning completed successfully!")
        print(f"Training artifacts saved to {output_dir}/")
        print(f"Distributed training used {world_size} GPUs")

if __name__ == "__main__":
    main()