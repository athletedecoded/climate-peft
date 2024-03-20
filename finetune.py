import os
import torch
import wandb
import random
import argparse
import warnings

from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import setup_chat_format, SFTTrainer
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline

from codecarbon import EmissionsTracker

from utils import init_wandb, build_prompts, get_preds, get_labels, evaluate

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='climate-sentiment')
    parser.add_argument('--project', type=str, required=True, help='wandb project name')
    parser.add_argument('--resume', type=str, required=False, help='wandb run path to resume training')
    return parser.parse_args()

def main():
    # Reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Manage CUDA memory
    torch.cuda.empty_cache()
    disable_caching()

    # Init WandB
    args = parse_args()
    logger, output_dir = init_wandb(args.project)

    # Init CodeCarbon Tracker
    cc_tracker = EmissionsTracker(project_name=wandb.run.name)

    # Dataset
    if wandb.config.build_prompts:
        if not os.path.exists("./data"):
            print(f"Making dir ./data")
            os.makedirs("./data") 
        _ = build_prompts(wandb.config.hf_dataset)

    # GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Finetuning on {device}")

    # BitsAndBytes Quantized Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # LoRA Config: see https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
    peft_config = LoraConfig(
        lora_alpha=wandb.config.alpha,
        lora_dropout=wandb.config.dropout,
        r=wandb.config.rank,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # Model Config
    model = AutoModelForCausalLM.from_pretrained(
        wandb.config.hf_model,
        device_map=device,
        attn_implementation=wandb.config.use_attn,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1 # tensor parallelism state

    # Tokenizer Config
    tokenizer = AutoTokenizer.from_pretrained(wandb.config.hf_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ChatML Config
    model, tokenizer = setup_chat_format(model, tokenizer)

    print("--------------------------------")
    print("          FINETUNING            ")
    print("--------------------------------")
    # Training Args
    args = TrainingArguments(
        output_dir=output_dir,                          # directory to save and repository id
        num_train_epochs=wandb.config.num_epochs,       # number of training epochs
        per_device_train_batch_size=1,                  # batch size per device during training
        gradient_accumulation_steps=4,                  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,                    # use gradient checkpointing to save memory
        optim=wandb.config.optimizer,                   # use fused adamw optimizer
        logging_steps=10,                               # log every 10 steps
        save_strategy="epoch",                          # save checkpoint every epoch
        evaluation_strategy="epoch",                    # save checkpoint every epoch
        learning_rate=wandb.config.lr,                  # learning rate, based on QLoRA paper
        bf16=True,                                      # use bfloat16 precision
        tf32=True,                                      # use tf32 precision
        max_grad_norm=0.3,                              # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                              # warmup ratio based on QLoRA paper
        lr_scheduler_type=wandb.config.lr_schedule,     # use cosine annealing
        report_to="wandb",                              # report metrics to wandb
        load_best_model_at_end=True,                    # find best model checkpoint       
        hub_model_id=wandb.config.hub_id,               # HF Hub ID for adapter
    )

    # Load train/val prompts from disk
    train_prompts = load_dataset("json", data_files="data/train_prompts.json", split='train')
    print(f"Loaded {train_prompts.num_rows} training samples")
    val_prompts = load_dataset("json", data_files="data/val_prompts.json", split='train')
    print(f"Loaded {val_prompts.num_rows} validation samples")

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_prompts,
        eval_dataset=val_prompts,
        peft_config=peft_config,
        max_seq_length=wandb.config.max_seq,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False, # No need to add additional separator token
        }
    )

    # Run finetuning & track with CodeCarbon
    cc_tracker.start()
    trainer.train()
    co2 = cc_tracker.stop()
    wandb.log({"Training CO2 emissions (kg)": co2})

    # save best model
    trainer.save_model()

    # Push adapter to HF Hub
    if wandb.config.push_adapter_to_hub:
        trainer.push_to_hub()
        print(f"Adapter pushed to HF Hub {wandb.config.hub_id}")

    # Free CUDA memory
    del model
    del trainer
    torch.cuda.empty_cache()

    # Merge adapter w/ base model and save
    # print("Merging adapter w/ base model")
    # merged_model = adapter.merge_and_unload()
    # merged_model.save_pretrained(f"{output_dir}/merged_model", safe_serialization=True, max_shard_size="2GB")
    # print(f"Model saved to: {output_dir}/merged_model")
    # tokenizer.save_pretrained(f"{output_dir}/merged_model")
    # print(f"Tokenizer saved to: {output_dir}/merged_model")

    print("--------------------------------")
    print("          EVALUATION            ")
    print("--------------------------------")

    # Load model with PEFT adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # Load test dataset
    test_prompts = load_dataset("json", data_files="data/test_prompts.json", split="train")
    print(f"Loaded {test_prompts.num_rows} test samples")

    # Load model into pipeline
    pipe = pipeline(
        task="text-generation", 
        model=model, 
        tokenizer=tokenizer
    )

    # Test Preds
    y_preds = get_preds(test_prompts, pipe)
    y_true = get_labels(test_prompts)

    # Evaluate
    evaluate(y_true, y_preds, output_dir, log_to_wandb=True)

if __name__ == '__main__':
    main()