# Climate PEFT

## Environment Config

**Install**

```
$ sudo apt-get install python3-venv
$ python3 -m venv ~/.venv
$ source ~/.venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

**Optional: Enable Flash Attention**

```
!MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

**WandB Login**
```
# Create account at https://wandb.ai/ > Init new project > Copy API key
$ wandb login # paste API key
```

**HuggingFace Hub Login**
```
# Create a HF Access Token > Copy Token
$ huggingface-cli login # paste token
```

## Finetuning w/ PEFT QLoRA

### 1. Runtime Config

Configure model and runtime parameters in [config-defaults.yaml](config-defaults.yaml) 

### 2. Run Finetuning

```
# Activate environment
$ source ~/.venv/bin/activate

# Gotchas: Manage CUDA memory
$ rm -rf ../.cache/huggingface/datasets

# Run finetuning
$ python3 finetune.py --project <wandb_project_name>
# i.e. python3 finetune.py --project climate-peft
```

A WandB Sweep can configure multiple training runs using the `sweep_config.yaml` file then running:
```
$ wandb sweep sweep_config.yml # Returns <wandb_sweep_id>
$ wandb agent <wandb_sweep_id>
```

<hr>

### 3. Model Evaluation



## ToDos

-[] Add resume from ckpt

<hr>