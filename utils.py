import os
import wandb
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score, accuracy_score

from datasets import load_dataset

## ---------------------------------
#                CONSTANTS
## ---------------------------------
IDX2LBL = {
  0: "risk",
  1: "neutral",
  2: "opportunity"
}

LBL2IDX = {
  "risk": 0,
  "neutral": 1,
  "opportunity": 2
}

SYSTEM_PROMPT = """
  Analyze the sentiment of the user provided content and determine if 
  it is describing risk, opportunity, or neutral sentiment related to 
  climate and the environment. Your response should be the corresponding 
  sentiment label "risk" or "opportunity" or "neutral".
"""

## ---------------------------------
#                FXNS
## ---------------------------------

def init_wandb(project_name):
    # Initialise wandb.config via Pytorch-Lightning
    logger = WandbLogger(project=project_name, name=None)
    _ = logger.experiment.config
    _ = logger.experiment.path

    # Check Flash Attention
    if wandb.config.use_attn == 'flash_attention_2':
        if torch.cuda.get_device_capability()[0] < 8:
            print('Hardware not supported for Flash Attention. Using "sdpa" instead')
            wandb.config.use_attn = 'sdpa'

    # Update run name
    timestamp = datetime.now().strftime('%y%m%d-%H%M')
    run_name = f"{timestamp}-{wandb.config.lr_schedule}-{wandb.config.optimizer}-A{wandb.config.alpha}"\
               f"-D{wandb.config.dropout}-R{wandb.config.rank}-S{wandb.config.max_seq}"
    if wandb.config.use_attn == 'flash_attention_2':
        run_name += "-FA"
    wandb.run.name = run_name
    wandb.config['run_name'] = run_name

    # Create output dir
    output_dir = os.path.join(wandb.config.output_dir, run_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Checkpoint directory created: {output_dir}')
        wandb.config['ckpt_dir'] = output_dir
        
        print("WandB Logger Initialized")
        return logger, output_dir
    else:
        raise Exception("Error: Failed to initialise wandb.config")

def resume_wandb(project_name, run_path):
    run_id = run_path.split("/")[-1]
    # restore config and logger
    wandb.restore(name='config.yaml',run_path=run_path, replace = True, root='./tmp')
    logger = WandbLogger(project=project_name, id=run_id, resume="must")
    _ = logger.experiment.config
    wandb.config.update("./tmp/config.yaml", allow_val_change=True)
    
    return logger, wandb.config['ckpt_dir']

def create_prompt(sample):
  return {
    "messages": [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": sample["text"]},
      {"role": "assistant", "content": IDX2LBL[sample["label"]]}
    ]
  }
  

def build_prompts(dataset_id):

    print(f"Building prompts from {dataset_id}...")
    # Load dataset from the hub
    dataset = load_dataset(dataset_id)

    # Shuffle and convert to ChatML prompt format
    tmp_data = dataset['train'].shuffle().map(create_prompt, remove_columns=dataset['train'].features, batched=False)
    test_data = dataset['test'].shuffle().map(create_prompt, remove_columns=dataset['test'].features, batched=False)

    # Split to train/val/test & save prompts to disk
    train_data = tmp_data.train_test_split(test_size=0.2)
    train_data['train'].to_json("data/train_prompts.json", orient="records")
    train_data['test'].to_json("data/val_prompts.json", orient="records")
    test_data.to_json("data/test_prompts.json", orient="records")
    print("train/val/test prompts saved to disk")

def get_preds(test_prompts, pipe):
    preds = []
    for sample in tqdm(test_prompts.select(range(test_prompts.num_rows))):
      prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
      outputs = pipe(prompt, max_new_tokens=10, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
      sample_pred = outputs[0]['generated_text'][len(prompt):].strip()
      if "opportunity" in sample_pred:
          preds.append(LBL2IDX["opportunity"])
      elif "risk" in sample_pred:
          preds.append(LBL2IDX["risk"])
      elif "neutral" in sample_pred:
          preds.append(LBL2IDX["neutral"])
      else:
          preds.append(LBL2IDX["neutral"])
      # preds.append(LBL2IDX[sample_pred])
    return preds

def get_labels(test_prompts):
    labels = []
    for sample in tqdm(test_prompts.select(range(test_prompts.num_rows))):
      sample_label = sample["messages"][2]["content"]
      labels.append(LBL2IDX[sample_label])
    return labels

def evaluate(y_true, y_preds, output_dir, log_to_wandb):

    plot_roc_curve(y_true, y_preds, output_dir, log_to_wandb)

    test_rec = recall_score(y_true, y_preds, average='micro')
    test_prec = precision_score(y_true, y_preds, average='micro')
    test_f1 = f1_score(y_true, y_preds, average='micro')
    test_acc = accuracy_score(y_true, y_preds)

    print(f"Test recall: {test_rec}")
    print(f"Test precision: {test_prec}")
    print(f"Test f1: {test_f1}")
    print(f"Test accuracy: {test_acc}")

    if log_to_wandb:
      wandb.log({
        "test_recall": test_rec,
        "test_precision": test_prec,
        "test_f1": test_f1,
        "test_accuracy": test_acc
      })

    
def plot_roc_curve(targets, preds, output_dir, log_to_wandb):
    """
    Adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    # Convert lists to numpy arrays
    preds = np.array(preds)
    targets = np.array(targets)

    # Set classes
    n_classes = len(np.unique(targets))
    
    # Binarize the targets/preds
    binarized_targets = label_binarize(targets, classes=np.arange(n_classes))
    binarized_preds = label_binarize(preds, classes=np.arange(n_classes))

    # Compute the ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(binarized_targets[:,i], binarized_preds[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(binarized_targets.ravel(), binarized_preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot the ROC curves for each class
    fig = plt.figure(figsize=(10,6))
    colors = plt.cm.tab10.colors
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC (class: {0}) (AUC = {1:0.2f})'.format(IDX2LBL[i], roc_auc[i]))

    # Plot the micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=2,
            label='Micro-average ROC (AUC = {0:0.2f})'.format(roc_auc["micro"]))

    # Plot the random chance line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Set the plot limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc="lower right")

    plt.savefig(f"{output_dir}/roc.jpg")

    if log_to_wandb:
      wandb.log({'ROC AUC':wandb.Image(fig)})
    