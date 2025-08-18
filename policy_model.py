import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,DataCollatorForLanguageModeling,Trainer,TrainingArguments


dataset_dpo=load_dataset("Anthropic/hh-rlhf",split ='train')
dataset_dpo.shape
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/sft-model/gpt2-medium-ref-final")
def tokenize_dpo(batch):
    tokenized_chosen = tokenizer(batch["chosen"],padding="max_length",truncation='longest_first',
                                 max_length=128)
    tokenized_rejected = tokenizer(batch["rejected"],padding="max_length",truncation='longest_first',
                            max_length=128)
    return {"chosen_input_ids": tokenized_chosen["input_ids"],
            "chosen_attention_mask": tokenized_chosen["attention_mask"],
                    "rejected_input_ids": tokenized_rejected["input_ids"],
            "rejected_attention_mask": tokenized_rejected["attention_mask"]}

tokenized_dataset_dpo = dataset_dpo.map(
    tokenize_dpo,
    batched=True,
    remove_columns=dataset_dpo.column_names
)

def sequence_log_probs(model, encodings,detach):
    
    token_ids=encodings["input_ids"]
    attention_mask=encodings["attention_mask"]
    if detach:
        with torch.no_grad():
            outputs = model(input_ids=token_ids, attention_mask=attention_mask)
            logits=outputs.logits
            log_probs=torch.log_softmax(logits,dim=-1)
    else:
        outputs = model(input_ids=token_ids, attention_mask=attention_mask)
        logits=outputs.logits
        log_probs=torch.log_softmax(logits,dim=-1)

    shift_logits = log_probs[:, :-1, :]                     
    shift_labels = token_ids[:, 1:]                         
    shift_mask   = attention_mask[:, 1:] 

    token_log_probs = shift_logits.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * shift_mask
    seq_log_probs_sums = token_log_probs.sum(dim=-1)

    return seq_log_probs_sums

def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):  # Copied from the paper
    """
    pi_logps: policy logprobs, shape (B,)
    ref_logps: reference model logprobs, shape (B,)
    yw_idxs: preferred completion indices in [0, B-1], shape (T,)
    yl_idxs: dispreferred completion indices in [0, B-1], shape (T,)
    beta: temperature controlling strength of KL penalty
    Each pair of (yw_idxs[i], yl_idxs[i]) represents the
    indices of a single preference pair.
    """
    pi_yw_logps, pi_yl_logps = pi_logps[yw_idxs], pi_logps[yl_idxs]
    ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs]
    pi_logratios = pi_yw_logps - pi_yl_logps
    ref_logratios = ref_yw_logps - ref_yl_logps
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    rewards = beta * (pi_logps - ref_logps).detach()
    return losses, rewards

from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader

tokenized_dataset_dpo.set_format(type="torch")

dataloader = DataLoader(tokenized_dataset_dpo, batch_size=4, shuffle=True)
reference_model = AutoModelForCausalLM.from_pretrained("/kaggle/input/sft-model/gpt2-medium-ref-final")
policy_model = AutoModelForCausalLM.from_pretrained("/kaggle/input/sft-model/gpt2-medium-ref-final")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_model = policy_model.to(device)
reference_model = reference_model.to(device)
for param in reference_model.parameters():
    param.requires_grad = False

policy_model.gradient_checkpointing_enable()

reference_model.eval()  
policy_model.train()
optimizer = AdamW(policy_model.parameters(), lr=1e-5)

beta = 0.1  
num_epochs = 3
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    for batch in tqdm(dataloader):  
        yw_tokens = {
        "input_ids": batch["chosen_input_ids"].to(device),
        "attention_mask": batch["chosen_attention_mask"].to(device)
        }
        yl_tokens = {
        "input_ids": batch["rejected_input_ids"].to(device),
        "attention_mask": batch["rejected_attention_mask"].to(device)
        }
        combined_tokens = {
            "input_ids":      torch.cat([yw_tokens["input_ids"],      yl_tokens["input_ids"]],      dim=0),
            "attention_mask": torch.cat([yw_tokens["attention_mask"], yl_tokens["attention_mask"]], dim=0),
        }
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                ref_logps = sequence_log_probs(reference_model, combined_tokens, detach=True)
                

        with torch.cuda.amp.autocast():
            pi_logps = sequence_log_probs(policy_model, combined_tokens, detach=False)
            

        batch_size = yw_tokens["input_ids"].shape[0]
        yw_idxs = torch.arange(0, batch_size, device=device)
        yl_idxs = torch.arange(batch_size, 2*batch_size, device=device)

        losses,rewards = dpo_loss(pi_logps,ref_logps,yw_idxs,yl_idxs,beta)
        loss=losses.mean()

        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
   


        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


policy_model.save_pretrained("/kaggle/working/gpt2-medium-policy-final")
tokenizer.save_pretrained("/kaggle/working/gpt2-medium-policy-final")