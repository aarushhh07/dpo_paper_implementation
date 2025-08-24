import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


model = AutoModelForCausalLM.from_pretrained("/kaggle/input/dpo-model/gpt2-medium-ref-final").to(device)
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/dpo-model/gpt2-medium-ref-final")


dataset = load_dataset("Anthropic/hh-rlhf", split="test")


def log_likelihood(model, tokenizer, text: str, device):
   
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    labels = input_ids.clone()  
    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss  

    n_tokens = attention_mask.sum().item()
    return -loss.item() * n_tokens  

def evaluate_pair(model, tokenizer, example, device):
    chosen_ll = log_likelihood(model, tokenizer, example["chosen"], device)
    rejected_ll = log_likelihood(model, tokenizer, example["rejected"], device)
    return chosen_ll > rejected_ll

subset = dataset.select(range(4000)) 
correct = sum(evaluate_pair(model, tokenizer, ex, device) for ex in subset)
print(f"Pairwise accuracy: {correct/len(subset):.3f}")