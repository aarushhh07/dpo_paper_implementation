import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,DataCollatorForLanguageModeling,Trainer,TrainingArguments

# Loading the SFT dataset from HuggingFace
dataset_sft=load_dataset("Dahoas/sft-hh-rlhf",split="train")

# Loading the model-gpt2
reference_model = AutoModelForCausalLM.from_pretrained('gpt2-medium')
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
reference_model.gradient_checkpointing_enable()

# Fine-tuning the model 
tokenizer.pad_token = tokenizer.eos_token
def tokenize_sft(batch):
    texts = [p + " " + r for p, r in zip(batch["prompt"], batch["response"])]
    return tokenizer(texts,padding="max_length",truncation='longest_first',max_length=512,return_tensors='pt')

tokenized_dataset_sft = dataset_sft.map(tokenize_sft,batched=True,remove_columns=dataset_sft.column_names)

collator_ref_model = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

training_args_ref_model = TrainingArguments(output_dir="/kaggle/working/gpt2-medium-ref",
                                            save_strategy="steps", 
                                            save_steps=250, 
                                            per_device_train_batch_size=2,
                                            num_train_epochs=1,fp16=1,
                                            gradient_accumulation_steps=8,learning_rate=5e-5, 
                                            weight_decay=0.01, warmup_steps=100, save_total_limit=2, 
                                            report_to="none")
trainer = Trainer(
    model=reference_model,
    args=training_args_ref_model,
    train_dataset=tokenized_dataset_sft,
    tokenizer=tokenizer,
    data_collator=collator_ref_model,
)

trainer.train()
trainer.save_model("/kaggle/working/gpt2-medium-ref-final")
tokenizer.save_pretrained("/kaggle/working/gpt2-medium-ref-final")