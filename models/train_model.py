import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from tqdm import tqdm

def train_sft(model, tokenizer, train_dataset, eval_dataset,
              batch_size=1, grad_accum=32, lr=1e-4, max_seq_len=1024,
              output_dir="./checkpoints"):
    """
    Supervised Fine-Tuning (SFT) with LoRA adapters.
    """
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding="max_length",
        max_length=max_seq_len,
        label_pad_token_id=-100
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=1,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=200,
        fp16=True,
        bf16=True,
        optim="adamw_8bit",
        save_total_limit=2,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator
    )
    trainer.train()
    trainer.save_model(output_dir)

def train_dpo(model, ref_model, preference_dataset, batch_size=4, lr=1e-5, epochs=1, device="cuda:0"):
    """
    Direct Preference Optimization (DPO) loop.
    Trains LoRA adapters using chosen/rejected preferences.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = DataLoader(preference_dataset, batch_size=batch_size, shuffle=True)
    model.train()
    ref_model.eval()

    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f"DPO Epoch {epoch+1}"):
            source = batch["source"]
            chosen = batch["chosen"]
            rejected = batch["rejected"]

            inputs = model.tokenizer(source, return_tensors="pt", padding=True, truncation=True).to(device)
            chosen_labels = model.tokenizer(chosen, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            rejected_labels = model.tokenizer(rejected, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

            chosen_loss = model(**inputs, labels=chosen_labels).loss
            rejected_loss = model(**inputs, labels=rejected_labels).loss

            with torch.no_grad():
                ref_chosen_loss = ref_model(**inputs, labels=chosen_labels).loss
                ref_rejected_loss = ref_model(**inputs, labels=rejected_labels).loss

            dpo_loss = -torch.log(torch.sigmoid((chosen_loss - rejected_loss) - 
                                                (ref_chosen_loss - ref_rejected_loss)))
            optimizer.zero_grad()
            dpo_loss.backward()
            optimizer.step()

def train_ppo(model, rollout_dataset, reward_fn, batch_size=4, lr=5e-6, epochs=1, device="cuda:0"):
    """
    PPO training loop for RLHF.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = DataLoader(rollout_dataset, batch_size=batch_size, shuffle=True)
    model.train()

    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f"PPO Epoch {epoch+1}"):
            source_texts = batch["source"]
            inputs = model.tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            outputs = model.generate(**inputs, max_length=128, do_sample=True, top_p=0.95, temperature=0.8)
            decoded_texts = [model.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

            rewards = torch.tensor(reward_fn(source_texts, decoded_texts), dtype=torch.float32, device=device)
            
            logits = model(**inputs).logits
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = torch.gather(log_probs, -1, outputs.unsqueeze(-1)).squeeze(-1)

            advantage = rewards - rewards.mean()
            loss = -(selected_log_probs * advantage).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
