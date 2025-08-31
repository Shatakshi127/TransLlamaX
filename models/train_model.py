import torch
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from base_model import BaseLLaMA3
from torch.utils.data import DataLoader

# Optional: PPO library
from trl import PPOTrainer, PPOConfig

class TransLLaMA_Trainer:
    """
    Trains LLaMA 3 8B on Hinglish->English or English->English translation
    using LoRA/QLoRA adapters.
    Supports SFT and RLHF (DPO/PPO) with adapter-only updates.
    """

    def __init__(self, base_model="llama-3-8b", device="cuda:0"):
        self.device = device
        self.base = BaseLLaMA3(base_model, device)
        self.model = self.base.model
        self.tokenizer = self.base.tokenizer

    def add_lora_adapters(self, r=16, alpha=32, dropout=0.05):
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            lora_dropout=dropout,
            bias="none"
        )
        self.model = get_peft_model(self.model, config)
        self.model.to(self.device)

    def train_sft(self, train_dataset, eval_dataset, batch_size=1, gradient_accumulation=32,
                  lr=1e-4, max_seq_len=1024, output_dir="./checkpoints"):
        collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            padding="max_length",
            max_length=max_seq_len,
            label_pad_token_id=-100
        )
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
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
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator
        )
        trainer.train()
        trainer.save_model(output_dir)

    def apply_rlhf_dpo(self, preference_dataset, batch_size=8, lr=5e-6, epochs=1):
        """
        Implements Direct Preference Optimization (DPO).
        preference_dataset: Dataset with {"query": str, "chosen": str, "rejected": str}
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        dataloader = DataLoader(preference_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for batch in dataloader:
                query = batch["query"]
                chosen = batch["chosen"]
                rejected = batch["rejected"]

                # Tokenize
                enc_q = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(self.device)
                enc_chosen = self.tokenizer(chosen, return_tensors="pt", padding=True, truncation=True).to(self.device)
                enc_rejected = self.tokenizer(rejected, return_tensors="pt", padding=True, truncation=True).to(self.device)

                # Log-probs
                logp_chosen = self.model(**enc_chosen, labels=enc_chosen["input_ids"]).loss
                logp_rejected = self.model(**enc_rejected, labels=enc_rejected["input_ids"]).loss

                # DPO loss: maximize preference
                loss = -(logp_chosen - logp_rejected).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print("DPO RLHF stage completed.")

    def apply_rlhf_ppo(self, reward_fn, rollout_dataset, batch_size=4, lr=5e-6, epochs=1):
        """
        Implements PPO RLHF using the TRL library.
        reward_fn: callable that takes generated text and returns reward scores
        rollout_dataset: Dataset with {"query": str}
        """
        # PPO configuration
        config = PPOConfig(
            model_name="llama-3-8b",
            learning_rate=lr,
            batch_size=batch_size,
            log_with=None
        )

        ppo_trainer = PPOTrainer(
            config=config,
            model=self.model,
            tokenizer=self.tokenizer
        )

        for epoch in range(epochs):
            for data in rollout_dataset:
                query = data["query"]
                enc = self.tokenizer(query, return_tensors="pt").to(self.device)
                response = self.model.generate(**enc, max_length=128)
                response_text = self.tokenizer.batch_decode(response, skip_special_tokens=True)

                # Compute reward
                rewards = reward_fn(response_text)

                # Update using PPO
                ppo_trainer.step([query], response_text, rewards)

        print("PPO RLHF stage completed.")
