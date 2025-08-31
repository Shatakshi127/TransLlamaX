import yaml
import logging
from models.train_model import TransLLaMA_Trainer
from datasets import load_dataset

# Load configs
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Training")

# Load dataset
train_dataset = load_dataset("json", data_files=config["dataset"]["train_path"])["train"]
eval_dataset = load_dataset("json", data_files=config["dataset"]["eval_path"])["train"]

# Initialize trainer
trainer = TransLLaMA_Trainer(device="cuda:0")
trainer.add_lora_adapters(
    r=config["lora"]["r"], 
    alpha=config["lora"]["alpha"], 
    dropout=config["lora"]["dropout"]
)

# Start training
trainer.train_sft(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    batch_size=config["training"]["batch_size"],
    gradient_accumulation=config["training"]["gradient_accumulation"],
    lr=config["training"]["learning_rate"],
    max_seq_len=config["training"]["max_seq_len"],
    output_dir=config["training"]["output_dir"]
)
logger.info("Training completed and model saved.")
