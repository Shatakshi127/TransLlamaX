import torch
from peft import LoraConfig, get_peft_model, TaskType

def add_lora_adapters(model, r=16, alpha=32, dropout=0.05, target_modules=None, device="cuda:0"):
    """
    Adds LoRA adapters to the model for parameter-efficient fine-tuning.
    """
    if target_modules is None:
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none"
    )
    model = get_peft_model(model, config)
    model.to(device)
    return model

def freeze_model_parameters(model):
    """
    Freezes all parameters except LoRA adapters.
    """
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
