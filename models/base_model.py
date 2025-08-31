import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class BaseLLaMA3:
    """
    Loads a base LLaMA-3-8B model for translation tasks.
    """

    def __init__(self, model_name="llama-3-8b", device="cuda:0"):
        self.device = device
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).to(device)

    def generate(self, input_texts, max_length=128, num_beams=4):
        """
        Generate text outputs from input texts.
        """
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=0.6,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2
        )
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
