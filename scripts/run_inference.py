import torch
from transformers import AutoTokenizer
from models.inference import Translator
import json

# Load model & tokenizer
translator = Translator(model_path="checkpoints")
tokenizer = translator.tokenizer

# Load sentences
with open("data/test_sentences.json") as f:
    sentences = json.load(f)

# Run inference
for sent in sentences:
    translation = translator.translate(sent)
    print(f"Hinglish: {sent}")
    print(f"English: {translation}\n")
