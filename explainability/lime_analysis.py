import numpy as np
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LimeTranslatorExplainer:
    """
    LIME explanations for translation models.
    Token-level explanations for Hinglish->English predictions.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.explainer = LimeTextExplainer(class_names=["translation"])

    def predict_fn(self, texts):
        """
        Prediction function for LIME.
        Returns probabilities for each text (dummy softmax for regression-like explanation)
        """
        self.model.eval()
        probs = []
        with torch.no_grad():
            for text in texts:
                tokens = self.tokenizer(text, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(**tokens, max_new_tokens=50)
                pred_text = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
                probs.append([len(pred_text[0])/50])  # Normalized length as proxy probability
        return np.array(probs)

    def explain(self, text, num_features=10, num_samples=500):
        explanation = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=self.predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        return explanation
