import shap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class SHAPTranslatorExplainer:
    """
    SHAP explanations for translation models.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict_fn(self, texts):
        self.model.eval()
        outputs_list = []
        with torch.no_grad():
            for text in texts:
                tokens = self.tokenizer(text, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(**tokens, max_new_tokens=50)
                pred_text = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
                outputs_list.append([len(pred_text[0])/50])
        return np.array(outputs_list)

    def explain(self, texts, nsamples=100):
        """
        SHAP explanations using KernelExplainer.
        """
        explainer = shap.KernelExplainer(self.predict_fn, np.array(texts))
        shap_values = explainer.shap_values(texts, nsamples=nsamples)
        return shap_values
