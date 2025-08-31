from models.base_model import BaseLLaMA3
from explainability.lime_analysis import LimeTranslatorExplainer
from explainability.shap_analysis import SHAPTranslatorExplainer
from explainability.visualization import plot_lime, plot_shap

# Load model
base_model = BaseLLaMA3("llama-3-8b")
model = base_model.model
tokenizer = base_model.tokenizer

texts = ["Mujhe ghar jana hai", "Kya tum khush ho?"]

# LIME explanation
lime_exp = LimeTranslatorExplainer(model, tokenizer)
explanation = lime_exp.explain(texts[0])
plot_lime(explanation, texts[0])

# SHAP explanation
shap_exp = SHAPTranslatorExplainer(model, tokenizer)
shap_values = shap_exp.explain(texts)
plot_shap(shap_values, texts)
