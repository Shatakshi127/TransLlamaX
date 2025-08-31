import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

def plot_lime(explanation, text):
    """
    Plots LIME token importances for a single text.
    """
    token_weights = dict(explanation.as_list())
    tokens, weights = zip(*token_weights.items())
    plt.figure(figsize=(10,5))
    sns.barplot(x=list(tokens), y=list(weights), palette="viridis")
    plt.title("LIME Token-level Importance")
    plt.xticks(rotation=45)
    plt.ylabel("Importance")
    plt.xlabel("Tokens")
    plt.show()

def plot_shap(shap_values, feature_names):
    """
    Plots SHAP values for token-level importance.
    """
    shap_values = np.array(shap_values)
    plt.figure(figsize=(12,6))
    sns.heatmap(shap_values, xticklabels=feature_names, yticklabels=False, cmap="coolwarm", center=0)
    plt.title("SHAP Token-level Importance")
    plt.xlabel("Tokens")
    plt.ylabel("Samples")
    plt.show()
