import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

def plot_metric_bar(df, metric="BLEU"):
    """
    Plots bar chart comparing models on a single metric.
    """
    plt.figure(figsize=(8,6))
    sns.barplot(x="Model", y=metric, data=df, palette="coolwarm")
    plt.title(f"Comparison of Models: {metric}")
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

def plot_heatmap(data, labels=None, title="Heatmap"):
    """
    Plots heatmap (useful for posthoc p-values)
    """
    plt.figure(figsize=(8,6))
    sns.heatmap(data, annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels, cmap="YlGnBu")
    plt.title(title)
    plt.tight_layout()
    plt.show()
