import pandas as pd
from .metrics import compute_metrics

def metrics_table(references, predictions_dict):
    """
    Creates a table comparing multiple models.
    predictions_dict: {model_name: [list of predictions]}
    """
    table = []
    for model, preds in predictions_dict.items():
        scores = compute_metrics(references, preds)
        scores["Model"] = model
        table.append(scores)
    df = pd.DataFrame(table)
    df = df[["Model","BLEU","METEOR","ROUGE-L","GLEU","BERTScore-F1","BLEURT"]]
    return df

def aggregate_scores(df, metric="BLEU"):
    """
    Aggregates scores for statistical analysis.
    """
    return df.pivot_table(index="Model", values=metric)
