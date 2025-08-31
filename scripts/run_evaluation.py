from evaluation import metrics, statistical_tests, quantitative_analysis
import json

# Load references & predictions
with open("data/references.json") as f:
    references = json.load(f)

with open("data/predictions.json") as f:
    predictions = json.load(f)

# Compute metrics
metric_scores = metrics.compute_all_metrics(predictions, references)
quantitative_analysis.generate_tables(metric_scores)

# Run statistical tests
statistical_tests.run_wilcoxon(metric_scores)
statistical_tests.run_friedman(metric_scores)
