from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from sacrebleu.metrics import BLEU, GLEU
from bert_score import score as bert_score
import torch
import evaluate

# Initialize metrics
rouge = Rouge()
bleu_metric = BLEU()
gleu_metric = GLEU()
bleurt = evaluate.load("bleurt")  # Requires internet for first download

def compute_metrics(references, predictions):
    """
    Compute multiple translation metrics.
    """
    # BLEU using sacrebleu
    bleu_score = bleu_metric.corpus_score(predictions, [references]).score

    # METEOR using NLTK
    meteor_scores = [meteor_score([ref], pred) for ref, pred in zip(references, predictions)]
    meteor_avg = sum(meteor_scores)/len(meteor_scores)

    # ROUGE-L using rouge
    rouge_scores = rouge.get_scores(predictions, references, avg=True)
    rouge_l = rouge_scores['rouge-l']['f']

    # GLEU
    gleu_score = gleu_metric.corpus_score(predictions, [references]).score

    # BERTScore
    P, R, F1 = bert_score(predictions, references, lang="en", rescale_with_baseline=True)
    bert_f1 = F1.mean().item()

    # BLEURT
    bleurt_scores = bleurt.compute(predictions=predictions, references=references)
    bleurt_avg = sum(bleurt_scores["scores"])/len(bleurt_scores["scores"])

    return {
        "BLEU": bleu_score,
        "METEOR": meteor_avg,
        "ROUGE-L": rouge_l,
        "GLEU": gleu_score,
        "BERTScore-F1": bert_f1,
        "BLEURT": bleurt_avg
    }
