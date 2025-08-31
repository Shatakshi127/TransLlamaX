# data/utils.py

"""
Data Utilities for TransLlamaX Project
--------------------------------------
This module handles:
- Loading raw datasets (FindNITAI, custom)
- Cleaning and preprocessing text
- Tokenization for LLaMA-3-8B training
- Saving processed datasets for model training/evaluation

Author: Shatakshi Saxena
"""

import os
import pandas as pd
import re
from transformers import LlamaTokenizer

class DataProcessor:
    def __init__(self, raw_data_dir='data/raw', processed_data_dir='data/processed', tokenizer_path='llama-3-8b'):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

        os.makedirs(self.processed_data_dir, exist_ok=True)

    def load_dataset(self, filename):
        """
        Load a dataset from CSV/TSV/JSON in the raw folder.
        Returns a pandas DataFrame.
        """
        path = os.path.join(self.raw_data_dir, filename)
        if filename.endswith('.csv'):
            df = pd.read_csv(path)
        elif filename.endswith('.tsv'):
            df = pd.read_csv(path, sep='\t')
        elif filename.endswith('.json'):
            df = pd.read_json(path, lines=True)
        else:
            raise ValueError("Unsupported file format: use CSV, TSV, or JSON")
        return df

    def clean_text(self, text):
        """
        Clean and normalize Hinglish text:
        - Remove extra whitespace
        - Standardize punctuation
        - Lowercase
        """
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # remove extra spaces
        text = re.sub(r'[“”]', '"', text)
        text = text.lower()
        return text

    def preprocess_dataset(self, df, src_col='hinglish', tgt_col='english'):
        """
        Apply cleaning and tokenization on source and target columns.
        Returns a tokenized dataset ready for training.
        """
        df[src_col] = df[src_col].apply(self.clean_text)
        df[tgt_col] = df[tgt_col].apply(self.clean_text)

        # Tokenization
        df['input_ids'] = df[src_col].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))
        df['labels'] = df[tgt_col].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))

        return df

    def save_processed_dataset(self, df, filename):
        """
        Save processed dataframe to CSV in the processed folder
        """
        path = os.path.join(self.processed_data_dir, filename)
        df.to_csv(path, index=False)
        print(f"Processed dataset saved to {path}")

# Example usage
if __name__ == "__main__":
    processor = DataProcessor()
    df_raw = processor.load_dataset('findnitai_train.csv')
    df_processed = processor.preprocess_dataset(df_raw)
    processor.save_processed_dataset(df_processed, 'findnitai_train_processed.csv')
