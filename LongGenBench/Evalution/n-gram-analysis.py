import json
import re
import time
import pandas as pd
import argparse

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from collections import Counter

def calculate_ngram_repetition(text, n=20, min_count=2):
    """
    Calculate the repetition rate of n-grams in a text using NLTK.
    
    Parameters:
    - text: The input text to analyze
    - n: Size of n-grams (default: 3, for trigrams)
    - min_count: Minimum count to consider a phrase as repeated (default: 2)
    
    Returns:
    - Dictionary with n-gram repetition metrics
    """
    # Download required NLTK resources if not already available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Clean and normalize the text
    text = text.lower()
    
    # Tokenize the text using NLTK
    tokens = word_tokenize(text)
    
    if len(tokens) < n:
        return {
            "repetition_rate": 0,
            "unique_ratio": 1 if len(tokens) > 0 else 0,
            "most_common": [],
            "total_ngrams": 0,
            "unique_ngrams": 0
        }
    
    # Generate n-grams using NLTK
    n_grams = list(ngrams(tokens, n))
    
    # Count n-grams using FreqDist
    fdist = FreqDist(n_grams)
    
    # Calculate metrics
    total_ngrams = len(n_grams)
    unique_ngrams = len(fdist)
    
    # Find repeated n-grams (appearing more than once)
    repeated_ngrams = [(gram, count) for gram, count in fdist.items() if count >= min_count]
    most_common_ngrams = sorted(repeated_ngrams, key=lambda x: x[1], reverse=True)
    
    # Calculate repetition rate (higher value means more repetition)
    repetition_rate = 1 - (unique_ngrams / total_ngrams) if total_ngrams > 0 else 0
    
    # Calculate unique ratio (lower value means more repetition)
    unique_ratio = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
    
    return {
        "repetition_rate": repetition_rate,
        "unique_ratio": unique_ratio,
        "most_common": most_common_ngrams[:10],  # Top 10 most common n-grams
        "total_ngrams": total_ngrams,
        "unique_ngrams": unique_ngrams
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM with command line arguments.')
    parser.add_argument('--data', type=str, required=True, help='data_path')
    return parser.parse_args()

args = parse_args()

#csv_file_path = "/home/yuhao/THREADING-THE-NEEDLE/Evalution/results/accuracy_results.csv"
model_name = args.data.split('/')[-1].replace('.json', '')
datas = read_json(args.data)

n_gram_data = []
unique_ngram_ratio = []
repetition = []

for d in datas:
    out_len = len(d["output_blocks"])
    n_gram_res = calculate_ngram_repetition("".join(d["output_blocks"][out_len//2:]))
    n_gram_data.append(n_gram_res)
    unique_ngram_ratio.append(n_gram_res['unique_ngrams']/n_gram_res['total_ngrams'])
    repetition.append(n_gram_res['repetition_rate'])

n_gram_data.append({"unique_ngram_ratio":unique_ngram_ratio})
write_json(args.data[:-5]+"_ngram_data.json",n_gram_data)

from scipy.stats.mstats import gmean

def mean(x):
    return sum(x)/len(x)

print("Unique NGram Ratio: ", mean(unique_ngram_ratio), "Repetition: ", mean(repetition))