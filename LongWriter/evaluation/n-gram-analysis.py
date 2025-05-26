import json
import re
import time
import pandas as pd
import argparse

def read_json(file_path):
    read_data = []
    with open(file_path, 'r') as file:
        read_data = [json.loads(l) for l in file.readlines()]
    return read_data


def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# def calculate_ngram_repetition(text, n=3):
#     """
#     Calculate the repetition rate of n-grams in a text.
    
#     Parameters:
#     - text: The input text to analyze
#     - n: Size of n-grams (default: 3, for trigrams)
    
#     Returns:
#     - Dictionary with n-gram repetition metrics
#     """
#     # Clean and normalize the text
#     # You can extend this preprocessing based on your needs
#     text = text.lower()
    
#     # Generate n-grams
#     words = text.split()
#     total_words = len(words)
    
#     if total_words < n:
#         return {
#             "repetition_rate": 0,
#             "unique_ratio": 1 if total_words > 0 else 0,
#             "most_common": [],
#             "total_ngrams": 0,
#             "unique_ngrams": 0
#         }
    
#     ngrams = []
#     for i in range(len(words) - n + 1):
#         ngram = tuple(words[i:i+n])
#         ngrams.append(ngram)
    
#     # Count the occurrences of each n-gram
#     ngram_count = {}
#     for ngram in ngrams:
#         if ngram in ngram_count:
#             ngram_count[ngram] += 1
#         else:
#             ngram_count[ngram] = 1
    
#     # Calculate repetition metrics
#     total_ngrams = len(ngrams)
#     unique_ngrams = len(ngram_count)
    
#     # Find the most common n-grams (those that appear more than once)
#     most_common = [(ngram, count) for ngram, count in ngram_count.items() if count > 1]
#     most_common.sort(key=lambda x: x[1], reverse=True)
    
#     # Calculate repetition rate (higher value means more repetition)
#     # This is 1 - (unique n-grams / total n-grams)
#     repetition_rate = 1 - (unique_ngrams / total_ngrams) if total_ngrams > 0 else 0
    
#     # Calculate the ratio of unique n-grams (lower value means more repetition)
#     unique_ratio = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
    
#     return {
#         "repetition_rate": repetition_rate,
#         "unique_ratio": unique_ratio,
#         "most_common": most_common[:10],  # Return top 10 most common n-grams
#         "total_ngrams": total_ngrams,
#         "unique_ngrams": unique_ngrams
#     }

import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from collections import Counter

def calculate_ngram_repetition(text, n=10, min_count=2):
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
    
    # import pdb; pdb.set_trace()
    return {
        "repetition_rate": repetition_rate,
        "unique_ratio": unique_ratio,
        "most_common": most_common_ngrams[:10],  # Top 10 most common n-grams
        "total_ngrams": total_ngrams,
        "unique_ngrams": unique_ngrams
    }

# def compare_texts_repetition(texts, text_names=None, n=3, min_count=2):
#     """
#     Compare multiple texts to determine which has more repetitive phrases.
    
#     Parameters:
#     - texts: List of text strings to compare
#     - text_names: Optional list of names for the texts (will use indices if not provided)
#     - n: Size of n-grams to analyze (default: 3)
#     - min_count: Minimum count to consider a phrase as repeated (default: 2)
    
#     Returns:
#     - Dictionary with comparison results
#     """
#     if text_names is None:
#         text_names = [f"Text {i+1}" for i in range(len(texts))]
    
#     if len(texts) != len(text_names):
#         raise ValueError("Number of texts and text names should match")
    
#     results = {}
#     for text, name in zip(texts, text_names):
#         results[name] = calculate_ngram_repetition(text, n, min_count)
    
#     # Determine which text has more repetition
#     repetition_rates = [(name, data["repetition_rate"]) for name, data in results.items()]
#     repetition_rates.sort(key=lambda x: x[1], reverse=True)
    
#     most_repetitive = repetition_rates[0][0] if repetition_rates else None
    
#     # Calculate additional comparative metrics
#     avg_repetition_rate = sum(rate for _, rate in repetition_rates) / len(repetition_rates) if repetition_rates else 0
    
#     return {
#         "detailed_results": results,
#         "most_repetitive": most_repetitive,
#         "repetition_ranking": repetition_rates,
#         "average_repetition_rate": avg_repetition_rate,
#         "ngram_size": n
# }


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

resp_len = []

for i in range(len(datas)):
    # import pdb; pdb.set_trace()
    n_gram_res = calculate_ngram_repetition(datas[i]["response"])
    n_gram_res['response_length'] = datas[i]['response_length']
    if n_gram_res['total_ngrams']==0 or datas[i]['response_length']<5000: continue
    resp_len.append(datas[i]['response_length'])
    n_gram_data.append(n_gram_res)
    unique_ngram_ratio.append(n_gram_res['unique_ngrams']/n_gram_res['total_ngrams'])
    repetition.append(n_gram_res['repetition_rate'])

n_gram_data.append({"unique_ngram_ratio":unique_ngram_ratio})
write_json(args.data[:-6]+"_ngram_data.json",n_gram_data)

from scipy.stats.mstats import gmean

def mean(x):
    return sum(x)/len(x)

print("Unique NGram Ratio: ", mean(unique_ngram_ratio), "Repetition: ", mean(repetition))

from matplotlib import pyplot as plt
plt.scatter(x=resp_len, y=repetition)
plt.xlabel("Response length")
plt.ylabel("Repetitions")
plt.savefig(args.data[:-6]+"_Scatter.png")

