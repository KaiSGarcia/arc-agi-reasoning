### Calculates rumination and lexical entropy metrics for a given condition ###

import json
import re
import math
import pandas as pd
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent # analysis --> scripts
BASE_DIR = SCRIPT_DIR.parent 
DATA_DIR = BASE_DIR / "results"


def basic_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def ngram_repetition_rate(text, n = 5): 
    words = basic_tokenize(text)
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)] # 5-token sequences
    total = len(ngrams)
    unique = len(set(ngrams))
    return 1.0 - unique / total 

def normalized_lexical_entropy(text):
    words = basic_tokenize(text)
    n = len(words)
    if n < 2:
        return 0.0
    counts = Counter(words)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p) # using formula given in Thoughtology paper
    return entropy / math.log2(n)

def compute_trace_metrics(text):
    return {
        "rumination_rate": ngram_repetition_rate(text),
        "lexical_entropy": normalized_lexical_entropy(text),
    }


def get_results(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    results = []
    for record in data:
        trace = record.get("reasoning", "")
        metrics = compute_trace_metrics(trace) # calculate metrics for each problem
        
        # add this data as 2 new columns to the original results data
        combined_entry = {
            "problem_name": record.get("problem_name"),
            "is_correct": record.get("is_correct"),
            **metrics
        }
        results.append(combined_entry)

    df = pd.DataFrame(results) # convert to Pandas df for easier indexing
    avg_rumination = df['rumination_rate'].mean()
    avg_entropy = df['lexical_entropy'].mean()

    print(f"Average 5-gram Rumination Rate: {avg_rumination:.4f}")
    print(f"Average Normalized Lexical Entropy: {avg_entropy:.4f}")
    print("\n")

    # correctness matrix
    print(df.groupby('is_correct')[['rumination_rate', 'lexical_entropy']].mean())
    return df
    

if __name__ == "__main__":
    RESULTS_PATH = DATA_DIR / "v3.2_default_results.json" # change file depending on what model + condition you want to work with
    results_df = get_results(RESULTS_PATH)