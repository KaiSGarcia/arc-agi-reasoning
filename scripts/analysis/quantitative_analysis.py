### Calculates all quantitative metrics for a given condition (except rumination/entropy) ###

import json
import statistics
import pandas as pd
from pathlib import Path

# make sure paths are consistent for model + condition
SCRIPT_DIR = Path(__file__).parent.parent # analysis --> scripts
BASE_DIR = SCRIPT_DIR.parent 
RESULTS_PATH = BASE_DIR / "results" / "v3.2_default_results.json" 
ALGORITHMS_PATH = BASE_DIR / "algorithms" / "v3.2_algorithms.json"
ANNOTATIONS_PATH = BASE_DIR / "annotations.csv"


with open(RESULTS_PATH, "r") as f: # just need results for basic accuracy/error stats (will merge other info further down)
    data = json.load(f)

################################################################################################
# OVERALL ACCURACY AND ERROR METRICS
print("\n" + "=" * 50)
print("OVERALL ACCURACY + ERRORS")
print("=" * 50)

total = len(data)
correct_results = [d for d in data if d['is_correct']]
incorrect_results = [d for d in data if not d['is_correct']]
accuracy = (len(correct_results) / total) * 100
print(f"Accuracy: {accuracy:.2f}%")

parsing_errors = []
logic_errors = []
for wrong in incorrect_results:
    if wrong["model_parsed_answer"] is None:
        parsing_errors.append(wrong["problem_name"])
    else:
        logic_errors.append(wrong["problem_name"])

print(f"Logic Errors: {len(logic_errors)}")
print(f"Parsing Errors: {len(parsing_errors)}")
print(f"Logic Error IDs: {', '.join(logic_errors) if logic_errors else 'None'}")
print(f"Parsing Error IDs: {', '.join(parsing_errors) if parsing_errors else 'None'}")

################################################################################################
# OVERALL TRACE LENGTHS (CORRECT/INCORRECT)
print("\n" + "=" * 50)
print("OVERALL TRACE LENGTHS")
print("=" * 50)

correct_lens = [len(d.get("reasoning", "")) for d in correct_results]
incorrect_lens = [len(d.get("reasoning", "")) for d in incorrect_results]

avg_correct = statistics.mean(correct_lens) if correct_lens else 0
avg_incorrect = statistics.mean(incorrect_lens) if incorrect_lens else 0
med_correct = statistics.median(correct_lens) if correct_lens else 0
med_incorrect = statistics.median(incorrect_lens) if incorrect_lens else 0
std_correct = statistics.stdev(correct_lens) if len(correct_lens) > 1 else 0
std_incorrect = statistics.stdev(incorrect_lens) if len(incorrect_lens) > 1 else 0

print(f"\n{'':30s} {'Correct':>10} {'Incorrect':>10}")
print(f"{'Mean (chars)':30s} {avg_correct:>10.0f} {avg_incorrect:>10.0f}")
print(f"{'Median (chars)':30s} {med_correct:>10.0f} {med_incorrect:>10.0f}")
print(f"{'Std Dev (chars)':30s} {std_correct:>10.0f} {std_incorrect:>10.0f}")

################################################################################################
# Merge with annotations for remaining metrics
annotations = pd.read_csv(ANNOTATIONS_PATH, encoding="latin-1")
baseline_df = pd.DataFrame(data)

baseline_df["problem_name"] = baseline_df["problem_name"].astype(str).str.strip()
annotations["task_id"] = annotations["task_id"].astype(str).str.strip() # use ID as shared key

merged = baseline_df.merge(
    annotations,
    left_on="problem_name",
    right_on="task_id",
    how="left"
).drop(columns=["task_id"]) # remove ID column (identical to problem_name)

# Merge with algorithms for remaining metrics
with open(ALGORITHMS_PATH) as f:
    algo_data = json.load(f)

algo_df = pd.DataFrame(algo_data)[["problem_name", "algorithm", "num_steps"]]
algo_df["problem_name"] = algo_df["problem_name"].astype(str).str.strip()

merged = merged.merge(algo_df, on="problem_name", how="left") # also merge on unique problem ID

################################################################################################
# ACCURACY BY COMPLEXITY VARIABLES
print("\n" + "=" * 50)
print("ACCURACY BY COMPLEXITY VARIABLES")
print("=" * 50)

def comp_var_analysis(df, col):
    stats = df.groupby(col)["is_correct"].agg(["mean", "count"]) # accuracy for each variable being tested
    stats["mean"] = (stats["mean"] * 100).round(2)
    stats.columns = ["Accuracy (%)", "N"]
    return stats

# Run for each complexity variable
for var in ["grid_change", "train_examples", "color_relevant"]:
    print(f"\nAnalysis for: {var}")
    print(comp_var_analysis(merged, var))

################################################################################################
# ACCURACY BY ALGORITHM STEP COUNT 
print("\n" + "=" * 50)
print("ACCURACY BY ALGORITHM STEP COUNT")
print("=" * 50)

step_analysis = merged.groupby("num_steps")["is_correct"].agg(["mean", "count"]) # accuracy for each possible step count
step_analysis["mean"] = (step_analysis["mean"] * 100).round(2)
step_analysis.columns = ["Accuracy (%)", "Total Problems"]
print(step_analysis.sort_index())

################################################################################################
# ACCURACY BY REASONING CATEGORY
print("\n" + "=" * 50)
print("ACCURACY BY REASONING CATEGORY")
print("=" * 50)

category_analysis = merged.groupby("category")["is_correct"].agg(["mean", "count"]) # accuracy for each category
category_analysis["mean"] = (category_analysis["mean"] * 100).round(2)
category_analysis.columns = ["Accuracy (%)", "N"]
print(category_analysis.sort_values(by="Accuracy (%)", ascending=False))

################################################################################################
# TRACE LENGTH BY CATEGORY 
print("\n" + "=" * 50)
print("TRACE LENGTH BY CATEGORY")
print("=" * 50)

merged["trace_length"] = merged["reasoning"].str.len() # add column for length of reasoning traces
length_analysis = merged.groupby("category")["trace_length"].agg(["mean", "median", "count"]) # trace length for each category

length_analysis.columns = ["Avg Length (Chars)", "Median Length", "N"]
length_analysis = length_analysis.round(0).astype(int)
print(length_analysis.sort_values(by="Avg Length (Chars)", ascending=False))

################################################################################################
# GRID CHANGE: CATEGORY BREAKDOWN
print("\n" + "=" * 50)
print("GRID CHANGE RELEVANCE: CATEGORY ANALYSIS")
print("=" * 50)

for is_rel in [True, False]:
    print(f"\n>>> Where grid_change == {is_rel}")
    subset = merged[merged["grid_change"] == is_rel] # filter for where grid change is either T/F (depends on input)
    
    stats  = subset.groupby("category")["is_correct"].agg(["mean", "count"]) # aggregate accuracy for each category
    stats["mean"] = (stats["mean"] * 100).round(2)
    stats.columns = ["Accuracy (%)", "N"]
    print(stats.sort_values(by="Accuracy (%)", ascending=False))

################################################################################################
# COLOR RELEVANCE: CATEGORY BREAKDOWN
print("\n" + "=" * 50)
print("COLOR RELEVANCE: CATEGORY ANALYSIS")
print("=" * 50)

for is_rel in [True, False]:
    print(f"\n>>> Where color_relevant == {is_rel}")
    subset = merged[merged["color_relevant"] == is_rel] # filter for where color relevance is either T/F (depends on input)

    stats  = subset.groupby("category")["is_correct"].agg(["mean", "count"]) # aggregate accuracy for each category
    stats["mean"] = (stats["mean"] * 100).round(2)
    stats.columns = ["Accuracy (%)", "N"]
    print(stats.sort_values(by="Accuracy (%)", ascending=False))