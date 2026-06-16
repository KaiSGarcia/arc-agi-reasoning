### Extracts all traces for a given results file and saves it in a "traces" folder ###

import json
from pathlib import Path

# Change RESULTS_FILE to match the model + condition you want to export
RESULTS_FILE = "v3.2_default_results.json"

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_PATH = BASE_DIR / "results" / RESULTS_FILE
TRACES_DIR = BASE_DIR / "traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)

# Output file takes its name from the results file, e.g. v3.2_default_traces.txt
output_filename = RESULTS_FILE.replace("_results.json", "_traces.txt")
OUTPUT_PATH = TRACES_DIR / output_filename

with open(RESULTS_PATH) as f:
    results = json.load(f)

with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
    for result in results:
        problem = result["problem_name"]
        trace = result["reasoning"]
        is_correct = result["is_correct"]

        out.write(f"Problem: {problem}\n")
        out.write(f"Correct: {is_correct}\n")
        out.write(f"Trace:\n{trace}\n")
        out.write("-" * 80 + "\n\n")

print(f"\nFinished, traces saved at {OUTPUT_PATH}")
