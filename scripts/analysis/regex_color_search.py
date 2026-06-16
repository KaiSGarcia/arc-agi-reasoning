### Conduct regex search across all traces (used in cross-model case study on color mappings) ###

import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent # analysis --> scripts
BASE_DIR = SCRIPT_DIR.parent 
DATA_DIR = BASE_DIR / "results"

RESULTS_PATH = DATA_DIR / "v3.2_default_results.json" # change file depending on what model + condition you want to work with

ARC_COLORS = {
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "magenta",
    7: "orange",
    8: "maroon",
    9: "purple",
}
# did not include "0:black", "black" appears in general contexts ("objects against a black background") not directly related to ARC mappings

COLOR_NAMES = list(ARC_COLORS.values()) + ["grey", "light blue", "cyan", "pink", "violet"] # account for color synonyms
SEARCH = re.compile(
    r'\b(' + '|'.join(re.escape(c) for c in COLOR_NAMES) + r')\b', # use word boundaries as anchors
    re.IGNORECASE # case-insensitive
)


with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

problems_with_colors = []
problems_without_colors = []

for record in data:
    name = record.get("problem_name", "unknown")
    trace = record.get("reasoning", "")

    color_detected = []
    for line_num, line in enumerate(trace.splitlines(), start=1): # get both line and line number/index for reference
        for match in SEARCH.finditer(line):
            color = match.group(0).lower() # if color is found

            # capture context around color mention to make analysis easier
            start = max(0, match.start() - 50)
            end = min(len(line), match.end() + 50)
            context = line[start:end].strip() # extract 50-token context window
            
            color_detected.append((color, context, line_num))

    if color_detected:
        problems_with_colors.append((name, color_detected))
    else:
        problems_without_colors.append(name)


print(f"Traces with colors: {len(problems_with_colors)}")
print(f"Traces without: {len(problems_without_colors)}")

if problems_with_colors:
    print("\n Problems with colors in traces: ")
    for prob_name, matches in problems_with_colors:
        print(f"\n Problem: {prob_name}")

        # only print 5 examples to prevent excessively long list (can manually explore after)
        for color, context, line_num in matches[:3]:
            print(f" Line {line_num:>4} [{color}]: ...{context}...")
        if len(matches) > 5:
            print(f" ... and {len(matches) - 3} more match(es)")
else:
    print("\n No color names for any of this condition's traces")
