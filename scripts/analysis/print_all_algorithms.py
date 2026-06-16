### Prints all algorithms for a given algorithms file to aid in manual analysis ##

import json
from pathlib import Path

# make sure paths are consistent for model + condition
SCRIPT_DIR = Path(__file__).parent.parent # analysis --> scripts
BASE_DIR = SCRIPT_DIR.parent 
ALGORITHMS_PATH = BASE_DIR / "algorithms" / "v3.2_algorithms.json"

with open(ALGORITHMS_PATH) as f:
    algorithms = json.load(f)

for algorithm in algorithms:
    problem_id = algorithm['problem_name']
    rule = algorithm['algorithm']
    print(problem_id)
    print(rule)
    print("------------------------------------------------")

