import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

SCRIPT_DIR = Path(__file__).parent.parent # eval --> scripts
BASE_DIR = SCRIPT_DIR.parent 
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "algorithms"


def extract_algorithm(text):
    match = re.search(r"FINAL_ALGORITHM:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def count_steps(algorithm):
    if algorithm is None:
        return None
    matches = re.findall(r'(?:^|\n)(\d+)\.', algorithm)
    if not matches:
        return None
    return max(int(m) for m in matches)


def query_model(prob_file):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    with open(prob_file, "r") as f:
        problem = json.load(f)

    prompt = (
    "You are an expert at solving abstract reasoning puzzles. These puzzles consist "
    "of an input grid, to which a transformation rule is applied, producing an output grid.\n"
    "Rules: Each number 0-9 represents a color. Here are some few-shot examples. \n\n"
    "Few-Shot Examples:\n"
    )

    for i, example in enumerate(problem['train']):
        prompt += f"Example {i+1} Input: {example['input']}\n"
        prompt += f"Example {i+1} Output: {example['output']}\n\n"

    prompt += (
        "Your task is to extrapolate the transformation rule from the few-shot examples and describe it as a step-by-step algorithm.\n\n"
        "Rules:\n"
        "- The algorithm must be in plain English, NOT code or pseudocode\n"
        "- Each step should be explicit and sequential\n"
        "- A human should be able to apply the steps manually to an input to get the corresponding output\n\n"
        "Output format:\n"
        "FINAL_ALGORITHM:\n"
        "1. Do this...\n"
        "2. Do this...\n"
        "...\n\n"
        "After outputting the complete algorithm, do NOT write anything more."
    )

    completion = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0
    )

    solution_text = completion.choices[0].message.content or ""
    algorithm = extract_algorithm(solution_text)
    
    return solution_text, algorithm


def run_generation(data_path, output_file):
    results = []

    files = [f for f in os.listdir(data_path) if f.endswith('.json')]
    for filename in files:
        file_path = os.path.join(data_path, filename)
        prob_name = Path(filename).stem

        print(f"Processing: {prob_name}")

        raw_solution, algorithm = query_model(file_path)
        while algorithm is None:
            print(f"Null output, rerunning...")
            raw_solution, algorithm = query_model(file_path)

        steps = count_steps(algorithm) # extract steps from algorithm text

        record = {
            "problem_name": prob_name,
            "algorithm": algorithm,
            "raw_response": raw_solution,
            "num_steps": steps,
        }

        results.append(record)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

    print(f"Algorithms saved to: {output_file}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "v3.2_algorithms.json"

    run_generation(DATA_DIR, output_path)