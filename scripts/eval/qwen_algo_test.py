import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import ast

# Load environment variables
load_dotenv()
api_key = os.getenv('DASHSCOPE_API_KEY')

SCRIPT_DIR = Path(__file__).parent.parent # eval --> scripts
BASE_DIR = SCRIPT_DIR.parent 
DATA_DIR = BASE_DIR / "data"
ALGO_DIR = BASE_DIR / "algorithms"
RESULTS_DIR = BASE_DIR / "results"


def clean_matrix(s):
    s = s.strip().replace("'", '"')
    try:
        obj = json.loads(s)
        if isinstance(obj, list) and obj and isinstance(obj[0], list):
            return obj
    except:
        pass
    try:
        obj = ast.literal_eval(s.replace('"', "'"))
        if isinstance(obj, list) and obj and isinstance(obj[0], list):
            return obj
    except:
        pass
    return None


def extract_final_answer(text):
    if not text:
        return None
    
    match = re.search(r'FINAL_ANSWER:\s*(\[\s*\[[\s\S]*?\]\s*\])', text)
    if match:
        result = clean_matrix(match.group(1))
        if result is not None:
            return result
        
    matches = re.findall(r'\[\s*\[[\s\S]*?\]\s*\]', text)
    if matches:
        result = clean_matrix(matches[-1])
        if result is not None:
            return result
        
    return None


def load_algorithms(path):
    with open(path, "r") as f:
        data = json.load(f)

    algo_map = {}
    for record in data:
        algo = record.get("algorithm")
        if algo:
            algo_map[record["problem_name"]] = algo

    return algo_map


def query_model(prob_file, algorithm):
    client = OpenAI(api_key=api_key, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

    with open(prob_file, 'r') as f:
        problem = json.load(f)

    test_input = problem['test'][0]['input']

    prompt = (
        "For this task, you will be given a step-by-step algorithm in advance that describes the transformation rule.\n"
        "Your task is to follow the algorithm EXACTLY, apply it to the given test input, and produce an output grid.\n\n"
        "Algorithm:\n"
        f"{algorithm}\n\n"
        "Test Input:\n"
        f"{test_input}\n\n"
        "IMPORTANT:\n"
        "- Follow the algorithm step by step\n"
        "- Do not modify the algorithm\n\n"
        "Output your final answer on the last line in this exact format:\n"
        "FINAL_ANSWER: [[row1], [row2], ...]\n\n"
        "Example:\n"
        "FINAL_ANSWER: [[0, 1, 2], [3, 4, 5]]\n\n"
        "Do not output anything after the FINAL_ANSWER line."
    )

    completion = client.chat.completions.create(
        model="qwen3.6-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        extra_body={"enable_thinking": True},
        stream=True,
    )

    solution_text = ""
    reasoning = ""
    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning += delta.reasoning_content
        if hasattr(delta, "content") and delta.content:
            solution_text += delta.content

    parsed_matrix = extract_final_answer(solution_text)
    return solution_text, reasoning, parsed_matrix


def run_evaluation(data_path, algo_path, output_file):
    results = []
    correct_count = 0
    algo_map = load_algorithms(algo_path)

    files = [f for f in os.listdir(data_path) if f.endswith('.json')]
    for filename in files:
        file_path = os.path.join(data_path, filename)
        prob_name = Path(filename).stem
        algorithm = algo_map[prob_name]

        with open(file_path, 'r') as f:
            prob_data = json.load(f)
            gold_answer = prob_data['test'][0]['output']

        print(f"Processing: {prob_name}")

        raw_solution, reasoning, parsed_matrix = query_model(file_path, algorithm)
        is_correct = (parsed_matrix == gold_answer)
        if is_correct:
            correct_count += 1
            print("correct")
        else:
            print("incorrect")

        if parsed_matrix is None:
            print("Null output")

        record = {
            "problem_name": prob_name,
            "gold_standard": gold_answer,
            "model_parsed_answer": parsed_matrix,
            "is_correct": is_correct,
            "raw_response": raw_solution,
            "reasoning": reasoning,
        }
        results.append(record)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

    print(f"Total Correct: {correct_count}/{len(results)}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    algo_path = ALGO_DIR / "qwen_algorithms.json"
    output_path = RESULTS_DIR / "qwen_algo_test_results.json"
 
    run_evaluation(DATA_DIR, algo_path, output_path)