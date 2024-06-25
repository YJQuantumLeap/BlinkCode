import importlib
import json
from utils import get_code, get_html_code, get_python_code, stream_jsonl
from eval.execution import check_correctness
import tqdm
from collections import defaultdict
import sys
import os
import argparse
sys.path.append(os.getcwd())


def write_result(result, sample, out_file, out_file_error):
    """Write the evaluation result to the appropriate output file.

    Parameters:
    result (dict): The result of the evaluation, including pass status and error types.
    sample (dict): The sample task details including task_id and type.
    out_file (str): The file path to write the successful results.
    out_file_error (str): The file path to write the error results.

    Returns:
    None
    """
    task_type = sample["type"]
    data = {}
    data['id'] = sample["id"]
    data['path'] = sample["path"]
    data['type'] = task_type
    if task_type in ['HumanEval-V', 'MBPP-V', 'GSM8K-V', 'MATH-V', 'VP']:
        data["passed"] = result["passed"]
        data['error_type'] = result["error_type"]
    else:
        data["error_type"] = result["error_type"]
        data["score"] = result["score"]
        data['Score_for_each_item'] = result['Score_for_each_item']
        data['MLLM_evaluate'] = result['MLLM_evaluate']
    data['original'] = sample
    if 'evaluation model error' in data['error_type']:
        with open(out_file_error, 'ab') as fp:
            fp.write((json.dumps(data) + "\n").encode('utf-8'))
        return
    with open(out_file, 'ab') as fp:
        fp.write((json.dumps(data) + "\n").encode('utf-8'))


def evaluate_MLLM_answer_correctness(
    sample_file: str,
    reset=False,
):
    """Evaluate the correctness of MLLM (Multi-Modal Language Model) answers from a sample file.

    Parameters:
    sample_file (str): The path to the sample file containing MLLM answers.
    reset (bool): Whether to reset the evaluation results. Default is False.

    Returns:
    None
    """
    out_file = sample_file.split('.')[0] + "_results.jsonl"
    out_file_error = os.path.join(os.getcwd(), sample_file.split('.')[
                                  0] + "_results_error.jsonl")

    if reset or not os.path.exists(out_file_error):
        with open(out_file_error, 'w') as f:
            pass
    if reset or not os.path.exists(out_file):
        with open(out_file, 'w') as f:
            pass

    already_in_output_file = {}
    for out_sample in stream_jsonl(out_file):
        already_in_output_file[out_sample['type'] + "_" + str(out_sample['id'])] = True
    need_to_evaluate_data = []
    for sample in stream_jsonl(sample_file):
        if f"{sample['type']}_{sample['id']}" not in already_in_output_file:
            need_to_evaluate_data.append(sample)
    results = []
    print("Reading samples...")
    viper_exec_code = None
    for sample in tqdm.tqdm(need_to_evaluate_data):
        if sample['type'] == 'VP' and "call_vp_function" not in sys.modules:
            print("Loading vipergpt model")
            module = importlib.import_module("call_vp_function")
            viper_exec_code = module.run_vp_code
            globals().update(vars(module))
            print("Load success!")

        type = sample['type']
        if type in ['HumanEval-V', 'MBPP-V', 'GSM8K-V', 'MATH-V', 'VP']:
            code = get_python_code(sample['MLLM_answer'])
            if type == 'VP':
                args = (sample, code, 200, viper_exec_code)
            else:
                args = (sample, code, 10)
        elif type in ['Webpage', 'Matplotlib', 'SVG', 'TikZ']:
            if type == 'Webpage':
                code = get_html_code(sample['MLLM_answer'])
            elif type == 'Matplotlib':
                code = get_python_code(sample['MLLM_answer'])
            elif type == 'SVG':
                code = get_code(sample['MLLM_answer'], '<svg',
                                '</svg>', is_contain_start_end=True)
            elif type == 'TikZ':
                code = get_code(sample['MLLM_answer'], start='\\documentclass',
                                end='\\end{document}', is_contain_start_end=True)
            args = (sample, code, 300)
        results.append(check_correctness(*args))
        write_result(results[-1], sample, out_file, out_file_error)

    print("Running test.py suites...")

    correct = defaultdict(list)
    score = defaultdict(list)
    for result in stream_jsonl(out_file):
        task_type = result['type']
        if task_type in ['HumanEval-V', 'MBPP-V', 'GSM8K-V', 'MATH-V', 'VP']:
            correct[task_type].append(result['passed'])
        else:
            score[task_type].append(result["score"])

    evaluate_result = defaultdict(list)
    keys = list(correct.keys()) + list(score.keys())
    for task_type in keys:
        if task_type in ['HumanEval-V', 'MBPP-V', 'GSM8K-V', 'MATH-V', 'VP']:
            evaluate_result[task_type] = sum(
                correct[task_type]) / len(correct[task_type])
            print(f'{task_type} correctness: {evaluate_result[task_type]}')
        else:
            evaluate_result[task_type] = sum(
                score[task_type]) / len(score[task_type])
            print(f'{task_type} score: {evaluate_result[task_type]}')
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluate the output.")
    parser.add_argument("--reset", action="store_true",
                        default=False, help="Reset the evaluate result.")
    parser.add_argument("--path", type=str, required=True,
                        help="Specify which output to evaluate.")
    args = parser.parse_args()
    evaluate_MLLM_answer_correctness(args.path, reset=args.reset)
