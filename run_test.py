from tqdm import tqdm
import sys
import os
from utils import load_jsonl, unsafe_execute_code, get_code, convert_png_to_jpeg, create_tempdir, time_limit, convert_html_to_image, process_figure_code, unsafe_execute_matplotlib_code, run_pdflatex, get_python_code, get_html_code, check_and_process_images
import fitz
import argparse
import json
import yaml
from models.base_model import get_model_class
import cairosvg
import datetime
import importlib

request_model = None
with open('configs/model_mapping.yaml', 'r') as f:
    config = yaml.safe_load(f)
two_image_model_mapping = [key for key,
                           val in config['two_image_models'].items()]
one_image_models_mapping = [key for key,
                            val in config['one_image_models'].items()]

# load refine prompt
with open(f"prompt/refine_prompt.json", 'r', encoding='utf-8') as file:
    all_prompt = json.load(file)
with open(f"prompt/base_prompt.json", 'r', encoding='utf-8') as file:
    base_prompt = json.load(file)

def processor_request(prompt, img1, img2=None, model="gpt-4-vision-preview"):
    """Processes a model-specific request using provided images and a text prompt. This function interfaces with a specified model to handle different types of image processing tasks based on the model's capabilities and the given inputs.

    Parameters:
    - prompt (str): The text prompt that guides the model on how to interpret and process the images.
    - img1 (str): The file path or URL to the first image.
    - img2 (str, optional): The file path or URL to the second image, if applicable.
    - model (str): The identifier for the model to use, which determines the processing technique and capabilities.

    Returns:
    - A tuple containing the result status (boolean) and the processed data or error message (str).
    """

    if "claude" in model:
        img1 = convert_png_to_jpeg(img1, img1.replace(
            'images', 'jpeg_images').replace('.png', '.jpeg'))
        if img2 != None:
            img2 = convert_png_to_jpeg(img2, img2.replace('.png', '.jpeg'))
    if img1 == "" or img1 == None:
        raise ValueError("img1 is empty")
    if model in one_image_models_mapping:
        if img2 == None:
            return request_model(prompt, image_path1=img1)
        else:
            raise ValueError("this model only support one image")
    else:
        if img2 == None:
            result, answer = request_model(prompt, image_path1=img1)
        else:
            result, answer = request_model(
                prompt, image_path1=img1, image_path2=img2)
    return result, answer


def mutiple_round_request(example, round_limit, model="gpt-4-vision-preview"):
    """Process a request for a given example using multiple rounds of refinement to generate answers.
    """
    input = base_prompt[example['type']]
    if example['type'] == 'VP':
        input = input.replace("{query}", example["query"]).replace("{ocr result}", example["ocr_result"])
    elif example['type'] == 'MBPP-V':
        input = input.replace("{function name}", example["function_name"])
    pre_image_path = os.path.join(os.getcwd(),  "data", example['path'])
    type = example['type']
    return_data = []
    result = processor_request(input, pre_image_path, model=model)
    if result[0] == False or round_limit == 1:
        return result[0], result[1]
    return_data.append(result[1])
    for i in range(round_limit - 1):
        if len(return_data) != i + 1:
            raise ValueError("len(return_data) != i + 1")
        pre_result = result
        isTerminatedEarly = 0  # Determine whether it needs to end early
        code = ""
        if type in ['HumanEval-V', 'MBPP-V', 'GSM8K-V', 'MATH-V']:
            code = get_python_code(result[1])
            if type == "MBPP-V":
                complete_code = code + '\n' + \
                    example['evaluation_function'] + "\n" + f"check({example['function_name'].split('(')[0]})"
            else:
                complete_code = code + '\n' + \
                    example['evaluation_function'] + "\n" + f"check({example['function_name']})"
            exec_result = unsafe_execute_code(complete_code)
            if code != "" and (exec_result == "True" or exec_result == "Your code has no syntax errors, but it doesn't pass all the test cases"):
                query = all_prompt[type]["code_no_error_single_image"].replace(
                    "{previous code}", code)
                isTerminatedEarly = 1
            elif code == "":
                query = input + "\nYou must write your code according to the format requirements."
            else:
                query = all_prompt[type]["code_error_single_image"].replace(
                    "{previous code}", code).replace("{error message}", exec_result)
            # MBPP refine requires a function name.
            if type == "MBPP-V" and code != "":
                query = query.replace("{function_signature}", example["function_name"])
            result = list(processor_request(
                query, pre_image_path, model=model))
        elif type in ['Webpage', 'Matplotlib', 'SVG', 'TikZ']:
            exec_result = ""
            with create_tempdir():
                now_path = os.getcwd()
                temp_image_path = os.path.join(now_path, "0.png")
                if type == 'Webpage':
                    code = get_html_code(result[1])
                    html_path = os.path.join(now_path, "0.html")
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                    try:
                        with time_limit(10):
                            convert_html_to_image(html_path, temp_image_path)
                            if code == "":
                                raise ValueError("the code is empty!")
                            exec_result = "True"
                    except Exception as e:
                        exec_result = (
                            f"The HTML code is not properly formatted and is not displaying correctly."
                            f"Encountered the following error when converting the HTML code to PDF: {e}"
                        )
                elif type == 'Matplotlib':
                    try:
                        code = get_python_code(result[1])
                        figure_code = process_figure_code(
                            code, temp_image_path)
                        exec_result = unsafe_execute_matplotlib_code(
                            figure_code)
                    except Exception as e:
                        print(f"Exception: {e}")
                        return [False, f"Unexpected error:{e}"]
                elif type == 'SVG':
                    code = get_code(result[1], '<svg',
                                    '</svg>', is_contain_start_end=True)
                    svg_path = os.path.join(now_path, '0.svg')
                    with open(svg_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                    try:
                        with time_limit(4):
                            cairosvg.svg2png(
                                url=svg_path, write_to=temp_image_path, background_color="white")
                        exec_result = "True"
                    except Exception as e:
                        exec_result = (
                            f"The SVG code is not properly formatted and is not displaying correctly."
                            f"CairoSVG encountered the following error when converting the SVG code to an image: {e}"
                        )
                elif type == 'TikZ':
                    code = get_code(result[1], start='\\documentclass',
                                    end='\\end{document}', is_contain_start_end=True)
                    tikz_path = os.path.join(now_path, '0.tex')
                    pdf_path = os.path.join(now_path, '0.pdf')
                    with open(tikz_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                    try:
                        with time_limit(12):
                            result_tikz = run_pdflatex(tikz_path, now_path)
                            if os.path.exists(pdf_path) == False:
                                raise Exception(result_tikz)
                            pdf_document = fitz.open(pdf_path)
                            for page_num in range(len(pdf_document)):
                                page = pdf_document.load_page(page_num)
                                pix = page.get_pixmap()
                                pix.save(temp_image_path)
                            if len(pdf_document) > 1:
                                raise Exception(
                                    "The TikZ code generates a multi-page PDF. Please provide TikZ code that generates a single-page PDF.")
                            if result_tikz != "True":
                                raise Exception(result_tikz)
                            exec_result = "True"
                    except Exception as e:
                        exec_result = (
                            f"The TikZ code is not properly formatted and is not displaying correctly."
                            f"pdflatex encountered the following error when converting the TikZ to a PDF: {e}"
                        )
                if model in one_image_models_mapping or os.path.exists(temp_image_path) == False or code == "" or os.path.getsize(temp_image_path) == 0:
                    if code == "":
                        query = input + "\nYou must write your code according to the format requirements."
                    elif exec_result != "True":
                        query = all_prompt[type]["code_error_single_image"].replace(
                            "{previous code}", code).replace("{error message}", exec_result)
                    else:  # exec_result == "True" and code != "":
                        query = all_prompt[type]["code_no_error_single_image"].replace(
                            "{previous code}", code)
                        isTerminatedEarly = 1
                    result = list(processor_request(
                        query, pre_image_path, model=model))
                else:
                    # code != "" and os.path.exists(temp_image_path) and os.path.getsize(temp_image_path) != 0
                    if exec_result != "True" and type == "tikz":
                        query = all_prompt[type]["code_error_two_image"].replace(
                            "{previous code}", code).replace("{error message}", exec_result)
                    else:
                        query = all_prompt[type]["code_no_error_two_image"].replace(
                            "{previous code}", code)
                    isTerminatedEarly = 1
                    check_and_process_images(temp_image_path)
                    result = list(processor_request(
                        query, pre_image_path, img2=temp_image_path, model=model))
        elif type == 'VP':
            code = get_python_code(result[1])
            if code != "":
                exec_result = run_vp_code(code, pre_image_path, 200)
            if code == "":
                query = input + "\nYou must write your code according to the format requirements."
            elif "Code execution error exception:" in exec_result:
                exec_result = exec_result.replace(
                    "Code execution error exception:", "")
                query = all_prompt[type]["code_error_single_image"].replace("{Image Patch}", all_prompt['Class_ImagePatch']).replace(
                    "{Ocr result}", example["ocr_result"]).replace("{orignal query}", example["query"]).replace("{previous code}", code).replace("{error message}", exec_result)
            else:
                query = all_prompt[type]["code_no_error_single_image"].replace("{Image Patch}", all_prompt['Class_ImagePatch']).replace(
                    "{Ocr result}", example["ocr_result"]).replace("{orignal query}", example["query"]).replace("{previous code}", code)
                isTerminatedEarly = 1
            result = list(processor_request(
                query, pre_image_path, model=model))
        else:
            raise Exception("type not supported")

        # The MLLM request succeeds, and no error message is reported, and you need to determine whether to end the request prematurely
        if result[0] == False:
            return result[0], result[1]
        elif isTerminatedEarly == 1 and result[0] == True:
            split_result = result[1].split('\n')
            is_false = split_result[0]
            if len(split_result) > 1 and is_false == "":  # the first line is empty
                is_false = split_result[1]
            if "false" in is_false.lower():
                # Decide how many times you want to end it early. append twice for the first time and once for the second time
                need_append_num = round_limit - i - 1
                for _ in range(need_append_num):
                    return_data.append(pre_result[1])
                return True, return_data
            else:
                return_data.append(result[1])
        else:
            return_data.append(result[1])
    return True, return_data


def test(data_path, output_paths, error_output_path, round_limit, start_id, end_id, dataset, model="gpt-4-vision-preview"):
    """Process test data to generate answers using a specified model and log results.

    Args:
        data_path (str): Path to the input JSONL file containing test data.
        output_paths (list): List of paths to the output JSONL files where results will be saved.
        round_limit (int): Number of rounds to run the test.
        start_id (int): Start ID for the test.
        end_id (int): End ID for the test.
        model (str, optional): The model to be used for processing the data. Defaults to "gpt-4-vision-preview".

    Returns:
        None
    """
    if round_limit != len(output_paths):
        raise ValueError("round_limit != len(output_paths)")
    if not os.path.exists(data_path):
        raise ValueError("data_path must existed")
    data = load_jsonl(data_path)
    need_test_data = []
    already_exist_data = {dist_data['type'] + f"_{dist_data['id']}": 1 for dist_data in load_jsonl(output_paths[0])}
    for d in data:
        unique_id = d['type'] + f"_{d['id']}"
        if d['type'] in dataset and unique_id not in already_exist_data and start_id <= int(d['id']) <= end_id:
            need_test_data.append(d)
    for example in tqdm(need_test_data):
        # load vipergpt model
        if 'VP' == example['type'] and "viper_exec_code" not in sys.modules:
            print("Loading vipergpt model")
            module = importlib.import_module("call_vp_function")
            globals().update(vars(module))
            print("Load success!")
        result, answer = mutiple_round_request(example, round_limit, model=model)
        example['MLLM_answer'] = ""
        if result == False:
            with open(error_output_path, 'a') as f:
                f.write(json.dumps(
                    {"type_id": f"{example['type']}_{example['id']}", "error": answer}, ensure_ascii=False) + '\n')
            continue
        if len(answer) != round_limit:
            raise ValueError(f"len(answer) != round_limit, {answer}")
        for i in range(round_limit):
            with open(output_paths[i], 'a') as f:
                example['MLLM_answer'] = answer[i]
                example['model'] = model
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        print('write success')


if __name__ == '__main__':
    dataset_choices = ['all', 'VP', 'Webpage', 'Matplotlib', 'SVG', 'TikZ', 'HumanEval-V', 'MBPP-V', 'GSM8K-V', 'MATH-V']
    parser = argparse.ArgumentParser(description="Run a model test.")
    parser.add_argument("--model_name", type=str,
                        required=True, help="Specify the model to use.")
    parser.add_argument("--round_limit", type=int,
                        default=3, help="Specify the number of rounds the model should run. If set to 1, the model will only run one round without refinement.")
    parser.add_argument("--resume", type=str, default=None, 
                    help="Resume the test. If provided, specify the path to resume from.")
    parser.add_argument('--dataset', type=str, choices=dataset_choices, required=True,
                    help="Name of the dataset to use. Must be one of the following: 'ALL', 'VP', 'Webpage', 'Matplotlib', 'SVG', 'TikZ', 'HumanEval-V', 'MBPP-V', 'GSM8K-V', 'MATH-V'.")
    parser.add_argument("--start_id", type=int, default=0, 
                        help="Start ID for the test (default: 0)")
    parser.add_argument("--end_id", type=int, default=999, 
                        help="End ID for the test (default: 999)")
    args = parser.parse_args()
    original_data_path = f"data/dataset.jsonl"
    model = args.model_name
    if args.dataset == "all":
        dataset = ["VP", "Webpage", "Matplotlib", "SVG", "TikZ", "HumanEval-V", "MBPP-V", "GSM8K-V", "MATH-V"]
    else:
        dataset = [args.dataset]
    resume = args.resume
    round_limit = args.round_limit
    now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_paths = []
    start_id = args.start_id
    end_id = args.end_id
    if len(dataset) != 1 and (start_id != 0 or end_id != 999):
        raise ValueError("If dataset is set all, start_id and end_id must be 0 and 999 respectively, because the id is connected to the dataset")
    if resume == None:
        if len(dataset) != 1:
            dataset_name = "all"
        else:
            dataset_name = dataset[0]
        os.makedirs(f"output/{model}/{dataset_name}_{now_time}", exist_ok=True)
        for i in range(round_limit):
            output_paths.append(f"output/{model}/{dataset_name}_{now_time}/output_{i}.jsonl")
            with open(output_paths[i], 'w') as f:
                pass
        error_ouput_path = f"output/{model}/{dataset_name}_{now_time}/error_output.jsonl"
        with open(error_ouput_path, 'w') as f:
            pass
    else:
        for i in range(round_limit):
            output_paths.append(f"{resume}/output_{i}.jsonl")
            if not os.path.exists(output_paths[-1]):
                raise ValueError(f"{output_paths[-1]} not existed")
        error_ouput_path = f"{resume}/error_output.jsonl"
        if not os.path.exists(error_ouput_path):
            raise ValueError(f"{error_ouput_path} not existed")
    if model not in two_image_model_mapping and model not in one_image_models_mapping:
        raise ValueError(f"model {model} not supported")
    print(f"load model: {model}")
    if model in two_image_model_mapping:
        test_model = get_model_class(
            config['two_image_models'][model])(model_name=model)
    else:
        test_model = get_model_class(
            config['one_image_models'][model])(model_name=model)
    request_model = test_model.forward
    print(f"load success")
    test(original_data_path, output_paths, error_ouput_path, round_limit, start_id, end_id, dataset, model=model)