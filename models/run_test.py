import os
import sys
import importlib
sys.path.append(os.getcwd())
from utils import load_jsonl, unsafe_execute_code, get_code, convert_png_to_jpeg, create_tempdir, time_limit, convert_html_to_image,process_figure_code, unsafe_execute_matplotlib_code, run_pdflatex, get_python_code, get_html_code, check_and_process_images
import cairosvg
from models.base_model import get_model_class
import yaml
import json
from time import sleep
from tqdm import tqdm 
import argparse
import fitz
import inspect

request_model = None
with open('configs/model_mapping.yaml', 'r') as f:
    config = yaml.safe_load(f)
two_image_model_mapping = [key for key, val in config['two_image_models'].items()]  
one_image_models_mapping = [key for key, val in config['one_image_models'].items()]  
 
#load refine prompt
with open(f"{os.getcwd()}/prompt/refine_prompt.json", 'r', encoding='utf-8') as file:
    all_prompt = json.load(file)

def processor_request(prompt, img1, img2 = None,model="gpt-4-vision-preview"):
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
        img1 = convert_png_to_jpeg(img1, img1.replace('images', 'jpeg_images').replace('.png', '.jpeg'))
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
            result, answer = request_model(prompt, image_path1=img1, image_path2=img2)
    return result, answer

def processor_refine_request(example, model="gpt-4-vision-preview"):
    """this function is used to refine the answer of the MLLM model
    Parameters:
    - example (dict): The example is not refine result
    - model (str): The identifier for the model to use, which determines the processing technique and capabilities.
    Returns:
    - A tuple containing the result status (boolean) and the processed data or error message (str).
    """
    input = example["prompt"]
    pre_image_path = os.path.join(os.getcwd(),  example['task_id'] + '.png')
    return_data = []
    result = [True, example['MLLM_answer']]
    type = example['type']
    for i in range(2):
        with open(f"logs/refine_ouput/{model}.txt", "a") as f:
            f.write(f"the {i} refine")
        if i == 1 and len(return_data) != 1:
            raise ValueError(" i == 1 but len(return_data) != 1")
        pre_result = result
        isTerminatedEarly = 0 # Determine whether it needs to end early
        if result[0] != True:
            return result[0], result[1]
        code = ""
        if type in ['HumanEval-V', 'MBPP-V', 'GSM8K-V', 'MATH-V']:
            code = get_python_code(result[1]) 
            complete_code = code + '\n' + example['test'] + "\n" + f"check({example['entry_point']})"
            exec_result = unsafe_execute_code(complete_code)
            if code != "" and (exec_result == "True" or exec_result == "Your code has no syntax errors, but it doesn't pass all the test cases"): 
                query = all_prompt[type]["code_no_error_single_image"].replace("{previous code}", code)
                isTerminatedEarly = 1
            elif code == "":
                query = input + "\nYou must write your code according to the format requirements."
            else:
                query = all_prompt[type]["code_error_single_image"].replace("{previous code}", code).replace("{error message}", exec_result)
            if type == "MBPP-V" and code != "": #MBPP refine requires a function name.
                function_name = input.split("```python\ndef ")[1].split(':\n```')[0]
                query = query.replace("{function_signature}", function_name)
            result = list(processor_request(query, pre_image_path, model=model))
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
                        figure_code = process_figure_code(code, temp_image_path)
                        exec_result = unsafe_execute_matplotlib_code(figure_code) 
                    except Exception as e:
                        print(f"Exception: {e}")
                        return [False, f"Unexpected error:{e}"]
                elif type == 'SVG':
                    code = get_code(result[1], '<svg', '</svg>', is_contain_start_end=True)
                    svg_path = os.path.join(now_path, '0.svg')
                    with open(svg_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                    try:
                        with time_limit(4):
                            cairosvg.svg2png(url=svg_path, write_to=temp_image_path, background_color="white")
                        exec_result = "True"
                    except Exception as e:
                        exec_result = (
                            f"The SVG code is not properly formatted and is not displaying correctly."
                            f"CairoSVG encountered the following error when converting the SVG code to an image: {e}"
                        )
                elif type == 'TikZ':
                    code = get_code(result[1], start='\\documentclass', end='\\end{document}', is_contain_start_end=True)
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
                                raise Exception("The TikZ code generates a multi-page PDF. Please provide TikZ code that generates a single-page PDF.")
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
                        query = all_prompt[type]["code_error_single_image"].replace("{previous code}", code).replace("{error message}", exec_result) 
                    else: #exec_result == "True" and code != "":
                        query = all_prompt[type]["code_no_error_single_image"].replace("{previous code}", code)
                        isTerminatedEarly = 1
                    result = list(processor_request(query, pre_image_path, model=model))
                else:
                    # code != "" and os.path.exists(temp_image_path) and os.path.getsize(temp_image_path) != 0
                    if exec_result != "True" and type == "tikz":
                        query = all_prompt[type]["code_error_two_image"].replace("{previous code}", code).replace("{error message}", exec_result)
                    else:
                        query = all_prompt[type]["code_no_error_two_image"].replace("{previous code}", code)
                    isTerminatedEarly = 1
                    check_and_process_images(temp_image_path)
                    result = list(processor_request(query, pre_image_path, img2=temp_image_path, model=model))
        elif type == 'VP':
            ocr_line = "####################OCR result####################"
            ocr_result = ocr_line + "\n" + input.split(ocr_line)[1] + "\n" + ocr_line
            vp_query = get_code(input, start="####################OCR result####################\n\nQuery:", end="If you think you can answer the question directly")
            code = get_python_code(result[1])
            if code != "":
                exec_result = run_vp_code(code, pre_image_path, 200)
            if code == "":
                query = input + "\nYou must write your code according to the format requirements."
            elif "Code execution error exception:" in exec_result:
                exec_result = exec_result.replace("Code execution error exception:", "")
                query = all_prompt[type]["code_error_single_image"].replace("{Image Patch}", all_prompt['Class_ImagePatch']).replace("{Ocr result}", ocr_result).replace("{orignal query}", vp_query).replace("{previous code}", code).replace("{error message}", exec_result)
            else:
                query = all_prompt[type]["code_no_error_single_image"].replace("{Image Patch}", all_prompt['Class_ImagePatch']).replace("{Ocr result}", ocr_result).replace("{orignal query}", vp_query).replace("{previous code}", code)
                isTerminatedEarly = 1
            result = list(processor_request(query, pre_image_path, model=model))
        else:
            raise Exception("type not supported")
        #log
        with open(f"logs/refine_ouput/{model}.txt", "a") as f:
            f.write('-'*30 + "\n")
            f.write(query + "\n")
            f.write("MLLM response +++++++++++++++: MLLM response\n")
            f.write(f"{result[1]}\n")

        if isTerminatedEarly == 1 and result[0] == True: #The MLLM request succeeds, and no error message is reported, and you need to determine whether to end the request prematurely
            split_result = result[1].split('\n')
            is_false = split_result[0]
            if len(split_result) > 1 and is_false == "":#the first line is empty
                is_false = split_result[1]
            if "false" in is_false.lower():
                #Decide how many times you want to end it early. append twice for the first time and once for the second time
                if i == 0:
                    return_data.append(pre_result[1])
                    return_data.append(pre_result[1])
                else:
                    return_data.append(pre_result[1])
                return True, return_data

                # return pre_result[0], pre_result[1]
            else:
                return_data.append(result[1])
        elif result[0] == True:
            return_data.append(result[1])
        else:
            #result[0] == False, need to return the error message
            return result[0], result[1]
    if result[0] == True:
        return True, return_data
    else:
        raise ValueError("It's impossible to get here")
                 
def test_not_refine(path, output, error_path, reset=False, model= "gpt-4-vision-preview"):
    """Process test data to generate answers using a specified model and log results.

    Args:
        path (str): Path to the input JSONL file containing test data.
        output (str): Path to the output JSONL file where results will be saved.
        error_path (str): Path to the error log file where errors will be saved.
        reset (bool, optional): If True, existing output and error files will be reset. Defaults to False.
        model (str, optional): The model to be used for processing the data. Defaults to "gpt-4-vision-preview".

    Returns:
        None
    """
    if reset or not os.path.exists(output):
        with open(output, 'w') as f:
            pass
    if reset or not os.path.exists(error_path):
        with open(error_path, 'w') as f:
            pass
    if not os.path.exists(path):
        raise ValueError("path must existed")
    temp_data = load_jsonl(output)  # only process the data that has not been in the output
    already_exist_data = {dist_data['task_id']:1 for dist_data in temp_data}
    original_data =  load_jsonl(path)
    need_test_data = []
    for data in original_data:
        if data['task_id'] not in already_exist_data:
            need_test_data.append(data)

    data = need_test_data
    for example in tqdm(data):
        img_path = os.path.join(os.getcwd(),  example['task_id'] + '.png')
        result, answer = processor_request(example["prompt"], img_path, model=model)
        if result == False:
            #log error information
            with open(error_path, 'a') as f:
                print("error: " , answer)
                f.write(json.dumps({"task_id":example['task_id'], "error":answer}, ensure_ascii=False) + '\n')
            continue
        with open(output, 'a') as f:
            example['MLLM_answer'] = answer
            example['model'] = model
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
            print('write success')

def test_refine(not_refine_path, output_path_1, output_path_2, error_path, reset=False, model= "gpt-4-vision-preview"):
    """Process data that needs to be refine, generate answers using the specified model, and record the results.

    Args:
        not_refine_path (str): Path to the input JSONL file containing unrefined test data.
        output_path_1 (str): Path to the output JSONL file where the first refined results will be saved.
        output_path_2 (str): Path to the output JSONL file where the second refined results will be saved.
        error_path (str): Path to the error log file where errors will be saved.
        reset (bool, optional): If True, existing output and error files will be reset. Defaults to False.
        model (str, optional): The model to be used for processing the data. Defaults to "gpt-4-vision-preview".
        
    Returns:
        None
    """
    if reset or not os.path.exists(output_path_1):
        with open(output_path_1, 'w') as f:
            pass
    if reset or not os.path.exists(output_path_2):
        with open(output_path_2, 'w') as f:
            pass
    if reset or not os.path.exists(error_path):
        with open(error_path, 'w') as f:
            pass
    os.makedirs("logs/refine_ouput", exist_ok=True)
    if not os.path.exists(not_refine_path):
        raise ValueError("not_refine_path must existed")
    not_refine_data = load_jsonl(not_refine_path)
    need_refine_data = []
    already_exist_data = {dist_data['task_id']:1 for dist_data in load_jsonl(output_path_2)}
    for data in not_refine_data:
        if data['task_id'] not in already_exist_data:
            need_refine_data.append(data)
    for example in tqdm(need_refine_data):
        #log
        with open(f"logs/refine_ouput/{model}.txt", "a") as f:
            f.write(f"task_id:{example['task_id']}\n")
        # load vipergpt model
        if 'VP' == example['type'] and  "viper_exec_code" not in sys.modules:
            print("Loading vipergpt model")
            module = importlib.import_module("call_vp_function")
            globals().update(vars(module))
            print("Load success!")
        result, answer = processor_refine_request(example, model = model)
        example['MLLM_answer'] = ""
        if result == False:
            with open(error_path, 'a') as f:
                f.write(json.dumps({"task_id":example['task_id'], "error":answer}, ensure_ascii=False) + '\n')
            continue
        if len(answer) != 2:
            raise ValueError(f"len(answer) != 2, {answer}")
        with open(output_path_1, 'a') as f:
            example['MLLM_answer'] = answer[0]
            example['model'] = model
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
        with open(output_path_2, 'a') as f:
            example['MLLM_answer'] = answer[1]
            example['model'] = model
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
        print('write success')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a model test.")
    parser.add_argument("--reset", action="store_true", default=False, help="Reset the output file and error file.")
    parser.add_argument("--model_name", type=str, required=True, help="Specify the model to use.")
    parser.add_argument("--refine", action="store_true", default=False, help="")
    args = parser.parse_args()
    original_data_path = 'data/MLLMCodeTest.jsonl'
    model = args.model_name
    refine = args.refine
    if refine:
        not_refine_output_path = f'output/{model}/not_refine/output.jsonl'
        output_path_1 = f'output/{model}/refine/output_1.jsonl'
        output_path_2 = f'output/{model}/refine/output_2.jsonl'
        refine_error_path = f'output/{model}/refine/output_error.jsonl'
        if not os.path.exists(f'output/{model}/refine'):
            os.makedirs(f'output/{model}/refine')
    else:
        not_refine_output_path = f'output/{model}/not_refine/output.jsonl'
        not_refine_error_path = f'output/{model}//not_refine/output_error.jsonl'
        if not os.path.exists(f'output/{model}/not_refine'):
            os.makedirs(f'output/{model}/not_refine')
    
    reset = args.reset
    if model not in two_image_model_mapping and model not in one_image_models_mapping:
        raise ValueError(f"model {model} not supported")
    print(f"load model: {model}")
    if model in two_image_model_mapping:
        test_model = get_model_class(config['two_image_models'][model])(model_name=model)
    else:
        test_model = get_model_class(config['one_image_models'][model])(model_name=model)
    request_model = test_model.forward
    print(f"load success")
    if refine:
        test_refine(not_refine_output_path ,output_path_1, output_path_2, refine_error_path,reset=reset, model=model)
    else:
        test_not_refine(original_data_path, not_refine_output_path, not_refine_error_path,reset=reset,model=model)

