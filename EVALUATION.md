# Run Evaluation for BlinkCode
## Generate Answers Using the Model
### Using Supported Models
- llava-v1.6-vicuna-13b-hf
- gpt-4-vision-preview
- gpt-4-turbo-2024-04-09
- gpt-4o-2024-05-13
- qwen-vl-max
- paligemma-3b-mix-224
#### Running All Tasks
To run all tasks using a specific model:
```bash
python run_test.py --model_name gpt-4o-2024-05-13 --round_limit 3 --dataset all 
```
#### Running a Specific Task
To run a specific task with task IDs ranging from 0 to 10:
```bash
python run_test.py --model_name gpt-4o-2024-05-13 --round_limit 3 --dataset GSM8K-V --start_id 0 --end_id 10
```
#### Resuming an Unfinished Test
```bash
python run_test.py --model_name gpt-4o-2024-05-13 --round_limit 3 --dataset all --resume output/gpt-4o-2024-05-13/all_20240624_211435
```
The results for each round will be saved in the following path:
```bash
./output/{model_name}/{start_time}/output_{i}.jsonl
```
If `round_limit` is greater than 1, the script will perform refinements. For example, if `round_limit` is set to 3, the model will refine the answers twice.
### Using Your Own Model
1. Implement Your Model
- Inherit from the [BaseModel](./models/base_model.py)  class and implement your custom model class in `./models/base_model.py`.
- If your model is used through an API, you can reference the [OpenAIModel](./models/base_model.py) class.
- Otherwise, you can reference the [LlavaV16Vicuna13BHF](./models/base_model.py) class.
2. Register Your Model:
- Add the class name and model name to the [configuration file](./configs/model_mapping.yaml) to register your model.
3. Start Testing:
- Follow the instructions in the [Using Supported Models](#using-supported-models) section.
### Evaluating Model-Generated Answers
To evaluate the generated code from your multimodal large model, use the following commands:
```bash
python evaluate_MLLM_answer.py --path output/gpt-4o-2024-05-13/all_20240624_211435/output_0.jsonl
```
Results will be saved in the same directory as the specified `--path`.