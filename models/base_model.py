import sys
import os
sys.path.append(os.getcwd())
from utils import encode_image, check_image_path
import dashscope
from dashscope import MultiModalConversation
import os
import random
import numpy as np
import requests
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
)
import torch
from PIL import Image
import requests
from typing import Optional, Tuple
from abc import ABC, abstractmethod
import importlib
import time
API_SECRET_KEY = os.getenv("openai_key")
BASE_URL = os.getenv("base_url", "https://api.openai.com/v1")

seed = 42
temperature = 0



def set_random_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, model_name: str):
        """Load model here.
        """
        pass

    @abstractmethod
    def forward(self, query: str, image_path1: Optional[str] = None, image_path2: Optional[str] = None) -> Tuple[bool, str]:
        """Every model should implement this method.

        There are three possible inputs:
        - query and no image
        - query and one image
        - query and two images

        If your model does not support two images, you can ignore the image_path2.

        Args:
            query (str): The query string.
            image_path1 (Optional[str]): Path to the first image (if any).
            image_path2 (Optional[str]): Path to the second image (if any).

        Returns:
            Tuple[bool, str]: A tuple containing a success flag and a message.
                - If the request to the model is successful:
                    - The first element is True.
                    - The second element is the newly generated text.
                - If the request to the model fails:
                    - The first element is False.
                    - The second element is an error message.
        """
        pass


class QwenModel(BaseModel):
    def __init__(self, model_name: str = "qwen-vl-max"):
        qwen_key = os.getenv("qwen_key")
        if qwen_key is None:
            raise ValueError("Please set the environment variable qwen_key")
        dashscope.api_key = qwen_key
        """Initialize the model with the specified name."""
        self.model_name = model_name

    def forward(self, query: str, image_path1: Optional[str] = None, image_path2: Optional[str] = None) -> Tuple[bool, str]:
        """Process the input data."""
        if check_image_path(image_path1, image_path2) == False:
            raise ValueError(
                f"image_path1: {image_path1}, image_path2: {image_path2}")
        if image_path1 is None and image_path1 is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"text": query},
                    ],
                }
            ]
        elif image_path2 is None and image_path1 is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"text": query},
                        {"text": "\image:\n"},
                        {"image": image_path1},
                    ],
                }
            ]
        elif image_path1 is not None and image_path2 is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"text": query},
                        {"text": "\nfirst image:\n"},
                        {"image": image_path1},
                        {"text": "\nsecond image:\n"},
                        {"image": image_path2},
                    ],
                }
            ]
        else:
            raise ValueError(
                f"image_path1: {image_path1}, image_path2: {image_path2}")
        try:
            response = MultiModalConversation.call(
                model=self.model_name, messages=messages)
            while (response["status_code"] == 429):
                time.sleep(20)
                # print(response)
                response = MultiModalConversation.call(
                    model=self.model_name, messages=messages)
            if response["status_code"] == 200:
                content = response["output"]["choices"][0]["message"]["content"][0]
                if "box" in content:
                    return True, content["box"]
                elif "text" in content:
                    return True, content["text"]
                else:
                    raise ValueError('No "text", no "box."')
            else:
                return False, str(response)
        except Exception as e:
            return False, f"Unexpected error:{e}"


class OpenAIModel(BaseModel):
    """This class can be used with any model that communicates using the OpenAI format.
    If you want to use other models, please modify self.base_url and self.api_key accordingly.
    """

    def __init__(self, model_name: str = "gpt-4o-2024-05-13"):
        """Initialize the model with the specified name."""
        self.model_name = model_name
        self.max_tokens = 4096
        self.temperature = temperature
        self.seed = seed
        self.base_url = BASE_URL
        self.key = API_SECRET_KEY

    def forward(self, query: str, image_path1: Optional[str] = None, image_path2: Optional[str] = None) -> Tuple[bool, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}",
        }
        if check_image_path(image_path1, image_path2) == False:
            raise ValueError(
                f"image_path1: {image_path1}, image_path2: {image_path2}")
        if image_path1 == None and image_path2 == None:
            content = [
                {"type": "text", "text": query},
            ]
        elif image_path2 == None and image_path1 != None:
            content = [
                {"type": "text", "text": query},
                {"type": "text", "text": "\image:\n"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image_path1)}",
                        "detail": "high",
                    },
                },
            ]
        elif image_path1 != "" and image_path2 != "":
            content = [
                {"type": "text", "text": query},
                {"type": "text", "text": "\nfirst image:\n"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image_path1)}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": "\nsecond image:\n"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image_path2)}",
                        "detail": "high",
                    },
                },
            ]
        else:
            raise ValueError(
                "when image_path2 provided, image_path1 must be provided")
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload)
            while (response.status_code == 429 and "Rate limit" in response.json()["error"]["message"]):
                time.sleep(20)
                response = requests.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                )
            if response.status_code == 413:
                return False, f"request  too long, {response}"
            response = response.json()
            if (
                "error" in response
                or response == {}
                or response["choices"][0]["message"]["content"] == ""
                or response["choices"][0]["message"]["content"] == {}
            ):
                return False, str(response)
            else:
                return True, response["choices"][0]["message"]["content"]
        except Exception as e:
            return False, f"Unexpected error:{e}"


class LlavaV16Vicuna13BHF(BaseModel):
    """This model suppert two images.
    """

    def __init__(self, model_name: str = "llava-v1.6-vicuna-13b-hf"):
        self.model_name = f"llava-hf/{model_name}"
        set_random_seed()
        self.token = None
        self.max_new_tokens = 4096
        self.processor = LlavaNextProcessor.from_pretrained(
            self.model_name, use_auth_token=self.token
        )
        self.temperature = temperature
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, query: str, image_path1: Optional[str] = None, image_path2: Optional[str] = None) -> Tuple[bool, str]:
        if check_image_path(image_path1, image_path2) == False:
            raise ValueError(
                f"image_path1: {image_path1}, image_path2: {image_path2}")
        images = []
        if image_path1 != "":
            images.append(Image.open(image_path1))
        if image_path2 != "":
            images.append(Image.open(image_path2))
        sigle_image_prompt_template = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:[INST]<image>\nquery [/INST]\nASSISTANT:"
        no_image_prompt_template = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:query \nASSISTANT:"
        two_image_prompt_template = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:[INST] <image>\nThis is the first image. [/INST]  [INST]\n <image>\nThis is the second image. [/INST]\nquery\nASSISTANT:"
        if image_path1 == "" and image_path2 == "":
            images = None
            prompt_template = [
                no_image_prompt_template.replace("query", query)]
        elif image_path2 == "":
            prompt_template = [
                sigle_image_prompt_template.replace("query", query)]
        elif image_path1 != "" and image_path2 != "":
            prompt_template = [
                two_image_prompt_template.replace("query", query)]
        else:
            raise ValueError(
                f"image_path1: {image_path1}, image_path2: {image_path2}")
        inputs = self.processor(
            text=prompt_template,
            images=images,
            padding=True,
            return_tensors="pt",
            max_length=20000,
            truncation=True,
        ).to(self.device)
        output = self.model.generate(
            **inputs, temperature=self.temperature, max_new_tokens=self.max_new_tokens
        )
        decode_result = self.processor.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        index = decode_result.find("ASSISTANT:")
        if index == -1:
            raise ValueError("index == -1")
        result = decode_result[index + len("ASSISTANT:"):].strip()
        if result == "" or result == " " * len(result):
            return True, ""
        return True, result


class PaliGemmaModel(BaseModel):
    """This model does support one images.
    """

    def __init__(self, model_name: str = "paligemma-3b-mix-224"):
        self.model_id = f"google/{model_name}"
        self.max_new_tokens = 4096
        self.temperature = temperature
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            revision="bfloat16",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.device = torch.device(self.device)
        self.model.to(self.device)

    def forward(self, query: str, image_path1: Optional[str] = None, image_path2: Optional[str] = None) -> Tuple[bool, str]:
        if check_image_path(image_path1, image_path2) == False:
            raise ValueError(
                f"image_path1: {image_path1}, image_path2: {image_path2}")
        if image_path1 != None and image_path2 != None:
            raise ValueError("This model does not support two images.")
        if image_path1 == None:
            raise ValueError("image_path1 is None.")
        image = Image.open(image_path1)
        model_inputs = self.processor(
            text=query, images=image, return_tensors="pt"
        ).to(self.model.device)
        input_len = model_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=self.temperature,
            )
            generation = generation[0][input_len:]
            decoded = self.processor.decode(
                generation, skip_special_tokens=True)
        return True, decoded


def get_model_class(class_name: str) -> BaseModel:
    module_name = 'models.base_model'
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Cannot find class {class_name} in module {module_name}: {e}")


if __name__ == "__main__":
    # model = LlavaV16Vicuna13BHF()
    # model = QwenModel()
    model = OpenAIModel(model_name="gpt-4o-2024-05-13")
    print(model.forward("1 + 1 = ?"))
