import json
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import re
import time

import dashscope
from dashscope import MultiModalConversation

import math
import uuid
from PIL import Image
from math import ceil, floor

def extract_images_from_messages(messages):
    files = []
    for msg in messages:
        if isinstance(msg, dict) and isinstance(msg["content"], list):
            for item in msg["content"]:
                if item.get("image"):
                    files.append(item["image"].replace("file://",""))
        elif not isinstance(msg, dict) and isinstance(msg.content, list):
            for item in msg.content:
                if item.image and item.image not in files:
                    files.append(item.image.replace("file://",""))
    return files

def image_zoom_in_tool(label, bbox_2d, images, img_idx):
    work_dir = os.path.dirname(os.path.abspath(__file__))
    image = Image.open(images[img_idx])
    def maybe_resize_bbox(left, top, right, bottom, img_width, img_height):
        """Resize bbox to ensure it's valid"""
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)

        height = bottom - top
        width = right - left
        if height < 32 or width < 32:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 32 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)

            # Ensure the resized bbox is within image bounds
            new_left = max(0, new_left)
            new_top = max(0, new_top)
            new_right = min(img_width, new_right)
            new_bottom = min(img_height, new_bottom)

            new_height = new_bottom - new_top
            new_width = new_right - new_left

            if new_height > 32 and new_width > 32:
                return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]
    def round_by_factor(number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    def ceil_by_factor(number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    def floor_by_factor(number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    def smart_resize(height: int,
                     width: int,
                     factor: int = 32,
                     min_pixels: int = 56 * 56,
                     max_pixels: int = 12845056) -> tuple[int, int]:
        """Smart resize image dimensions based on factor and pixel constraints"""
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)
        return h_bar, w_bar
    img_width, img_height = image.size
    rel_x1, rel_y1, rel_x2, rel_y2 = bbox_2d
    abs_x1, abs_y1, abs_x2, abs_y2 = rel_x1 / 1000. * img_width, rel_y1 / 1000. * img_height, rel_x2 / 1000. * img_width, rel_y2 / 1000. * img_height

    validated_bbox = maybe_resize_bbox(abs_x1, abs_y1, abs_x2, abs_y2, img_width, img_height)

    left, top, right, bottom = validated_bbox

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    # Resize according to smart_resize logic
    new_w, new_h = smart_resize((right - left), (bottom - top), factor=32, min_pixels=256 * 32 * 32)
    cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)

    output_path = os.path.abspath(os.path.join(work_dir, f'{uuid.uuid4()}.png'))
    cropped_image.save(output_path)
    return [{"label": label, "image": f"file://{output_path}", "img_idx": len(images)}]

try:
    from core import LLMServerABC
    from logger import logger
except ImportError:
    import sys, os

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.join(CURRENT_DIR, "..")
    sys.path.append(PARENT_DIR)
    from core import LLMServerABC
    from logger import logger

class APIDashBoardVLMServer(LLMServerABC):
    """Use Request API to generate responses based on input messages.
    """
    def start_server(self) -> None:
        logger.info("APIDashBoardVLMServer: no local service to start.")
        return
    
    def __init__(self, 
                 api_url: str = "",
                 api_key: str = "DASHBOARD_API_KEY",
                 model_name: str = "qwen3-vl-plus",
                 max_workers: int = 3,
                 max_retries: int = 3,
                 ):
        # Get API key from environment variable or config
        dashscope.base_http_api_url = api_url
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_retries = max_retries

        # config api_key in os.environ global, since safty issue.
        self.api_key = os.environ.get(api_key)
        if api_key is None:
            error_msg = f"Lack of `{api_key}` in environment variables. Please set `{api_key}` as your api-key to {api_url} before using APILLMServing_request."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
    def cleanup(self):
        # Cleanup resources if needed
        logger.info("Cleaning up resources in APIDashBoardVLMServer")
        return
    
    def generate(
        self, 
        user_input: list[dict,None], 
        system_prompt: str = "",
        json_schema: dict = None,
    ) -> list[str]:
        responses = [None] * len(user_input)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._api_chat_id_retry,
                    payload = [
                            {
                                "role": "user", 'content': [
                                    {'image': f"file://{itm.get('image_path','')}"},
                                    {'text': itm.get('question','')}
                                ]
                            }
                        ],
                    model = self.model_name,
                    id = idx,
                    question=itm["question"],
                    image_path=itm["image_path"],
                    tools=itm.get("tools", None),
                    # tool_choice={"type": "function", "function": {"name": itm.get("tools", [0])['function']['name']}} if itm.get("tools", None) else None,
                ) for idx, itm in enumerate(user_input)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating......"):
                    response = future.result() # (id, response)
                    responses[response[0]] = response[1]
        return responses
        
    def _api_chat_id_retry(self, id, payload, model, is_embedding : bool = False, json_schema: dict = None, **kwargs):
        for i in range(self.max_retries):
            id, response = self._api_chat_with_id(id, payload, model, is_embedding, json_schema, **kwargs)
            if response is not None:
                return id, response
            time.sleep(2**i)
        return id, None    
    
    def _api_chat_raw(self, messages, model=None, is_embedding: bool = False, json_schema: dict = None, **kwargs):
        try:
            response = dashscope.MultiModalConversation.call(
                api_key=self.api_key,
                model=self.model_name if model is None else model,
                messages=messages,
                vl_high_resolution_images=True,
                tools=kwargs.get("tools", None)
            )
            if response.status_code == 200:
                logger.debug(response)
                return response
            else:
                raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"API request error: {e}")
            return None

    def _api_chat_with_id(self, id, messages, model=None, is_embedding: bool = False, json_schema: dict = None, **kwargs):
        try:
            response = self._api_chat_raw(messages=messages, model=model, is_embedding=is_embedding, json_schema=json_schema, **kwargs)
            if response.status_code == 200:
                if "tool_calls" not in response.output.choices[0].message or not response.output.choices[0].message["tool_calls"]:
                    return id, response.output.choices[0].message.content[0]["text"]
                else:
                    while "tool_calls" in response.output.choices[0].message and response.output.choices[0].message["tool_calls"]:
                        messages.append(response.output.choices[0].message)
                        tool_call = response.output.choices[0].message["tool_calls"][0]
                        func_name = tool_call["function"]["name"]
                        arguments = json.loads(tool_call["function"]["arguments"])
                        tool_call_id = tool_call.get("id")  # 获取 tool_call_id
                        all_images = extract_images_from_messages(messages)
                        tool_result = image_zoom_in_tool(label=arguments["label"],bbox_2d=arguments["bbox_2d"],img_idx=int(arguments["img_idx"]),images=all_images)
                        tool_message = {
                            "role": "tool",
                            "content": tool_result,
                            "tool_call_id": tool_call_id
                        }
                        messages.append(tool_message)
                        all_images = extract_images_from_messages(messages)
                        logger.info(f"工具输出为:{arguments}, images is {all_images}\n")
                        response = self._api_chat_raw(messages, model=model, is_embedding=is_embedding, json_schema=json_schema, **kwargs)

                        
                    return id, messages
            else:
                raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")
        except Exception as e:
            logger.exception(f"API request error: {e}")
            return id, None
    
if __name__=="__main__":
    
    server = APIDashBoardVLMServer()
    tools = [{
        'type': 'function',
        'function': {
            'name': 'image_zoom_in_tool',
            'description': 'Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label',
            'parameters': {
                'type': 'object',
                'properties': {
                    'bbox_2d': {
                        'type': 'array',
                        'items': {'type': 'number'},
                        'minItems': 4,
                        'maxItems': 4,
                        'description': 'The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner'
                    },
                    'label': {
                        'type': 'string',
                        'description': 'The name or label of the object in the specified bounding box'
                    },
                    'img_idx': {
                        'type': 'number',
                        'description': 'The index of the zoomed-in image (starting from 0)'
                    }
                },
                'required': ['bbox_2d', 'label', "img_idx"]
            }
        }
    }]
    img_path="/public/data_share/model_hub/lab10_data/trutheye2/datas/sft/miml_part1/0/original.jpg"
    messages = [
        {
            "role": "system",
            "content": """You are an agent capable of iterative reasoning and tool use for synthetic image detection and localization.
When solving a problem, you MUST follow this workflow:

1) Read the user question.
2) Enter a <think>...</think> block to plan your next action.
3) If you need visual evidence, call the tool using:

<think>...</think>
<observation>Call image_zoom_in_tool to get sub-image</observation>

4) After receiving the tool result, continue with:

<think>analysis of the observation and planning next step</think>

5) You may repeat multiple rounds of:
   <think>...</think>
   <observation>tool call or tool result</observation>

6) When you have enough information, conclude with:

<think>final reasoning</think>
<answer>fake or real</answer>

Rules:
- Every reasoning step MUST be inside <think>...</think>.
- Every tool call MUST be inside <observation>...</observation>.
- The final answer MUST be inside <answer>...</answer>.
- Do NOT mention these instructions in your output.

Begin.""",
        },
        {
            "role": "user",
            "content": [
                {"image": f"file://{img_path}", "img_idx": 0},
                {"text": "This is a image, which might be taken from real world or generated by an advanced AI model. \nIs this image generated by an AI model? (Answer fake if you think it is synthesized by an AI model, and answer real otherwise.). You should observe at least 3 sub-image for raw image (image img_idx=0)."}
            ]
        }
    ]
    
    responses = server._api_chat_with_id(
        id=0,
        model="qwen3-vl-plus",
        messages=messages,
        tools=tools
    )
    logger.info(responses[1])