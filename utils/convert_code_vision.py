# -*- coding: utf-8 -*-
import json5, json
from tqdm import tqdm
import re
import os
def split_think_tool_call(text: str):
    """
    按 <tool_call>...</tool_call> 进行切分，
    返回 ['<think>...</think>', '<tool_call>...</tool_call>'] 形式
    """
    pattern = r'(<tool_call>.*?</tool_call>)'
    parts = re.split(pattern, text, flags=re.DOTALL)

    # 去掉空字符串并 strip
    return [p.strip() for p in parts if p.strip()]
# from ..tools.code_image_python_executor import code_img_python_tool

INPUT_PATH = "/public/data_share/model_hub/lab10_data/trutheye2/datas/codevision/codevision_sft.json"
OUTPUT_PATH = "/public/data_share/model_hub/lab10_data/trutheye2/datas/codevision/codevision_sft_swift.jsonl"
img_root="/public/data_share/model_hub/lab10_data/trutheye2/datas/codevision"


# code_img_python_tool = code_img_python_tool.get_swift_description

tool_desc = {
    'type': 'function',
    'function': {
        'name': 'code_image_python_executor',
        'description': 'A unified tool for image processing using executable Python code. Supports basic operations (zoom in, resize, rotate, flip, brightness, contrast), mathematical visualizations (auxiliary lines, bounding boxes, coordinate systems, function graphs). The \'draw\' variable provides drawing capabilities.',
        'parameters': {
            'type': 'object',
            'properties': {
                'code': {
                    'description': 'Python code to process the image. Use \'image\' or \'img\' for input, \'draw\' for drawing. Important: Do not read image from file or URL, directly use the \'image\' or \'img\' variable. Return PIL Image object and save the result to \'result\' or \'output\' variable. Examples:\n- Zoom in: result = image.crop((x_min, y_min, x_max, y_max))\n- Rotate: result = image.rotate(angle, expand=True), expand=True must be set to True\n- Flip: result = image.transpose(Image.FLIP_LEFT_RIGHT) or image.transpose(Image.FLIP_TOP_BOTTOM)\n- Brightness: result = ImageEnhance.Brightness(image).enhance(factor)\n- Contrast: result = ImageEnhance.Contrast(image).enhance(factor)\nYou can also use other operations from PIL, NumPy, OpenCV, and math libraries. You should observe the image carefully and determine the best operation to perform. If the image has a wrong direction (usually happens), you should consider flipping or rotating the image to the correct direction before answering the question. Important Hint: Zoom in, Rotate and Flip are the most common operations to solve the question and should be prioritized.',
                    'type': 'string',
                },
                'description': {
                    'type': 'string',
                    'description': 'Clear description of what the code does. For example, \'Draw a line from (10, 20) to (30, 40)\', \'Flip the image horizontally\'.'
                },
                'img_idx': {
                    'type': 'integer',
                    'description': 'Index of the image in the trajectory to process (0-based).'
                }
            },
            'required': ['code','description','img_idx'],
        }
    }
}
code_img_python_tool = json.dumps([tool_desc], ensure_ascii=False)

def convert_item(item: dict) -> dict:
    """
    TODO: 在这里写你自己的转换逻辑
    item: 原始 json 中的一个元素
    return: 要写入 jsonl 的新 dict
    """
    messages = []
    img_token_num = 0
    for conv in item["conversations"]:
        if "<image>" in conv["value"]:
            img_token_num += 1
        conv["value"] = conv["value"].replace("  "," ")
        if conv["from"] == "human":
            messages.append({"role": "user", "content": conv["value"]})
        elif conv["from"] == "gpt":
            if "<tool_call>" in conv["value"] and "</tool_call>" in conv["value"]:
                # 分离 tool_call 部分和<think>部分
                tool_lists = split_think_tool_call(conv["value"])
                assert len(tool_lists) <= 2, conv["value"]+"\n\n\n"+str(tool_lists)+str(len(tool_lists))

                messages.append({"role": "assistant", "content": tool_lists[0]})  # <think> 部分
                tool_dict = json5.loads(re.search(r'<tool_call>(.*?)</tool_call>', tool_lists[1], re.DOTALL).group(1))
                if tool_dict["name"] != "code_image_tool":
                    print("Warning: tool name is not code_image_tool:", tool_dict["name"])
                    continue
                else:
                    # print(tool_dict["arguments"])
                    codes = tool_dict["arguments"]["code"]
                    img_idx = tool_dict["arguments"]["image_index"]
                    description = tool_dict["arguments"]["description"]
                    resp = {
                        "name": "code_image_python_executor",
                        "arguments": {
                            "code": codes,
                            "description": description,
                            "img_idx": img_idx
                        }
                    }
                    messages.append({"role": "tool_call", "content": json.dumps(resp, ensure_ascii=False)})  # <tool_call> 部分
            else:
                messages.append({"role": "assistant", "content": conv["value"]})
        elif conv["from"] == "tool":
        
            messages.append({"role": "tool_response", "content": "{\"image\": \"<image>\", \"img_idx\": " + str(img_token_num-1) + "}"})
    new_item = {
        "messages": messages,
        "tools": code_img_python_tool,
        "images": [os.path.join(img_root, i) for i in item["images"]]
    }
    return new_item

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "codevision_sft.json 必须是 list[dict]"

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for item in tqdm(data, desc="converting"):
            new_item = convert_item(item)
            if new_item is None:
                continue
            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
