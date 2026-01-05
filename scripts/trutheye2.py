#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import time
import torch
import json
import multiprocessing as mp
from tqdm import tqdm
import argparse
from datasets import load_dataset
from loguru import logger
import json5
from PIL import Image
import tempfile, uuid, random

from transformers import AutoModelForImageTextToText, AutoProcessor,AutoTokenizer
from qwen_vl_utils import process_vision_info

import json5, re

def extract_tool_call(content):
    think_content = re.findall("<think>(.*?)</think>", content, flags=re.DOTALL)
    tool_call_content = re.findall("<tool_call>(.*?)</tool_call>", content, flags=re.DOTALL)
    answer_call_content = re.findall("<answer>(.*?)</answer>", content, flags=re.DOTALL)

    answer_call_content = answer_call_content[0].strip() if answer_call_content else ""
    think_content = think_content[0].strip() if think_content else ""
    tool_call_content = tool_call_content[0].strip() if tool_call_content else ""

    if (think_content and tool_call_content) or answer_call_content:
        return {
            "think": think_content,
            "tool_call": json5.loads(tool_call_content)["arguments"] if tool_call_content else "",
            "answer": answer_call_content
        }
    else:
        return {
            "think": "",
            "tool_call": "",
            "answer": ""
        }

def get_answer(text: str):
    """
    1. 去掉 <think>...</think> 标签
    2. 尝试直接 json5.loads
    3. 若失败，提取 ```json ... ``` 内容再 loads
    4. 仍失败则返回 None
    """
    if not text or not isinstance(text, str):
        return None

    # Step 1: remove <think>...</think>
    cleaned = re.sub(r"<think>.*</think>", "", text, flags=re.DOTALL).strip()

    # Step 2: try direct json5 loads
    try:
        return json5.loads(cleaned)
    except Exception:
        pass

    # Step 3: extract ```json ... ``` block
    match = re.search(r"```json\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if match:
        json_block = match.group(1).strip()
        try:
            return json5.loads(json_block)
        except Exception:
            pass

    # Step 4: all failed
    # return {"label": "real"}
    assert False, text

def get_prob(tokenizer,response, probs):
    if response=="fake":
        label_token_id = int(tokenizer.encode("fake")[0])
    elif response=="Fake":
        label_token_id = int(tokenizer.encode("Fake")[0])
    elif response=="FAKE":
        label_token_id = int(tokenizer.encode("FAKE")[0])
    else:
        label_token_id = int(tokenizer.encode("fake")[0])
    pro_token_id = probs[0,label_token_id].item()
    return pro_token_id

def init_model(model_path, device_id):
  model = AutoModelForImageTextToText.from_pretrained(
      model_path,
      torch_dtype=torch.bfloat16,
      attn_implementation="flash_attention_2"
  ).eval().to(f"cuda:{device_id}")
  processor = AutoProcessor.from_pretrained(model_path)
  tokenizer = AutoTokenizer.from_pretrained(model_path)

  return model, processor, tokenizer

def exec_code(code_str, messages, tmpdir):
    images = extract_images_from_messages(messages)

    # 解析输入图片路径
    
    img_ref = images[code_str["tool_call"]["img_idx"]]
    if img_ref.startswith("file://"):
        img_path = img_ref.replace("file://", "")
    else:
        img_path = img_ref

    image = Image.open(img_path)

    # 使用临时目录
    output_path = os.path.join(tmpdir, f"{uuid.uuid4()}.png")

    ns = {
        "image": image,
        "Image": Image,
        "ImageFile": ImageFile,
        "ImageDraw": ImageDraw,
        "ImageChops": ImageChops,
        "ImageFilter": ImageFilter,
        "ImageOps": ImageOps
    }

    exec(code_str["tool_call"]["code"], ns)

    result = ns["result"]

    result = ns.get("result")
    def _get_first(ns, keys):
        """从 ns 中按顺序取第一个存在且不为 None 的值"""
        for k in keys:
            if k in ns and ns[k] is not None:
                return ns[k]
        return None
    # ===== 从 exec namespace 中鲁棒提取 bbox =====
    x1 = _get_first(ns, ["x_min", "xmin", "left", "x1"])
    y1 = _get_first(ns, ["y_min", "ymin", "top", "y1"])
    x2 = _get_first(ns, ["x_max", "xmax", "right", "x2"])
    y2 = _get_first(ns, ["y_max", "ymax", "bottom", "y2"])

    bbox = []
    # logger.debug(f"{x1},{y1},{x2},{y2}")
    # 确保四个值都存在
    if all(v is not None for v in [x1, y1, x2, y2]):
        try:
            # 强制转为 int（防止模型给 float / numpy）
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # 自动修正顺序
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # 简单合法性检查（可选）
            if x2 > x1 and y2 > y1:
                bbox = [x1, y1, x2, y2]
        except Exception:
            bbox = []

    # 保存到临时文件
    try:
        result.save(output_path)
        return {
                "type": "image",
                "image": f"file://{output_path}"
            }, bbox
    except Exception as E:
        return {
                "type": "text",
                "text": f"Error in execute code {E}. Do NOT generate any code that assigns or outputs an `answer` variable or directly produces labels such as \"real\" or \"fake\". Your code must only perform image processing and produce intermediate results (e.g., result image). Final classification is handled externally."
            }, []

def extract_images_from_messages(messages):
    files = []
    for msg in messages:
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if item.get("image") and item["image"] not in files:
                    files.append(item["image"])
    return files

def model_chat(model, processor, messages, **kwargs):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)

    inputs = processor(text=text, images=images, videos=videos, do_resize=False, return_tensors="pt")
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text

def to_file(data, file_path):
    """
    保存数据到指定路径，根据后缀自动选择 json 或 jsonl 格式。
    data: List[dict]
    file_path: str, 例如 "xxx.json" 或 "xxx.jsonl"
    """
    # 确保目录存在
    dir_name = os.path.dirname(file_path)
    if dir_name:  # 避免空字符串
        os.makedirs(dir_name, exist_ok=True)

    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".json":
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif ext == ".jsonl":
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def iter_files_1(root_dir: str):
    """递归扫描目录下所有图片/视频"""
    IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"
    }

    VIDEO_EXTENSIONS = {
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpg", ".mpeg", ".m4v"
    }
    if os.path.isdir(root_dir):
        tos=[]
        for r, _, files in os.walk(root_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS:
                    tos.append(os.path.join(r, fn))
        return tos
    elif root_dir.endswith(".jsonl"):
      full_dataset = load_dataset('json', data_files={'test': root_dir})['test']
    elif root_dir.endswith(".json"):
        with open(root_dir, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    return full_dataset
    
def build_messages(file_path):
    
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
                        'type': 'string'
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
    txtra_desc = "{\"name\": <function-name>, \"arguments\": <args-json-object>}"

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": f"You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{code_img_python_tool}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{txtra_desc}\n</tool_call>"}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                },
                {"type": "text", "text": "Is the image taken from the real world or generated by an advanced AI model? Answer 'real' if you think it is real, or 'fake' if you think it is AI-generated."}
            ]
        }
    ]

        # Locate the shadow of the paper fox, report the bbox coordinates in JSON format.
# ====== 模拟推理 ======
def run_inference(model, processor, data_itm, **kwargs):
    if os.environ.get("IMG_ROOT"):
      data_itm["file_path"] = os.path.join(os.environ.get("IMG_ROOT"),data_itm["file_path"])
    messages = build_messages(file_path=data_itm["file_path"])
    # logger.info(f"Start: 模型输入为：\n{messages}")
    answer_bbox = []
    raw_response = None
    
    res = ""
    output_text = ""
    bboxes = []
    times = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        while True:
            output_text = model_chat(model, processor, messages)
            # print(output_text)
            
            res = extract_tool_call(output_text)
            if times >=5:
                break
            times += 1
            if res.get("answer"):
                messages.append({
                    "role": "assistant", "content": output_text
                })
                break
            elif not res or ("<tool_call>" in output_text and "</tool_call>" in output_text):
                try:
                    code_result, bbox = exec_code(res,messages,tmpdir)
                    if bbox:
                        bboxes.append(bbox)
                except Exception as E:
                    code_result = {"type": "text", "text":str(E)}
                messages.extend(
                    [
                        {"role":"assistant", "content": res["think"]},
                        {"role":"tool_call", "content": res["tool_call"]},
                        {"role":"tool_response", "content": [code_result]}
                    ]
                )

        data_itm.update({
            "raw_response": messages,
            "answer": res.get("answer", random.choice(["real", "fake"])),
            "answer_bbox": bboxes
        })
    return data_itm

# ====== worker 进程 ======
def worker_loop(gpu_id, task_queue, result_queue, model_path):
    torch.cuda.set_device(gpu_id)   # 绑定 GPU
    model,processor,tokenizer = init_model(model_path,gpu_id)

    while True:
        data_itm = task_queue.get()
        if data_itm == "exit":  # 结束信号
            break
        try:
            result = run_inference(model, processor, data_itm, tokenizer=tokenizer)
            if result:
                result_queue.put(result)
            else:
                result_queue.put(None)
        except Exception as e:
            logger.exception(f"[Worker {gpu_id}] ERROR on: {e}")
            result_queue.put(None)

    print(f"[Worker {gpu_id}] Exiting.")


def main():
    ap = argparse.ArgumentParser(description="并行收集文件路径并保存 JSONL")
    ap.add_argument("--root", required=True, help="遍历的根目录")
    ap.add_argument("--to_path", help="保存的目录")
    ap.add_argument("--model_path", help="保存的目录")
    ap.add_argument("--batch_size", type=int, default=1000, help="批处理大小")
    args = ap.parse_args()
    num_gpus = torch.cuda.device_count()
    print(f"[INFO] Found {num_gpus} GPUs")
    print(f"args is {vars(args)}")

    # 1 get files
    files = list(iter_files_1(args.root))

    # 2 队列
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # 3 启动 worker，4 个 workers 对象
    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_loop, args=(gpu_id, task_queue, result_queue, args.model_path))
        p.start()
        workers.append(p)

    # 4 投递任务到队列中
    for f in files:
        task_queue.put(f)

    # 6 收集结果并分批写入 jsonl
    collected = 0
    finished = 0
    os.makedirs(os.path.dirname(args.to_path), exist_ok=True)
    with open(args.to_path, "w", encoding="utf-8") as f, tqdm(total=len(files), desc="Processing") as pbar:
        while finished < len(files):
            res = result_queue.get()
            if res:
                # buffer.append(res)
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                collected += 1

                if collected % 1000 == 0 or collected==20:
                    f.flush()

            finished += 1
            pbar.update(1)

    # 5 投递结束信号
    for _ in range(num_gpus):
        task_queue.put("exit")

    # 7 等待子进程退出
    for p in workers:
        p.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()