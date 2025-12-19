import re, json5

from ..logger import logger

def extract_images_from_messages(messages):
    files = []
    for msg in messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                if item.image and item.image not in files:
                    files.append(item.image)
    return files

def extract_code(text: str) -> str:
    # Match triple backtick blocks first
    triple_match = re.search(r'```[^\n]*\n(.+?)```', text, re.DOTALL)
    if triple_match:
        text = triple_match.group(1)
    else:
        try:
            text = json5.loads(text)['code']
        except Exception as e:
            logger.exception(e)
    # If no code blocks found, return original text
    return text

def split_think_tool_call(text: str):
    """
    按 <tool_call>...</tool_call> 进行切分，
    返回 ['<think>...</think>', '<tool_call>...</tool_call>'] 形式
    """
    pattern = r'(<tool_call>.*?</tool_call>)'
    parts = re.split(pattern, text, flags=re.DOTALL)

    # 去掉空字符串并 strip
    return [p.strip() for p in parts if p.strip()]

if __name__ == "__main__":
    sample_text = """
    <think>
    Here is my thought process.
    </think>
    <tool_call>
    {"name": "code_image_tool", "arguments": {"code": "result = image.rotate(90, expand=True)", "description": "Rotate the image 90 degrees clockwise", "img_idx": 0}}
    </tool_call><answer>asd</answer><think><>
    """
    parts = split_think_tool_call(sample_text)
    print(parts)