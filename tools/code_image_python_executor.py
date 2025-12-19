import math
import os
import uuid
from io import BytesIO
from math import ceil, floor
from typing import List, Union

import requests
from PIL import Image

from ..logger import logger
from core.tool import BaseTool, register_tool
from utils.utils import extract_code


@register_tool('code_image_python_executor')
class PythonExecutorTool(BaseTool):
    name = 'code_image_python_executor'
    description = 'A unified tool for image processing using executable Python code. Supports basic operations (zoom in, resize, rotate, flip, brightness, contrast), mathematical visualizations (auxiliary lines, bounding boxes, coordinate systems, function graphs). The \'draw\' variable provides drawing capabilities.'
    parameters = {
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
            },
        },
        'required': ['code','img_idx','description'],
    }
    def __init__(self, cfg: dict = None):
        super().__init__(cfg)
    
    def call(self, params: Union[str, dict], **kwargs):
        pass
