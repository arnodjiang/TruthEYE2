import sys, os
import time

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PARENT_DIR = os.path.join(CURRENT_DIR, "..")
# sys.path.append(PARENT_DIR)

from prompts import *
from server import APIDashBoardVLMServer


class DataGenPipeline:
    def __init__(self, output_file=None, input_file=None, **kwargs):
        self.input_file = input_file
        self.output_file = output_file
        self.s1_quanju_server = APIDashBoardVLMServer() # 生成全局的理解任务，然后调用工具查看子图
        self.s1_quanju_prompt = ThinkingWithImagesPrompt()

    def pipeline_s1_quanju(self, **kwargs):
        s1_res = self.s1_quanju_server._api_chat_id_retry()

    def run(self, **kwargs):
        ## 1 计算全局理解花费的时间
        start_time = time.time()
        s1_res = self.pipeline_s1_quanju(**kwargs)

    
if __name__ == "__main__":
    pass