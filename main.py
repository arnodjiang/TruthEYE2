

from dotenv import load_dotenv
load_dotenv()

import os, argparse

from pipeline.data_gen_pipeline import DataGenPipeline

def init_args():
    parser = argparse.ArgumentParser(description='TruthEYE2', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--task', '-t', help='任务类型')
    parser.add_argument('--input_file', '-i', help='输入的文件（jsonl 或者 json）', default=None)
    parser.add_argument('--output_file', '-o', help='输出的文件（jsonl 或者 json）', default=None)
    args = parser.parse_args()
    return args

def main(args):
    if args.task == 'data_imcot':
        pipeline = DataGenPipeline(input_file=args.input_file,output_file=args.output_file)
        pipeline.run()

if __name__ == "__main__":
  args = init_args()
  main(args)