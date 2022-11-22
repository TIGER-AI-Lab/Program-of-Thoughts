import openai
from time import sleep
from tool import *
import json
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Any
import os
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--key", default='OPENAI_KEY', type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=-1, type=int)
parser.add_argument("--dry_run", default=False, action='store_true')
args = parser.parse_args()

def linearize_table(example):
    string = ''
    for k, v in example['table_for_pd'].items():
        succ = False
        try:
            v = [float(x) for x in v]
        except:
            pass
        string += k.replace(' ', '_') + ' = ' + str(v) + '\n'
    return string.strip('\n')

def create_reader_request(example: Dict[str, Any]) -> str:
    string = f"""\"\"\"
Read the following lists:
{linearize_table(example)}

Question: {example["question"]}"""
    if example['choices']:
        string += f' Options: {example["choices"]}'
    string += """
\"\"\""""
    return string


if __name__ == "__main__":
    with open('data/tabmwp_test.json') as f:
        tabwmp_test = json.load(f)

    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")

    correct, wrong = 0, 0

    keys = list(tabwmp_test.keys())[args.start:args.end]

    filename = f'outputs/tabmwp_zs_s{args.start}_e{args.end}_{dt_string}.jsonl'
    writer = open(filename, 'w')

    for idx in tqdm(keys):
        example = tabwmp_test[idx]
        full_prompt = f"""
import math
import numpy as np
import statistics

{create_reader_request(example)}
# Answer this question by implementing a solver() function. If options are given, try to select answer from the options.
def solver():
    # Let's write a Python program step by step, and then return the answer
    # Firstly, we need define the following variable:
"""
        if args.dry_run:
            print(full_prompt)
            print('=======================')
            continue

        # greedy decoding
        got_result = False
        while not got_result:
            try:
                result = openai.Completion.create(
                    engine='code-davinci-002',
                    prompt=full_prompt,
                    api_key=os.getenv(args.key),
                    max_tokens=400,
                    temperature=0.0,
                    top_p=1,
                    n=1,
                    stop=['\n\n'],
                    logprobs=1,
                    logit_bias={"1303": -2}
                )
                got_result = True
            except Exception:
                sleep(3)

        program = synthesize_program(result['choices'][0]['text'], full_prompt)
        ans = safe_execute(program)
        prediction = floatify_ans(simplify_ans(ans, True))

        # Process ground truth ansewr according to TabWMP.
        gt_ans = example['answer']
        if example['ans_type'] in ['integer_number', 'decimal_number']:
            if '/' in gt_ans:
                gt_ans = int(gt_ans.split('/')[0]) / int(gt_ans.split('/')[1])
            elif ',' in gt_ans:
                gt_ans = float(gt_ans.replace(',', ''))
            elif '%' in gt_ans:
                gt_ans = float(gt_ans.split('%')[0]) / 100
            else:
                gt_ans = float(gt_ans)
        elif example['ans_type'].endswith('_text'):
            gt_ans = str(gt_ans)
        else:
            raise ValueError(example['ans_type'])

        if finqa_equal(prediction, gt_ans):
            correct += 1
        else:
            wrong += 1

        print(program)
        print(prediction, '$',  gt_ans, '$', correct / (correct + wrong))

        tmp = {'question': example['question'], 'answer': gt_ans,
               'executed': prediction, 'generated': program,
               'table': example['table'], 'choices': example['choices'],
               'id': idx}
        writer.write(json.dumps(tmp) + '\n')

    writer.close()
    print()
    print(correct / (correct + wrong))
