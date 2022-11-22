import openai
from time import sleep
from tool import *
from typing import Dict, Any
from datetime import datetime
from tqdm import tqdm
import os
import json
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--key", default='OPENAI_KEY', type=str)
parser.add_argument("--dry_run", default=False, action='store_true')
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=-1, type=int)
args = parser.parse_args()


with open('data/multiarith.json') as f:
    multiarith_data = json.load(f)

if __name__ == "__main__":
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")

    correct, wrong = 0, 0

    multiarith_data = multiarith_data[args.start:args.end]
    filename = f'outputs/multiarith_zs_s{args.start}_e{args.end}_{dt_string}.jsonl'
    print(filename)
    writer = open(filename, 'w')
    for example in tqdm(multiarith_data):
        full_prompt = f"""
import math
import numpy as np

# Question: {example['sQuestion'].strip()}
# Answer this question by implementing a solver() function.
def solver():
    # Let's think step by step to derive the answer, and then return the answer
    # In the first step, we can define the following variable:
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
                    max_tokens=300,
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
        prediction = floatify_ans(ans)
        if prediction is not None:
            prediction = abs(prediction)
        
        gt_ans = example['lSolutions'][0]
        if finqa_equal(prediction, gt_ans):
            correct += 1
        else:
            wrong += 1

        print(program)
        print(prediction, '$', gt_ans, '$', correct / (correct + wrong))

        tmp = {'question': example['sQuestion'], 'executed': prediction, 
               'generated': program, 'answer': gt_ans}
        writer.write(json.dumps(tmp) + '\n')

    writer.close()
    print()
    print(correct / (correct + wrong))
