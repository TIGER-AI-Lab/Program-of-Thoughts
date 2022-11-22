import openai
from time import sleep
from tool import *
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--key", default='OPENAI_KEY', type=str)
parser.add_argument("--dry_run", default=False, action='store_true')
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=-1, type=int)
args = parser.parse_args()

key = os.getenv(args.key)
print(key)

with open('data/gsm8K.json') as f:
    gsm_test = json.load(f)

if __name__ == "__main__":
    now = datetime.now() 
    dt_string = now.strftime("%m_%d_%H_%M")
    engine = 'code-davinci-002'

    correct, wrong = 0, 0

    gsm_test = gsm_test[args.start:args.end]

    filename = f'outputs/gsm8K_zs_s{args.start}_e{args.end}_{dt_string}.jsonl'
    print(filename)

    writer = open(filename, 'w')
    for example in tqdm(gsm_test):
        full_prompt = f"""
import math
import numpy as np

# Question: {example['question']}
# Answer this question by implementing a solver() function.
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
                    engine=engine,
                    prompt=full_prompt,
                    api_key=key,
                    max_tokens=360,
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
        prediction = floatify_ans(simplify_ans(ans, False))
        gt_ans = example['answer']
        if finqa_equal(prediction, gt_ans):
            correct += 1
        else:
            wrong += 1

        print(program)
        print(prediction, '$', gt_ans, '$', correct / (correct + wrong))

        try:
            tmp = {'question': example['question'], 'answer': gt_ans, 'executed': prediction, 'generated': program}
            writer.write(json.dumps(tmp) + '\n')
        except Exception:
            continue
            
    writer.close()
    print()
    print(correct / (correct + wrong))