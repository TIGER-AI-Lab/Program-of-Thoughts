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
parser.add_argument("--greedy", default=False, action='store_true')
parser.add_argument("--dry_run", default=False, action='store_true')
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=-1, type=int)
args = parser.parse_args()


def create_reader_request(example: Dict[str, Any]) -> str:
    string = f'# Passage: {example["Body"]}\n'
    string += f'# Question: {example["Question"]}\n'
    return string

with open('data/SVAMP.json') as f:
    svamp_data = json.load(f)

prompt_4shot = """
Read the following passages to answer questions with Python code, store the result as a 'ans' variable:

# Passage: Each pack of dvds costs 76 dollars. If there is a discount of 25 dollars on each pack
# Question: How much do you have to pay to buy each pack?
original_price = 76
discount = 25
ans = original_price - discount

# Passage: Paco had 26 salty cookies and 17 sweet cookies. He ate 14 sweet cookies and 9 salty cookies.
# Question: How many salty cookies did Paco have left?
original_salty_cookies = 26
eaten_salty_cookies = 9
ans = original_salty_cookies - eaten_salty_cookies

# Passage: Haley grew 9 trees in her backyard. After a typhoon 4 died. Then she grew 5 more trees.
# Question: How many trees does she have left?
original_trees = 9
died_trees = 4
regrew_trees = 5
ans = original_trees - died_trees + regrew_trees

# Passage: Julia played tag with 5 kids on monday, 9 kids on tuesday and 15 kids on wednesday.
# Question: How many kids did she play with on monday and wednesday?
num_kids_on_monday = 5
num_kids_on_wednesday = 15
ans = num_kids_on_monday + num_kids_on_wednesday

# Passage: Danny collects bottle caps and wrappers. He found 22 bottle caps and 30 wrappers at the park. Now he has 17 bottle caps and 57 wrappers in his collection.
# Question: How many wrappers did danny have at first?
num_wrappers_now = 57
found_wrappers = 30
ans = num_wrappers_now - found_wrappers

# Passage: There were 16 roses and 3 orchids in the vase. Jessica cut some more roses and orchids from her flower garden. There are now 7 orchids and 13 roses in the vase.
# Question: How many orchids did she cut?
num_orchids_now = 13
num_orchids_before = 3
ans = num_orchids_now - num_orchids_before
"""

if __name__ == "__main__":
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")

    correct, wrong = 0, 0

    svamp_data = svamp_data[args.start:args.end]

    if args.greedy:
        filename = f'outputs/svamp_s{args.start}_e{args.end}_{dt_string}.jsonl'
    else:
        filename = f'outputs/svamp_sc_s{args.start}_e{args.end}_{dt_string}.jsonl'
    writer = open(filename, 'w')

    writer.write(json.dumps({'demonstration': prompt_4shot}) + '\n')
    for example in tqdm(svamp_data):
        full_prompt = prompt_4shot + "\n"
        full_prompt += create_reader_request(example)
        if args.dry_run:
            print(full_prompt)
            print('=======================')
            continue

        if args.greedy:
            # greedy decoding
            got_result = False
            while not got_result:
                try:
                    result = openai.Completion.create(
                        engine='code-davinci-002',
                        prompt=full_prompt,
                        api_key=os.getenv(args.key),
                        max_tokens=256,
                        temperature=0.0,
                        top_p=1,
                        n=1,
                        stop=['\n\n'],
                        logprobs=1
                    )
                    got_result = True
                except Exception:
                    sleep(3)
        else:
            # self-consistency decoding
            got_result = False
            while not got_result:
                try:
                    result = openai.Completion.create(
                        engine='code-davinci-002',
                        prompt=full_prompt,
                        api_key=os.getenv(args.key),
                        max_tokens=256,
                        temperature=0.5,
                        top_p=1,
                        n=30,
                        stop=['\n\n'],
                        logprobs=1
                    )
                    got_result = True
                except Exception as e:
                    sleep(3)

        # self-consistency decoding or greedy decoding.
        result_counter = Counter()
        codes = parse_api_result(result)
        for r in codes:
            ans = safe_execute(r)
            ans = floatify_ans(ans)
            if ans is not None:
                result_counter.update([ans])

        if len(result_counter) > 0:
            prediction = result_counter.most_common(1)[0][0]
        else:
            prediction = None

        if finqa_equal(prediction, example['Answer']):
            correct += 1
        else:
            wrong += 1

        tmp = {'question': example['Question'], 'passage': example['Body'],
               'executed': prediction, 'generated': codes, 'answer': example['Answer']}

        writer.write(json.dumps(tmp) + '\n')

    writer.close()
    print()
    print(correct / (correct + wrong))
