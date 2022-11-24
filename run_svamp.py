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

# Passage: James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop.
# Question: How many red stickers does James have?
original_red_stickers = 93
used_red_stickers = 31
ans = original_red_stickers - used_red_stickers

# Passage: Allen went to supermarket to buy eggs, each egg costs 80 dollars, if the discount is 29 dollars.
# Question: How much do you have to pay to buy for each egg?
original_egg_price_in_dollars = 80
discount_dollars = 29
ans = original_egg_price_in_dollars - discount_dollars

# Passage: Dianna collects both cases and books. He bought 22 cases and 5 books from the store. Now he has 57 cases and 25 books.
# Question: How many books did danny have at first?
num_books_bought_at_store = 5
num_books_now = 25
ans = num_books_now - num_books_bought_at_store

# Passage: There were 108 chickens and 20 sheeps at the farm, some of chickens and sheeps were sold. There are 87 chickens and 18 sheeps left now.
# Question: How many chickens were sold?
num_chicken_before = 108
num_chicken_now = 87
ans = num_chicken_before - num_chicken_now

# Passage: Katty scored 2 goals on monday, 8 goals on tuesday and 9 goals on wednesday.
# Question: How many did Katty score on monday and wednesday?
num_goals_on_monday = 2
num_goals_on_wednesday = 9
ans = num_goals_on_monday + num_goals_on_wednesday

# Passage: There are 5 girls and 4 boys in the Masquerade, 12 more girls and 7 more boys joined. 
# Question: How many more girls than boys are in the Masquerade?
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
ans = total_girls - total_boys

# Passage: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creasm, and he has 2 ice creams left now. 
# Question: How much ice creasm did Getty purchase?
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
ans = total_ice_creams - num_ice_creams_bought_by_joseph
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
                result_counter.update([abs(ans)])

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
