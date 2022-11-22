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
parser.add_argument("--greedy", default=False, action='store_true')
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=-1, type=int)
parser.add_argument("--dry_run", default=False, action='store_true')
args = parser.parse_args()

prompt_4shot = """
Read the following table regarding "Coin collections" and then write Python code to answer a question:

Name | Number of coins
Braden | 76
Camilla | 94
Rick | 86
Mary | 84
Hector | 80
Devin | 83
Emily | 82
Avery | 87

Question: Some friends discussed the sizes of their coin collections. What is the mean of the numbers?
# Python Code, return ans
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
ans = sum(number_of_coins_for_different_person) / len(number_of_coins_for_different_person)


Read the following table regarding "" and then write Python code to answer a question:

Price | Quantity demanded | Quantity supplied
$155 | 22,600 | 5,800
$275 | 20,500 | 9,400
$395 | 18,400 | 13,000
$515 | 16,300 | 16,600
$635 | 14,200 | 20,200

Question: Look at the table. Then answer the question. At a price of $155, is there a shortage or a surplus? Choose from the the options: [shortage, surplus]
# Python Code, return ans
quantity_demanded_price_155 = 22600
quantity_supplied_price_155 = 5800
if quantity_demanded_at_price_155 > quantity_supplied_at_price_155:
    ans = 'shortage'
else:
    ans = 'surplus'


Read the following table regarding "Cans of food collected" and then write Python code to answer a question:

Samir | 7
Kristen | 4
Dakota | 7
Jamie | 8
Maggie | 9

Question: Samir's class recorded how many cans of food each student collected for their canned food drive. What is the median of the numbers?
# Python Code, return ans
cans = [7, 4, 5, 8, 9]
cans = sorted(cans)
middle1 = (len(cans) - 1) // 2
middle2 = len(cans) // 2
ans = (cans[middle1] + cans[middle2]) / 2


Read the following table regarding "" and then write Python code to answer a question:

toy boat | $5.54
toy guitar | $8.23
set of juggling balls | $5.01
trivia game | $8.18
jigsaw puzzle | $5.30
toy dinosaur | $3.00

Question: Lorenzo has $13.50. Does he have enough to buy a toy guitar and a set of juggling balls? Choose from the the options: ['yes', 'no']
# Python Code, return ans
guitar_price = 8.23
juggling_balls = 5.01
total_money = 13.5
if total_money > juggling_balls + guitar_price:
    ans = "yes"
else:
    ans = "no"
"""


def create_reader_request(example: Dict[str, Any]) -> str:
    string = f'Read the following table regarding "{example["table_title"]}" and then write Python code to answer a question::\n\n'
    string += example['table'] + '\n\n'
    string += f'Question: {example["question"]}'
    if example['choices']:
        string += f' Please select from the following options: {example["choices"]}'
    string += '\n# Python Code, return ans\n'
    return string


if __name__ == "__main__":
    with open('data/tabmwp_test.json') as f:
        tabwmp_test = json.load(f)

    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")

    correct, wrong = 0, 0

    keys = list(tabwmp_test.keys())[args.start:args.end]

    if args.greedy:
        filename = f'outputs/tabmwp_s{args.start}_e{args.end}_{dt_string}.jsonl'
    else:
        filename = f'outputs/tabmwp_sc_s{args.start}_e{args.end}_{dt_string}.jsonl'
    writer = open(filename, 'w')

    for idx in tqdm(keys):
        example = tabwmp_test[idx]
        full_prompt = prompt_4shot + "\n\n"
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

        tmp = {'question': example['question'], 'answer': gt_ans,
               'executed': prediction, 'generated': codes,
               'table': example['table'], 'choices': example['choices'],
               'id': idx}
        writer.write(json.dumps(tmp) + '\n')

    writer.close()
    print()
    print(correct / (correct + wrong))
