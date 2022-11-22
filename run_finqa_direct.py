import json
from time import sleep
from tqdm import tqdm
import os
import openai
from datetime import datetime
from eval_tatqa.tatqa_utils import extract_one_num_from_str
from tool import *
from typing import Dict, Any
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--key", default='OPENAI_KEY', type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--greedy", default=False, action='store_true')
parser.add_argument("--dry_run", default=False, action='store_true')
parser.add_argument("--end", default=-1, type=int)
args = parser.parse_args()

def create_reader_request_processed(example: Dict[str, Any]):
    prompt = 'Read the following text and table, and then answer a question:\n'
    if example['text']:
        prompt += example['text'] + '\n'
    prompt += example['table'].strip() + '\n'
    prompt += 'Question: {}\n'.format(example['question'])
    prompt += 'Answer:'
    return prompt

prompt_4shot = """Read the following text and table, and then answer a question:
$ in millions | year ended december 2014 | year ended december 2013 | year ended december 2012
fixed income currency and commodities client execution | $ 8461 | $ 8651 | $ 9914
equities client execution1 | 2079 | 2594 | 3171
commissions and fees | 3153 | 3103 | 3053
securities services | 1504 | 1373 | 1986
total equities | 6736 | 7070 | 8210
total net revenues | 15197 | 15721 | 18124
operating expenses | 10880 | 11792 | 12490
pre-tax earnings | $ 4317 | $ 3929 | $ 5634
Question: what was the percentage change in pre-tax earnings for the institutional client services segment between 2012 and 2013?
Answer:-30.3%


Read the following text and table, and then answer a question
during the year ended march 31 , 2012 , the company has recorded $ 3.3 million in stock-based compensation expense for equity awards in which the prescribed performance milestones have been achieved or are probable of being achieved .
- | number of shares ( in thousands ) | weighted average grant date fair value ( per share )
restricted stock and restricted stock units at beginning of year | 407 | $ 9.84
granted | 607 | 18.13
vested | -134 ( 134 ) | 10.88
forfeited | -9 ( 9 ) | 13.72
restricted stock and restricted stock units at end of year | 871 | $ 15.76
Question: during the 2012 year , did the equity awards in which the prescribed performance milestones were achieved exceed the equity award compensation expense for equity granted during the year?
Answer:no


Read the following text and table, and then answer a question
annual sales of printing papers and graphic arts supplies and equipment totaled $ 3.5 billion in 2012 compared with $ 4.0 billion in 2011 and $ 4.2 billion in 2010 , reflecting declining demand and the exiting of unprofitable businesses .
in millions | 2012 | 2011 | 2010
sales | $ 6040 | $ 6630 | $ 6735
operating profit | 22 | 34 | 78
Question: what percent of distribution sales where attributable to printing papers and graphic arts supplies and equipment in 2011?
Answer:52.8%


Read the following text and table, and then answer a question
- | september 24 2005 | september 25 2004 | september 27 2003
beginning allowance balance | $ 47 | $ 49 | $ 51
charged to costs and expenses | 8 | 3 | 4
deductions ( a ) | -9 ( 9 ) | -5 ( 5 ) | -6 ( 6 )
ending allowance balance | $ 46 | $ 47 | $ 49
Question: what was the highest ending allowance balance , in millions?
Answer:51
"""

if __name__ == "__main__":
    with open('data/finqa_dev.json') as f:
        finqa_dev = json.load(f)

    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")

    correct, wrong = 0, 0

    if args.greedy:
        filename = f'outputs/finqa_direct_s{args.start}_e{args.end}_{dt_string}.jsonl'
    else:
        filename = f'outputs/finqa_direct_sc_s{args.start}_e{args.end}_{dt_string}.jsonl'
    writer = open(filename, 'w')

    for example in tqdm(finqa_dev):
        full_prompt = prompt_4shot + "\n\n"
        full_prompt += create_reader_request_processed(example)
        if args.dry_run:
            print(full_prompt)
            print('=======================')
            break

        if args.greedy:
            # greedy decoding
            got_result = False
            while not got_result:
                try:
                    result = openai.Completion.create(
                        engine='code-davinci-002',
                        prompt=full_prompt,
                        api_key=os.getenv(args.key),
                        max_tokens=512,
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
                        max_tokens=512,
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
            ans = extract_one_num_from_str(r)
            if not ans:
                if 'yes' in r.lower() or 'true' in r.lower():
                    ans = 'yes'
                elif 'no' in r.lower() or 'false' in r.lower():
                    ans = 'no'
            if ans is not None:
                if type(ans) in [dict]:
                    result_counter.update(list(ans.values()))
                elif type(ans) in [list, tuple]:
                    result_counter.update([float(ans[0])])
                elif type(ans) in [str]:
                    result_counter.update([ans])
                else:
                    try:
                        result_counter.update([float(ans)])
                    except Exception:
                        continue

        if len(result_counter) > 0:
            prediction = result_counter.most_common(1)[0][0]        
        else:
            prediction = None

        if prediction is None:
            wrong += 1
        elif finqa_equal(prediction, example['answer'], True, True):
            correct += 1
        else:
            wrong += 1

        example.update({'generated': codes, 'executed': prediction})
        writer.write(json.dumps(example) + '\n')

    print()
    print('accuracy: ', correct / (correct + wrong))
    writer.close()
