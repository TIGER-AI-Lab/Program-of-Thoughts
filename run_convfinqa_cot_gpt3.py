import json
from time import sleep
from tqdm import tqdm
import os
import openai
from datetime import datetime
from tool import finqa_equal, parse_api_result, safe_execute
from eval_tatqa.tatqa_utils import extract_one_num_from_str
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
    prompt = 'Read the following text and table, and then answer the last question in a series of questions:\n'
    if example['golden_text']:
        prompt += example['golden_text'].strip() + '\n'
    if example['golden_table']:
        prompt += example['golden_table'].strip() + '\n'
    #prompt += '\n'
    prompt += 'Questions: '
    prompt += " ".join(example['questions'][:-1])
    prompt += '\n'
    prompt += f'Question: {example["questions"][-1]}\n'
    return prompt


prompt_4shot = """Read the following text and table, and then answer the last question in a series of questions:
- | shares available for awards | shares subject to outstanding awards
2009 global incentive plan | 2322450 | 2530454
2004 stock incentive plan | - | 5923147
Questions: how many shares are subject to outstanding awards is under the 2009 global incentive plan? what about under the 2004 stock incentive plan? how many total shares are subject to outstanding awards? what about under the 2004 stock incentive plan?
Question: what proportion does this represent?
The share subject to outstanding awards under the 2009 global incentive plan is 2530454, and the share subject to outstanding awards under the 2004 stock incentive plan is 5923147. The total share subject to outstanding awards is 8453601. The proportion is 70.1%. So the answer is 70.1%.


Read the following text and table, and then answer the last question in a series of questions:
compensation expense the company recorded $ 43 million , $ 34 million , and $ 44 million of expense related to stock awards for the years ended december 31 , 2015 , 2014 , and 2013 , respectively . 
Questions: what is the compensation expense the company recorded in 2015? what about in 2014? what is the total compensation expense the company recorded in 2015 and 2014? what is the total expenses including 2013?
Question: what is the average for three years?
The compensation expense the company recorded in 2015 is $ 43 million, and the compensation expense the company recorded in 2014 is $ 34 million. The total compensation expense the company recorded in 2015 and 2014 is $ 77 million. The total expenses including 2013 is $ 121 million. The average for three years is $ 40.3 million. So the answer is $ 40.3 million.


Read the following text and table, and then answer the last question in a series of questions:
the net loss on disposal of those assets was $ 344000 for 2005 and $ 43000 for 2004 . 
Questions: what was the net loss on disposal of assets in 2005? what was the value in 2004? what was the change in value?
Question: what was the percent change?
The net loss on disposal of assets in 2005 was $ 344000, and the value in 2004 was $ 43000. The change in value is $ 301000. The percent change is 700.0%. So the answer is 700.0%.


Read the following text and table, and then answer the last question in a series of questions:
location | operations conducted | approximatesquare feet | leaseexpirationdates
dublin ireland | global supply chain distribution and administration offices | 160000 | owned
athlone ireland | commercial research and development manufacturing | 80000 | owned
bogart georgia | commercial research and development manufacturing | 70000 | owned
smithfield rhode island | commercial research and development manufacturing | 67000 | owned
Questions: what is the square feet of the owned global supply chain distribution and administration offices? what is the square feet of the owned commercial research and development manufacturing? what is the sum of those values? what is the total sum including square feet of commercial research and development manufacturing in bogart, georgia? what is the total sum including square feet of commercial research and development manufacturing in smithfield, rhode island?
Question: what is the total sum of square feet owned?
All values of square feet owned are 160000, 80000, 70000, 67000. The total sum of square feet owned is 377000. So the answer is 377000.
"""

if __name__ == "__main__":
    with open('data/convfinqa_dev.json') as f:
        convfinqa_dev = json.load(f)

    convfinqa_dev = convfinqa_dev[args.start:args.end]

    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")

    correct, wrong = 0, 0

    if args.greedy:
        filename = f'outputs/convfinqa_cot_gpt3_s{args.start}_e{args.end}_{dt_string}.jsonl'
    else:
        filename = f'outputs/convfinqa_cot_gpt3_sc_s{args.start}_e{args.end}_{dt_string}.jsonl'
    writer = open(filename, 'w')

    for example in tqdm(convfinqa_dev):
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
                        engine='text-davinci-002',
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
                        engine='text-davinci-002',
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
        codes = [code.split('answer is')[-1].strip() for code in codes]

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
