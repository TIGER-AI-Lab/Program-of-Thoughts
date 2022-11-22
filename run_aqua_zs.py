from typing import Dict, Any
import os
import json
from tqdm import tqdm
from datetime import datetime
import openai
from time import sleep
import math
import argparse
from tool import *
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--key", default='OPENAI_KEY', type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=-1, type=int)
parser.add_argument("--dry_run", default=False, action='store_true')

args = parser.parse_args()

def create_reader_request(example: Dict[str, Any]) -> str:
    string = f'# Question: {example["question"]}'
    options = [x.split(')')[-1] for x in example["options"]]
    options = ', '.join(options)

    if 'following' in string:
        string += '\n'
        string += f'# Options: {options}'
    # string += '\n'
    # string += f'# Answer options: {example["options"]}'
    return string

def prompt_for_choice(question: str, options: str, prediction: str) -> str:
    prompt = """
Find the closest options based on the question and prediction.

Question: A company produces 420 units of a particular computer component every month, at a production cost to the company of $110 per component, and sells all of the components by the end of each month. What is the minimum selling price per component that will guarantee that the yearly profit (revenue from sales minus production costs) will be at least $626,400 ?
Options: ['A)226', 'B)230', 'C)240', 'D)260', 'E)280']
Prediction: 234.28571428571428
Closest Option: B

Question: In how many ways can the letters of the word "PROBLEC" be rearranged to make 7 letter words such that none of the letters repeat?
Options: ['A)2!', 'B)3!', 'C)7!', 'D)8!', 'E)9!']
Prediction: 5040
Closest Option: C

Question: An exam is given in a certain class. The average (arithmetic mean) of the highest score and the lowest score is equal to x. If the average score for the entire class is equal to y and there are z students in the class, where z > 5, then in terms of x, y, and z, what is the average score for the class excluding the highest and lowest scorers?
Options: ['A)(zy – 2x)/z', 'B)(zy – 2)/z', 'C)(zx – y)/(z – 2)', 'D)(zy – 2x)/(z -2)', 'E)(zy – x)/(z + 2)']
Prediction: (-2*x + y*z)/(z - 2)
Closest Option: D

Question: Find the total no. of distinct bike no.'s that can beformed using 2 letters followed by 2 no.'s. How many letters need to be distinct?
Options: ["A)74453", "B)64543", "C)74325", "D)65000", "E)97656"]
Prediction = 67600
Closest Option: D

Question: A wire in the shape of rectangle of length 27 cm and breadth 17 cm is rebent to form a square. What will be the measure of each side?
Options: ['A)9', 'B)11', 'C)22', 'D)25', 'E)31']
Prediction = [-21.42428528562855, 21.42428528562855]
Closest Option: C

Question: A point on the edge of a fan blade that is rotating in a plane 10 centimeters from the center of the fan. What is the distance traveled, in centimeters, by this point after 30 seconds when the fan runs at the rate of 300 revolutions per minutes?
Options: ['A)750pi', 'B)1500pi', 'C)1875pi', 'D)3000pi', 'E)7500pi']
Prediction: 9424.77
Closest Option: D
    """
    prompt += f'\nQuestion: {question}\nOptions: {options}\nPrediction: {prediction}\nClosest Option: '
    got_result = False
    while not got_result:
        try:
            result = openai.Completion.create(
                engine='code-davinci-002',
                prompt=prompt,
                api_key=os.getenv('OPENAI_KEY_GG'),
                max_tokens=256,
                temperature=0.0,
                top_p=1,
                n=20,
                stop=['\n'],
                logprobs=1
            )
            got_result = True
        except Exception:
            sleep(3)

    return result['choices'][0]['text'].strip()


if __name__ == "__main__":
    aqua_test = []
    with open('data/aqua_test.jsonl') as f:
        for line in f:
            tmp = json.loads(line)
            aqua_test.append(tmp)

    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")

    correct, wrong = 0, 0
    
    aqua_test = aqua_test[args.start:args.end]

    filename = f'outputs/aqua_zs_s{args.start}_e{args.end}_{dt_string}.jsonl'
    print(filename)
    writer = open(filename, 'w')
    for example in tqdm(aqua_test):
        full_prompt = f"""
import math
import numpy as np
import statistics

{create_reader_request(example)}
# Answer this question by implementing a solver() function, use for loop if necessary.
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
                    max_tokens=480,
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

        if prediction is None:
            chosen_option = 'None'
        else:
            chosen_option = prompt_for_choice(
                example['question'], example['options'], prediction)

        if chosen_option == example['correct']:
            correct += 1
        else:
            wrong += 1
        
        print(program)
        print(prediction, '$', chosen_option, '$', example['correct'], '$', correct / (correct + wrong))
        tmp = {'question': example['question'],
               'generated': program,
               'generated_prediction': str(prediction),
               'options': example['options'],
               'answer': example['correct'],
               'prediction': chosen_option}

        writer.write(json.dumps(tmp) + '\n')

    writer.close()
    print()
    print(correct / (correct + wrong))
