from typing import Dict, Any
import os
import json
from tqdm import tqdm
from datetime import datetime
import openai
from time import sleep
import sympy
from sympy.solvers import solve
from sympy import Symbol
import math
import argparse
from tool import simplify_ans, parse_api_result, safe_execute
from sympy import simplify
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--key", default='OPENAI_KEY', type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=-1, type=int)
parser.add_argument("--greedy", default=False, action='store_true')
parser.add_argument("--dry_run", default=False, action='store_true')

args = parser.parse_args()

def create_reader_request(example: Dict[str, Any]) -> str:
    string = f'# Question: {example["question"]}\n'
    string += f'# Answer option: {example["options"]}'
    return string

prompt_4shot = """
# Write Python Code to solve the following questions. Store your result as a variable named 'ans'.
from sympy import Symbol
from sympy import simplify
import math
from sympy import solve_it
# solve_it(equations, variable): solving the equations and return the variable value.

# Question: In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr and the time of flight increased by 30 minutes. The duration of the flight is:
# Answer option: ['A)1 hour', 'B)2 hours', 'C)3 hours', 'D)4 hours', 'E)5 hours']
duration = Symbol('duration', positive=True)
delay = 30 / 60
total_disntace = 600
original_speed = total_disntace / duration
reduced_speed = total_disntace / (duration + delay)
solution = solve_it(original_speed - reduced_speed - 200, duration)
ans = solution[duration]

# Question: M men agree to purchase a gift for Rs. D. If 3 men drop out how much more will each have to contribute towards the purchase of the gift?
# Answer options: ['A)D/(M-3)', 'B)MD/3', 'C)M/(D-3)', 'D)3D/(M2-3M)', 'E)None of these']
M = Symbol('M')
D = Symbol('D')
cost_before_dropout = D / M
cost_after_dropout = D / (M - 3)
ans=simplify(cost_after_dropout - cost_before_dropout)

# Question: A sum of money at simple interest amounts to Rs. 815 in 3 years and to Rs. 854 in 4 years. The sum is:
# Answer option: ['A)Rs. 650', 'B)Rs. 690', 'C)Rs. 698', 'D)Rs. 700', 'E)None of these']
deposit = Symbol('deposit', positive=True)
interest = Symbol('interest', positive=True)
money_in_3_years = deposit + 3 * interest
money_in_4_years = deposit + 4 * interest
solution = solve_it([money_in_3_years - 815, money_in_4_years - 854], [deposit, interest])
ans = solution[deposit]

# Question: Find out which of the following values is the multiple of X, if it is divisible by 9 and 12?
# Answer option: ['A)36', 'B)15', 'C)17', 'D)5', 'E)7']
options = [36, 15, 17, 5, 7]
for option in options:
    if option % 9 == 0 and option % 12 == 0:
        ans = option
        break

# Question: 35% of the employees of a company are men. 60% of the men in the company speak French and 40% of the employees of the company speak French. What is % of the women in the company who do not speak French?
# Answer option: ['A)4%', 'B)10%', 'C)96%', 'D)90.12%', 'E)70.77%']
num_women = 65
men_speaking_french = 0.6 * 35
employees_speaking_french = 0.4 * 100
women_speaking_french = employees_speaking_french - men_speaking_french
women_not_speaking_french=num_women - women_speaking_french
ans = women_not_speaking_french / num_women

# Question: In one hour, a boat goes 11 km/hr along the stream and 5 km/hr against the stream. The speed of the boat in still water (in km/hr) is:
# Answer option: ['A)4 kmph', 'B)5 kmph', 'C)6 kmph', 'D)7 kmph', 'E)8 kmph']
boat_speed = Symbol('boat_speed', positive=True)
stream_speed = Symbol('stream_speed', positive=True)
along_stream_speed = 11
against_stream_speed = 5
solution = solve_it([boat_speed + stream_speed - along_stream_speed, boat_speed - stream_speed - against_stream_speed], [boat_speed, stream_speed])
ans = solution[boat_speed]

# Question: The difference between simple interest and C.I. at the same rate for Rs.5000 for 2 years in Rs.72. The rate of interest is?
# Answer option: ['A)10%', 'B)12%', 'C)6%', 'D)8%', 'E)4%']
interest_rate = Symbol('interest_rate', positive=True)
amount = 5000
amount_with_simple_interest = amount * (1 + 2 * interest_rate / 100)
amount_with_compound_interest = amount * (1 + interest_rate / 100) ** 2
solution = solve_it(amount_with_compound_interest - amount_with_simple_interest - 72, interest_rate)
ans = solution[interest_rate]

# Question: The area of a rectangle is 15 square centimeters and the perimeter is 16 centimeters. What are the dimensions of the rectangle?
# Answer option: ['A)2&4', 'B)3&5', 'C)4&6', 'D)5&7', 'E)6&8']
width = Symbol('width', positive=True)
height = Symbol('height', positive=True)
area = 15
permimeter = 16
solution = solve_it([width * height - area, 2 * (width + height) - permimeter], [width, height])
ans = (solution[width], solution[height])
"""

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

    if args.greedy:
        filename = f'outputs/aqua_s{args.start}_e{args.end}_{dt_string}.jsonl'
    else:
        filename = f'outputs/aqua_sc_s{args.start}_e{args.end}_{dt_string}.jsonl'
        
    writer = open(filename, 'w')
    writer.write(json.dumps({'demonstration': prompt_4shot}) + '\n')
    for example in tqdm(aqua_test):
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
                        temperature=0.3,
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
            pred = simplify_ans(ans)
            if pred is not None:
                result_counter.update([pred])
        print(result_counter)

        if len(result_counter) > 0:
            prediction = result_counter.most_common(1)[0][0]        
        else:
            prediction = None

        if prediction is None:
            chosen_option = 'A'
        else:
            chosen_option = prompt_for_choice(
                example['question'], example['options'], prediction)

        if chosen_option == example['correct']:
            correct += 1
        else:
            wrong += 1

        tmp = {'question': example['question'],
               'generated': codes,
               'generated_prediction': str(prediction),
               'options': example['options'],
               'answer': example['correct'],
               'prediction': chosen_option}

        writer.write(json.dumps(tmp) + '\n')

    writer.close()
    print()
    print(correct / (correct + wrong))
