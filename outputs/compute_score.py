import json
import argparse
import glob
import sys
sys.path.append("../")
from tool import finqa_equal

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", required=True, type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=1000000000, type=int)
parser.add_argument("--relaxed", default=False, action='store_true')
parser.add_argument("--tolerance", default=False, action='store_true')
parser.add_argument("--show", default=False, action='store_true')
args = parser.parse_args()

for name in glob.glob(args.inputs):
    correct, wrong, missing = 0, 0, 0
    with open(name) as f:
        for i, line in enumerate(f):
            if i < args.start or i > args.end:
                continue
            data = json.loads(line)
            if 'demonstration' in data or 'prompt' in data:
                continue
            else:
                # Select the prediction field.
                if 'executed' in data:
                    prediction = data['executed']
                else:
                    prediction = data['prediction']
                    if args.relaxed:
                        # For multi-choise question, pick A option
                        if prediction == 'None':
                            prediction = 'A'
                # Calculate the evaluation score
                if prediction is None:
                    missing += 1
                elif finqa_equal(prediction, data['answer'], args.relaxed, args.tolerance):
                    correct += 1
                else:
                    if args.show:
                        print(data['question'])
                        print(data['executed'], data['answer'])
                        print()
                    wrong += 1

    print('file name: ', name)
    print('num of examples: ', correct + wrong + missing)
    print('num of missing returns', missing)
    print('accuracy: ', correct / (correct + wrong + missing))
