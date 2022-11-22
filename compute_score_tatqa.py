import json
import argparse
import glob

from eval_tatqa.tatqa_metric import TaTQAEmAndF1

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", required=True, type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=1000000000, type=int)
args = parser.parse_args()

for name in glob.glob(args.inputs):
    em_and_f1 = TaTQAEmAndF1()
    with open(name) as f:
        missing = 0
        for i, line in enumerate(f):
            if i < args.start or i > args.end:
                continue
            data = json.loads(line)
            pred_ans = data['pred_answer']
            pred_scale = data['pred_scale']
            em_and_f1(data, prediction=pred_ans, pred_scale=pred_scale)
            if not pred_ans:
                missing += 1

    print('file name: ', name)
    print('num of examples: ', em_and_f1._count)
    print('num of missing returns', missing)
    print('accuracy: ', em_and_f1.get_overall_metric()[0])
