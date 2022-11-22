import argparse
import json

args = argparse.ArgumentParser()
args.add_argument('--input', type=str)
args.add_argument('--output', type=str)
args = args.parse_args()

results = []
with open(args.input) as f:
    for line in f:
        info = json.loads(line)
        prediction = info['executed']
        print(prediction)
        qid = info['id']
        if prediction == 'yes':
            results.append({'id': qid, 'predicted': ['greater(', '1', '0', ')', 'EOF']})
        elif prediction == 'no':
            results.append({'id': qid, 'predicted': ['greater(', '0', '1', ')', 'EOF']})
        elif prediction is None:
            results.append({'id': qid, 'predicted': ['add(', '0', '0', ')', 'EOF']})
        else:
            results.append({'id': qid, 'predicted': ['add(', str(float(prediction)), '0', ')', 'EOF']})

print(len(results))
json.dump(results, open(args.output, 'w'), indent=4)