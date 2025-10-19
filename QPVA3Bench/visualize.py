import json
from pprint import pprint

dataset = json.load(open('QPVA3Bench.json', 'r'))
dataset = list(dataset.values())
print('Input the idx to visualize. -1 to exit.')
while True:
    idx = int(input())
    if idx < 0:
        break
    pprint(dataset[idx], width=200)


