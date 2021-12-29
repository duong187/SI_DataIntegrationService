import json

with open('./hanoi_itviec.json', 'r') as f:
    dict = json.load(f)
    print(type(dict[0]))