import json
import os
import math

path = '/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/dataset_albef.json'

data = json.load(open(path, 'r'))

n_train = math.floor(len(data) * 0.9)
valid_data = data[n_train:]

with open('/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/dataset_albef_itm.json', 'w', encoding='utf-8')as f:
    f.write(json.dumps(valid_data, ensure_ascii=False))