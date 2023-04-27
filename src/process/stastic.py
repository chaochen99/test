import json
import numpy as np


datatset = json.load(open('/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/dataset_albef.json', 'r'))
total_len = []

for item in datatset:
    total_len.append(len(item['caption'].replace(' ', '')))

total_len = np.array(total_len)

print(np.mean(total_len))
print(np.percentile(total_len,60))
print(np.percentile(total_len,65))
print(np.percentile(total_len,70))
print(np.percentile(total_len, 75))
print(np.percentile(total_len, 80))
print(np.percentile(total_len, 85))
print(np.percentile(total_len, 90))
print(np.percentile(total_len, 95))

