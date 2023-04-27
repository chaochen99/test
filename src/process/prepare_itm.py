import pickle
import os
import json
import math
import tqdm
import random
import copy

random.seed(10)

path = '/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/data_processed_ori/title_val.pkl'

with open(path, "rb") as f:
    data = pickle.load(f)

l = len(data)

data_new = []

for i in range(len(data)):
    j = i
    while i == j:
        j = random.randint(0, l-1) 
    item =  copy.deepcopy(data[i])
    item['input_ids'] = data[j]['input_ids']
    item['bbox'] = data[j]['bbox']
    data[i]['itm_flag'] = 1
    item['itm_flag'] = 0
    data_new.append(data[i])
    data_new.append(item)

random.seed(11)
random.shuffle(data_new)

print(len(data_new))

with open('/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/data_processed_ori/itm_val.pkl', 'wb') as f:
    pickle.dump(data_new, f, protocol=4)

# path = '/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/data_processed/title.pkl'

# with open(path, "rb") as f:
#     data = pickle.load(f)

# n_train = math.floor(len(data) * 0.9)
# valid_data = data[n_train:]


# with open('/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/data_processed/title_val.pkl', 'wb') as f:
#     pickle.dump(valid_data, f, protocol=4)
