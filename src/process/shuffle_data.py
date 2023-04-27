import random
import os
import pickle
import json

random.seed(10)
data=[]
root = '/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/data_processed'
input_names = os.listdir(root)
# notification_slack(f"input_file_length: {len(input_names)}")
for file_name in input_names:
    if file_name[0] in ['i','a','t']:
        continue
    with open(f"{root}/{file_name}", "rb") as f:
        d = pickle.load(f)
        data += d

random.shuffle(data)

print(len(data))

with open(f"{root}/all.pkl", 'wb') as f:
    pickle.dump(data, f, protocol=4)

# random.seed(10)
# data=[]
# root = '/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/data_processed_ori'
# input_names = os.listdir(root)
# # notification_slack(f"input_file_length: {len(input_names)}")
# for file_name in input_names:
#     with open(f"{root}/{file_name}", "rb") as f:
#         d = pickle.load(f)
#         data += d

# random.shuffle(data)

# with open(f"{root}/all.pkl", 'wb') as f:
#     pickle.dump(data, f, protocol=4)


# random.seed(10)
# data = json.load(open('/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/dataset_albef.json', 'r'))

# random.shuffle(data)

# with open('/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/dataset_albef.json', 'w', encoding='utf-8')as f:
#     f.write(json.dumps(data, ensure_ascii=False))