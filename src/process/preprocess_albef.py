import json
import tqdm
import os

dataset = json.load(open('/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/dataset.json', 'r'))
albef_dataset = []

for page in tqdm.tqdm(dataset['documents']):
    texts = []
    for word_info in page['document']:
        texts.append(word_info['text'])
    text = ' '.join(texts)
    if os.path.exists('/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/AB_img/'+page['img']['fname']):
        albef_dataset.append({
            'image': page['img']['fname'],
            'caption': text
        })
    print(len(albef_dataset))

with open('/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/dataset_albef.json', 'w', encoding='utf-8')as f:
    f.write(json.dumps(albef_dataset, ensure_ascii=False))