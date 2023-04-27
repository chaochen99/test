# -*-coding:utf-8-*-

import argparse
from cProfile import label
from logging import raiseExceptions
import os
from posixpath import split
from PIL import Image
import pickle
import json

import torch
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3Tokenizer, XLMRobertaTokenizer, AutoTokenizer
from torchvision import transforms
from dall_e.utils import map_pixels
from dall_e import load_model
import tqdm


from utils import utils, masking_generator
from model.tokenization_layoutlmv3_albef import LayoutLMv3Tokenizer_cn


window_size = (14, 14)
num_masking_patches = 75
max_mask_patches_per_block = None
min_mask_patches_per_block = 16

# generating mask for the corresponding image
mask_generator = masking_generator.MaskingGenerator(
            window_size, num_masking_patches=num_masking_patches,
            max_num_patches=max_mask_patches_per_block,
            min_num_patches=min_mask_patches_per_block,
        )


def main(args):
    print(args, flush=True)
    device = torch.device('cpu')
    encoder = load_model("/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/encoder.pkl", device)

    # tokenizer = LayoutLMv3Tokenizer(os.path.join(args.tokenizer_vocab_dir, 'vocab.json'), os.path.join(args.tokenizer_vocab_dir, 'merges.txt'))
    tokenizer = LayoutLMv3Tokenizer_cn.from_pretrained("data/tokenizer_albef", apply_ocr=False)
    # tokenizer_t = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

    ids = range(tokenizer.vocab_size)
    vocab = tokenizer.convert_ids_to_tokens(ids)

    visual_bbox = utils.init_visual_bbox()

    ori_data = json.load(open(args.dataset_path, 'r'))['documents']

    for iter in range(0, len(ori_data), args.split_size):
        split_data = ori_data[iter : iter + args.split_size]
        # notification_slack(f"{args.output_filename}: split_file{iter}. file_size is {len(split_data)}.")
        print(f"start! extraction words and bboxes from pdf. length is {len(split_data)}", flush=True)
        # words, bboxes = utils.extraction_text_from_pdf(args.pdf_file_dir, split_file_names)
        enc_input_ids = []
        enc_bboxes = []

        for i in tqdm.tqdm(range(len(split_data)), mininterval=1):
            page = split_data[i]
            if not os.path.exists('/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/AB_img/'+page['img']['fname']):
                continue
            words = []
            bboxes = []
            for word_info in page['document']:
                words.append(word_info['text'])
                bbox = utils.normalize_bbox(word_info['box'], [page['img']['width'], page['img']['height']])
                bboxes.append(bbox)
            try:
                enc = tokenizer(text=words, boxes = bboxes, add_special_tokens=False)
                # print('enc["input_ids"]', len(enc["input_ids"]))
                
                # enc_t = tokenizer_t(''.join(words), add_special_tokens=False) 
                # print('enc_t.input_ids', len(enc_t.input_ids))
                enc_input_ids.append(enc["input_ids"])
                enc_bboxes.append(enc["bbox"])
            except Exception as e:
                print('Error: ', split_data.pop(i), e, flush=True)

        print(f"{args.output_filename}: finish tokenize.")

        #Original image (resized + normalized): pixel_values
        #Image prepared for DALL-E encoder (map_pixels): pixel_values_dall_e
        pixel_values = []
        # labels_list = []
        bool_masked_pos_list = []
        feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
        for page in tqdm.tqdm(split_data, mininterval=30):
            if not os.path.exists('/nlp_group/wuxing/yuhuimu/AB-layoutlmv3/AB_img/'+page['img']['fname']):
                continue
            image = Image.open(os.path.join(args.image_file_dir, page['img']['fname']))
            #pixel_values
            pixel_value = feature_extractor(image)["pixel_values"]
            pixel_values.append(pixel_value)
            #pixel_values_dall_e
            visual_token_transform = transforms.Compose([
                    transforms.Resize((112,112), transforms.InterpolationMode.LANCZOS),
                    transforms.ToTensor(),
                ])
            # pixel_values_dall_e = visual_token_transform(image).unsqueeze(0)
            # pixel_values_dall_e = map_pixels(pixel_values_dall_e)
            with torch.no_grad():
                # z_logits = encoder(pixel_values_dall_e)
                # input_ids = torch.argmax(z_logits, axis=1).flatten(1)
                #create mask position
                bool_masked_pos = mask_generator()
                bool_masked_pos = torch.from_numpy(bool_masked_pos).unsqueeze(0)
                bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
                # labels = input_ids[bool_masked_pos]
                # labels_list.append(labels)
                bool_masked_pos_list.append(bool_masked_pos)

        # notification_slack(f"{args.output_filename}: start create subset token!.")
        print("start! create subset tokens", flush=True)
        tokens, bboxes, doc_ids = utils.subset_tokens_from_document_light_albef(enc_input_ids, enc_bboxes, vocab, max_len=512)
        print(f"{args.output_filename}: fiish create subset token! and createing dataset.")
        dataset = []
        for i in tqdm.tqdm(range(len(tokens)), mininterval=30):
            al_labels = utils.create_alignment_label(
                visual_bbox=visual_bbox,
                text_bbox=bboxes[i],
                bool_mi_pos=bool_masked_pos_list[doc_ids[i]][0],
                )
            dataset.append({"input_ids": tokens[i], 
            "bbox": bboxes[i],

            "pixel_values": pixel_values[doc_ids[i]][0], 
            # "label": labels_list[doc_ids[i]],
            "bool_masked_pos": bool_masked_pos_list[doc_ids[i]][0],
            "alignment_labels": al_labels
            })
        print(f"{args.output_filename}: start saving....")

        with open(f"{args.output_dir}{args.output_filename}/wpa.pkl", 'wb') as f:
            pickle.dump(dataset, f, protocol=4)
        print(f"{args.output_filename}: saved: {iter}.pkl")
    
    # notification_slack(f"{args.output_filename}: finish all process!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_vocab_dir", type=str)
    parser.add_argument("--image_file_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)
    parser.add_argument("--datasize", type=int)
    parser.add_argument("--split_size", type=int)
    args = parser.parse_args()
    main(args)