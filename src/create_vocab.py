# -*-coding:utf-8-*-

import argparse
import os
import json

from transformers import AutoTokenizer


def main(args):
    # data = json.load(open(args.input_path))
    # training_corpus = []
    # for page in data['documents']:
    #     for word in page['document']:
    #         training_corpus.append(word['text'])
    old_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, args.vocab_size)
    old_tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()
    main(args)