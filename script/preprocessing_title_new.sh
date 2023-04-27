python ./src/preprocessing_title.py \
    --image_file_dir /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/AB_img \
    --tokenizer_vocab_dir /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/tokenizer_new \
    --dataset_path /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/dataset_itm.json \
    --output_dir /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/ \
    --output_filename data_processed \
    --split_size 10000