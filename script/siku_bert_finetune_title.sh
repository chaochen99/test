export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

lr=1e-5

nohup python -m torch.distributed.launch --nnodes=1 --master_addr=10.80.205.179 --node_rank=${1}  --nproc_per_node=4   --master_port 29501 \
--use_env ./src/bert_finetune_title.py \
--config ./configs/bert_ft_${lr}.yaml \
--text_encoder SIKU-BERT/sikubert \
--output_dir /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/output/sikubert/finetune_title_lr${lr} > /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/logs/sikubert_finetune_title_lr${lr}.log 2>&1&