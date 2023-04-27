export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5


batch_size=16
learning_rate=1e-5
name=albef_finetune_wpa_lr${learning_rate}_bs_${batch_size}

mkdir -p /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/output/${name}/

nohup python -m torch.distributed.launch --nnodes=1 --master_addr=10.80.205.179 --node_rank=${1}  --nproc_per_node=4   --master_port 29501 \
--use_env ./src/albef_finetune_wpa.py \
--input_file /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/data_processed_albef_wpa/ \
--tokenizer_vocab_dir /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/tokenizer/ \
--output_model_dir /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/output/${name}/ \
--model_name microsoft/layoutlmv3-base-chinese \
--ratio_train 0.9 \
--batch_size ${batch_size} \
--learning_rate ${learning_rate} \
--max_epochs 40 \
--pretrained /nlp_group/wuxing/ALBEF/output/Pretrain_zh/succ_model/checkpoint_29.pth > logs/${name}_${1}.log 2>&1

wait 

batch_size=12
learning_rate=1e-6
tokenizer_tag=
data_tag=_ori
name=finetune_wpa${tokenizer_tag}_lr${learning_rate}_bs_${batch_size}

mkdir -p /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/output/${name}/

nohup python -m torch.distributed.launch --nnodes=1 --master_addr=10.80.205.179 --node_rank=${1}  --nproc_per_node=4   --master_port 29501 \
--use_env ./src/finetune_wpa.py \
--input_file /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/data_processed${data_tag}/ \
--tokenizer_vocab_dir /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/tokenizer${tokenizer_tag}/ \
--output_model_dir /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/output/${name}/ \
--model_name microsoft/layoutlmv3-base-chinese \
--ratio_train 0.9 \
--batch_size ${batch_size} \
--learning_rate ${learning_rate} \
--max_epochs 40 \
--pretrained /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/output/pretrain_lr1e-5_bs_5/epoch_19/checkpoint.pth > logs/${name}_${1}.log 2>&1
