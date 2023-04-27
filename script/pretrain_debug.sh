# output_dir=pretrain_lr_1e-4_dtasize_18_batch32

# max_epochs=20

# uper_range=`expr $max_epochs - 1`


# mkdir -p ./data/train/${output_dir}/

# for i in `seq 0 ${uper_range}`
# do
#     mkdir -p ./data/train/${output_dir}/epoch_${i}/
# done

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5


name=debug

nohup python -m torch.distributed.launch --nnodes=1 --master_addr=10.80.205.179 --node_rank=${1}  --nproc_per_node=4   --master_port 29501 \
--use_env ./src/pretrain_wpa.py \
--input_file /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/data_processed_ori/ \
--tokenizer_vocab_dir /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/data/tokenizer/ \
--output_model_dir /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/output/${name}/ \
--model_name microsoft/layoutlmv3-base-chinese \
--ratio_train 0.9 \
--batch_size 12 \
--learning_rate 1e-5 \
--max_epochs 20 \
--pretrained /nlp_group/wuxing/yuhuimu/AB-layoutlmv3/pytorch_model.bin > logs/${name}_${1}.log 2>&1 &
