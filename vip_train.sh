data_path="$1" # "/root/data-repo/deepfashion"
if [ -z "$data_path" ];then
    echo "no data"    
    exit -1
fi
python -m torch.distributed.launch --nproc_per_node=8 --master_port=30107 train.py \
    --dataset_path  "$data_path"\
    --batch_size 16 \
    --exp_name "pidm_deepfashion"