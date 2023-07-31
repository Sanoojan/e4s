

CUDA_VISIBLE_DEVICES=4,5 python  -m torch.distributed.launch \
        --nproc_per_node=2 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=22223 \
        scripts/train.py



