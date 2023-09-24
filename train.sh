

# CUDA_VISIBLE_DEVICES=0,1 python  -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --nnodes=1 \
#         --node_rank=0 \
#         --master_addr=localhost \
#         --master_port=22223 \
#         scripts/train.py


# python src/utils/create_mask.py
# python scripts/train.py \
#         --exp_dir='running_exp_celeba_diffusion_combined_img_change' \
#         --dataset_name='celeba' \
#         --out_size=1024 


# python scripts/train.py \
#         --exp_dir='running_exp_celeba_diffusion_combined_no_face_parsing' \
#         --dataset_name='celeba' \
#         --id_lambda=0.9 \
#         --face_parsing_lambda=0.0 \
#         --out_size=1024 

# python scripts/train.py \
#         --exp_dir='Experiments/celeba/diffusion/combined_no_face_parsing_reenactment' \
#         --dataset_name='celeba' \
#         --id_lambda=0.9 \
#         --face_parsing_lambda=0.0 \
#         --out_size=1024 


# CUDA_VISIBLE_DEVICES=3 python -W ignore scripts/train.py \
#         --exp_dir='Experiments/celeba/diffusion/DDIM_denoising_added_ours1' \
#         --dataset_name='celeba' \
#         --id_lambda=0.3 \
#         --face_parsing_lambda=0.0 \
#         --out_size=1024 

CUDA_VISIBLE_DEVICES=2 python -W ignore scripts/train.py \
        --exp_dir='Experiments/celeba/diffusion/DDIM_denoising_added_ours2_with_diffusion_loss_id_0.05' \
        --dataset_name='celeba' \
        --id_lambda=0.05 \
        --face_parsing_lambda=0.0 \
        --out_size=1024 