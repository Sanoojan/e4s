# CUDA_VISIBLE_DEVICES=3 python scripts/face_swap.py --source=example/input/faceswap/peng/peng_800.jpg  --target=example/input/faceswap/andy.jpg \
#                             --verbose=True \
#                             --checkpoint_path='/home/sb1/sanoojan/e4s/running_exp_celeba_diffusion/checkpoints/iteration_200000.pt' 


CUDA_VISIBLE_DEVICES=3 python scripts/face_swap.py \
    --source=example/input/faceswap/peng/peng_800.jpg \
    --target=example/input/faceswap/andy.jpg \
    --verbose=True \
    --checkpoint_path='/home/sb1/sanoojan/e4s/running_exp_celeba_diffusion/checkpoints/iteration_200000.pt' 
