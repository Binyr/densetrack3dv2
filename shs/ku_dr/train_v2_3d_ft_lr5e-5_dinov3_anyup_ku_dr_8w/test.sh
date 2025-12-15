source /mnt/shared-storage-user/binyanrui/miniconda3/bin/activate
conda activate densetrack3dv2

cd /mnt/shared-storage-user/binyanrui/Projects/datasets/DenseTrack3Dv2

sudo python scripts/eval/eval_3d.py --ckpt logdirs/densetrack3dv2_ft_lr5e-5_8w_dinov3_anyup_ku_dr/model_densetrack3dv2_final.pth \
    --exp_dir logdirs/densetrack3dv2_ft_lr5e-5_8w_dinov3_anyup_ku_dr/results_pstudio_fix \
    --use_dino dinov3_vitl16