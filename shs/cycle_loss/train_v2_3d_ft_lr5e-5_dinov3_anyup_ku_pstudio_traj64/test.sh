source /mnt/shared-storage-user/binyanrui/miniconda3/bin/activate
conda activate densetrack3dv2

cd /mnt/shared-storage-user/binyanrui/Projects/datasets/DenseTrack3Dv2

sudo python scripts/eval/eval_3d.py --ckpt logdirs/cycle_loss/densetrack3dv2_ft_lr5e-5_dinov3_anyup_ku_pstudio_traj64_2/model_densetrack3dv2_final.pth \
    --exp_dir logdirs/cycle_loss/densetrack3dv2_ft_lr5e-5_dinov3_anyup_ku_pstudio_traj64_2/results_pstudio_fix \
    --use_dino dinov3_vitl16