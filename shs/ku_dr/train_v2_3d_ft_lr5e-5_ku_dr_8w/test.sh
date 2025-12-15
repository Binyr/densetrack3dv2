source /mnt/shared-storage-user/binyanrui/miniconda3/bin/activate
conda activate densetrack3dv2

cd /mnt/shared-storage-user/binyanrui/Projects/datasets/DenseTrack3Dv2

python scripts/eval/eval_3d.py --ckpt logdirs/ku_dr/densetrack3dv2_ft_lr5e-5_8w_ku_dr/model_densetrack3dv2_final.pth \
    --exp_dir logdirs/ku_dr/densetrack3dv2_ft_lr5e-5_8w_ku_dr/results_pstudio_fix \