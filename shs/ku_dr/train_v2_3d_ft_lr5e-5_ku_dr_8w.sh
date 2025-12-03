source /mnt/shared-storage-user/binyanrui/miniconda3/bin/activate
conda activate densetrack3dv2

cd /mnt/shared-storage-user/binyanrui/Projects/datasets/DenseTrack3Dv2

accelerate launch  --num_machines 1 --num_processes 2 --mixed_precision 'no' scripts/train/train_acc_3d.py num_steps=80000 \
    ckpt_path=logdirs/ku_dr/densetrack3dv2_ft_lr5e-5_8w_ku_dr \
    traj_per_sample=128 \
    restore_ckpt=logdirs/densetrack3dv2/model_densetrack3dv2_final.pth lr=0.00005 \
    use_dr=True

python scripts/eval/eval_3d.py --ckpt logdirs/densetrack3dv2_ft_lr5e-5_4w_dinov3_anyup_ku_dr/model_densetrack3dv2_final.pth \
    --exp_dir logdirs/ku_dr/densetrack3dv2_ft_lr5e-5_4w_ku_dr/results \
    --use_dino dinov3_vitl16