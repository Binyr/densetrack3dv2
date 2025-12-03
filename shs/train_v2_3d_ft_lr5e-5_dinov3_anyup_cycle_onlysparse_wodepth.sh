source /mnt/shared-storage-user/binyanrui/miniconda3/bin/activate
conda activate densetrack3dv2

cd /mnt/shared-storage-user/binyanrui/Projects/datasets/DenseTrack3Dv2

accelerate launch  --num_machines 1 --num_processes 2 --mixed_precision 'no' scripts/train/train_acc_3d.py num_steps=40000 \
    ckpt_path=logdirs/densetrack3dv2_ft_lr5e-5_4w_dinov3_anyup_cycle_onlysparse_traj64_wodepth \
    radius_corr="[3,3,3,7]" stride_corr="[1,1,1,1]" radius_supp="[3,3,3,7]" stride_supp="[1,1,1,1]" \
    use_dino=dinov3_vitl16  dino_size="[24,32]" use_anyup=True  \
    merge_dino_method=corr merge_dino_corr_method="extra" traj_per_sample=64 freeze_dino=True \
    use_dense=False cycle_loss=True \
    restore_ckpt=logdirs/densetrack3dv2_ft_lr5e-5_4w_dinov3_anyup/model_densetrack3dv2_final.pth lr=0.00005

python scripts/eval/eval_3d.py --ckpt logdirs/densetrack3dv2_ft_lr5e-5_4w_dinov3_anyup_cycle_onlysparse_traj64_wodepth/model_densetrack3dv2_final.pth \
    --exp_dir logdirs/densetrack3dv2_ft_lr5e-5_4w_dinov3_anyup_cycle_onlysparse_traj64_wodepth/results \
    --use_dino dinov3_vitl16