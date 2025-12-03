source /mnt/shared-storage-user/binyanrui/miniconda3/bin/activate
conda activate densetrack3dv2

cd /mnt/shared-storage-user/binyanrui/Projects/datasets/DenseTrack3Dv2

accelerate launch  --num_machines 1 --num_processes 2 --mixed_precision 'no' scripts/train/train_acc_3d.py num_steps=40000 \
    ckpt_path=logdirs/densetrack3dv2_ft_lr5e-5_4w_dinov3_anyup_cycle_onlysparse_traj64 \
    radius_corr="[3,3,3,7]" stride_corr="[1,1,1,1]" radius_supp="[3,3,3,7]" stride_supp="[1,1,1,1]" \
    use_dino=dinov3_vitl16  dino_size="[24,32]" use_anyup=True  \
    merge_dino_method=corr merge_dino_corr_method="extra" traj_per_sample=64 freeze_dino=True \
    use_dense=False cycle_loss=True \
    restore_ckpt=logdirs/densetrack3dv2_ft_lr5e-5_4w_dinov3_anyup/model_densetrack3dv2_final.pth lr=0.00005

python scripts/eval/eval_3d.py --ckpt logdirs/densetrack3dv2_ft_lr5e-5_4w_dinov3_anyup_cycle_onlysparse_traj64/model_densetrack3dv2_final.pth \
    --exp_dir logdirs/densetrack3dv2_ft_lr5e-5_4w_dinov3_anyup_cycle_onlysparse_traj64/results \
    --use_dino dinov3_vitl16



rjob submit \
    --name=deltav2-gpu2-anyup-dr \
    --gpu=2 \
    --memory=400000 \
    --cpu=32 \
    --namespace ailab-idc3 \
    --charged-group=idc3_gpu \
    --private-machine=group \
    --mount=gpfs://gpfs1/binyanrui:/mnt/shared-storage-user/binyanrui \
    --mount=gpfs://gpfs1/idc2-shared:/mnt/shared-storage-user/idc2-shared \
    --mount=gpfs://gpfs1/si-data:/mnt/shared-storage-user/si-data \
    --mount=gpfs://gpfs1/dongjunting-group:/mnt/shared-storage-user/dongjunting-group \
    --image=registry.h.pjlab.org.cn/ailab-idc2-idc2_gpu/binyanrui:20250902180512 \
    --custom-resources=brainpp.cn/fuse=1 \
    -P 1 \
    -e DISTRIBUTED_JOB=true \
-- bash -exc /mnt/shared-storage-user/binyanrui/Projects/datasets/DenseTrack3Dv2/shs/train_v2_3d_ft_lr5e-5_dinov3_anyup_ku_dr.sh