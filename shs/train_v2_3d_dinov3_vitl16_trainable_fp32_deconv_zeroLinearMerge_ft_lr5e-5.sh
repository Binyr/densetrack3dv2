source /mnt/shared-storage-user/binyanrui/miniconda3/bin/activate
conda activate densetrack3dv2

cd /mnt/shared-storage-user/binyanrui/Projects/datasets/DenseTrack3Dv2

accelerate launch  --num_machines 1 --num_processes 2 --mixed_precision 'no' scripts/train/train_acc_3d.py num_steps=40000 \
    ckpt_path=logdirs/densetrack3dv2_dinov3_vitl16_trainable_fp32_deconv_zeroLinearMerge_ft_lr5e-5_4w \
    use_dino=dinov3_vitl16  dino_size=32 freeze_dino=False use_deconv=True use_merge_conv=True \
    restore_ckpt=logdirs/densetrack3dv2/model_densetrack3dv2_final.pth lr=0.00005

