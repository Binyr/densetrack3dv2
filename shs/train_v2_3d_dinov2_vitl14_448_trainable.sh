source /mnt/shared-storage-user/binyanrui/miniconda3/bin/activate
conda activate densetrack3dv2

cd /mnt/shared-storage-user/binyanrui/Projects/datasets/DenseTrack3Dv2

accelerate launch  --num_machines 1 --num_processes 4  --mixed_precision 'no' scripts/train/train_acc_3d.py \
    num_steps=100000 \
    ckpt_path=logdirs/densetrack3dv2_dinov2_vitl14_448_trainable \
    use_dino=dinov2_vitl14 \
    dino_size=32 \
    freeze_dino=False
