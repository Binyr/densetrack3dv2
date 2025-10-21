source /mnt/shared-storage-user/binyanrui/miniconda3/bin/activate
conda activate densetrack3dv2

cd /mnt/shared-storage-user/binyanrui/Projects/datasets/DenseTrack3Dv2

accelerate launch  --num_machines 1 --num_processes 4  --mixed_precision 'no' scripts/train/train_acc_3d.py \
    num_steps=50000 \
    ckpt_path=logdirs/densetrack3dv2_freeze_fnet_corr4DCNN_temperal_attn_48x72_ft \
    freeze_fnet=True \
    freeze_corr4DCNN=True \
    freeze_temperal_attn=True \
    dH=48 \
    dW=72 \
    restore_ckpt=checkpoints/densetrack3dv2.pth \
    lr=0.00005
