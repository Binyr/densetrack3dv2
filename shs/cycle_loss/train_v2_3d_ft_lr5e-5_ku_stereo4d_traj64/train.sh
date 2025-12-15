source /mnt/shared-storage-user/binyanrui/miniconda3/bin/activate
conda activate densetrack3dv2

cd /mnt/shared-storage-user/binyanrui/Projects/datasets/DenseTrack3Dv2

accelerate launch  --num_machines 1 --num_processes 2 --mixed_precision 'no' scripts/train/train_acc_3d.py num_steps=40000 \
    ckpt_path=logdirs/cycle_loss/densetrack3dv2_ft_lr5e-5_ku_stereo4d_traj64 \
    traj_per_sample=64 \
    restore_ckpt=logdirs/densetrack3dv2_ft_lr5e-5_traj128/model_densetrack3dv2_final.pth lr=0.00005 \
    use_stereo4d=True dataset_repeats="[20,1]" \

sudo python scripts/eval/eval_3d.py --ckpt logdirs/cycle_loss/densetrack3dv2_ft_lr5e-5_ku_stereo4d_traj64/model_densetrack3dv2_final.pth \
    --exp_dir logdirs/cycle_loss/densetrack3dv2_ft_lr5e-5_ku_stereo4d_traj64/results_pstudio_fix \