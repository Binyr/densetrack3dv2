source /mnt/shared-storage-user/binyanrui/miniconda3/bin/activate
conda activate densetrack3dv2

cd /mnt/shared-storage-user/binyanrui/Projects/datasets/DenseTrack3Dv2

sudo python scripts/eval/eval_3d.py --ckpt logdirs/cycle_loss/densetrack3dv2_ft_lr5e-5_ku_stereo4d_traj64_1vs10/model_densetrack3dv2_final.pth \
    --exp_dir logdirs/cycle_loss/densetrack3dv2_ft_lr5e-5_ku_stereo4d_traj64_1vs10/results_pstudio_fix \