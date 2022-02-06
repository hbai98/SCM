#!/bin/bash
#SBATCH -J imageNet
#SBATCH -p p-A100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6

cd /home/baihaotian/programs/TS-CAM
source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate cu113

module load cuda11.2/toolkit/11.2.2
module load cuda11.2/fft/11.2.2
module load cuda11.2/blas/11.2.2

GPU_ID=(0,1)
NET='deit'
NET_SCALE='small'
SIZE='224'
export CUDA_VISIBLE_DEVICES=${GPU_ID[@]}

python ./tools_cam/train_cam.py --config_file ./configs/ILSVRC/${NET}_tscam_${NET_SCALE}_patch16_${SIZE}.yaml --lr 5e-4 MODEL.CAM_THR 0.1