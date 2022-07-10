#!/bin/bash
#SBATCH -J imageNet
#SBATCH -p p-A100
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
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
MODEL='scm'
export CUDA_VISIBLE_DEVICES=${GPU_ID[@]}

# WORK_DIR="/mntnfs/med_data2/haotian/work_dirs/"
# WORK_DIR="/home/baihaotian/programs/TS-CAM/"
# PATH_='ckpt/ImageNet/small/ckpt/model_best.pth'
# # PATH_='ckpt/ImageNet/test_base/ckpt/model_best_epoch_2.pth'
# WORK_DIR=${WORK_DIR}$(echo ${PATH_})

python ./tools_cam/train_cam.py --config_file ./configs/ILSVRC/${NET}_${MODEL}_${NET_SCALE}_patch16_${SIZE}.yaml --lr 1e-6 MODEL.CAM_THR 0.1
# --resume ${WORK_DIR}
