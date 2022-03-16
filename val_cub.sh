#!/bin/bash
#SBATCH -J fViT
#SBATCH -p p-RTX2080
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

cd /home/baihaotian/programs/TS-CAM
source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
# conda activate cu102
conda activate cu113

module load cuda11.2/toolkit/11.2.2
module load cuda11.2/fft/11.2.2
module load cuda11.2/blas/11.2.2

GPU_ID=0
NET='deit'
NET_SCALE='small'
SIZE='224'
MODEL='fcam'
export CUDA_VISIBLE_DEVICES=${GPU_ID}
# PATH_='ckpt/CUB/test_norm/ckpt/model_best.pth'
# PATH_='ckpt/CUB/cls_repre/ly3/ckpt/model_best.pth'
# PATH_='ckpt/CUB/cls_repre/ly8/ckpt/model_best.pth'
WORK_DIR="/mntnfs/med_data2/haotian/work_dirs/"
PATH_='ckpt/CUB/cubv2_scm_v2/ckpt/model_best.pth'
# WORK_DIR='/home/baihaotian/programs/TS-CAM/'
# PATH_='pretrained/tscam_cub.pth'
WORK_DIR=${WORK_DIR}$(echo ${PATH_})


python ./tools_cam/test_cam.py --config_file ./configs/CUB/${NET}_${MODEL}_${NET_SCALE}_patch16_${SIZE}.yaml --resume ${WORK_DIR} TEST.SAVE_BOXED_IMAGE True MODEL.CAM_THR 0.4