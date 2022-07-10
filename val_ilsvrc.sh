#!/bin/bash
#SBATCH -J fViT
#SBATCH -p p-V100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12



cd /home/baihaotian/programs/TS-CAM
source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
# conda activate cu102
conda activate cu113

NET='deit'
NET_SCALE='small'
SIZE='224'
MODEL='scm'

WORK_DIR="/hpc/users/CONNECT/haotianbai/work_dir/SCM"
PATH_='ckpt/ImageNet/small/ckpt/model_best.pth'
WORK_DIR=${WORK_DIR}$(echo ${PATH_})

python ./tools_cam/test_cam.py --config_file configs/ILSVRC/${NET}_${MODEL}_${NET_SCALE}_patch16_${SIZE}.yaml --resume ${WORK_DIR} TEST.SAVE_BOXED_IMAGE True 
# PATH_='ckpt/CUB/model/deit_tiny/ckpt/model_best_top1_loc.pth'
# PATH_='ckpt/CUB/model/conformer_small/ckpt/model_best_top1_loc.pth'
# PATH_='ckpt/CUB/cls_repre/ly4/ckpt/model_best_top1_loc.pth'
# PATH_='ckpt/ImageNet/ly4/deit_small/ckpt/model_best.pth'