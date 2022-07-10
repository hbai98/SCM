#!/bin/sh		
#BSUB -J Vis_deit
#BSUB -n 4  
#BSUB -m g-node02
#BSUB -q gpu         
#BSUB -gpgpu 1
#BSUB -o out.%J      
#BSUB -e err.%J  
#BSUB -W 48:00

nvidia-smi

module load anaconda3
source activate
conda activate cu113

NET='deit'
NET_SCALE='small'
SIZE='224'
MODEL='scm'
WORK_DIR="/hpc/users/CONNECT/haotianbai/work_dir/scm"
PATH_='/model/resnet/model_best.pth'
WORK_DIR=${WORK_DIR}$(echo ${PATH_})
# --resume ${WORK_DIR}
python ./tools_cam/train_cam.py --config_file ./configs/CUB/SCM_Resnet50.yaml --lr 5e-5 MODEL.CAM_THR 0.1