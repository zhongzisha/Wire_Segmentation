#!/bin/bash
##SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --account=owner-guest
#SBATCH --partition=q04
##SBATCH --gres=gpu:3
#SBATCH --nodelist=g39
#SBATCH --gres=gpu:1
##SBATCH --mem-per-gpu=30G
##SBATCH --cpus-per-task=2
##SBATCH --mem=100G
#SBATCH -o /share/home/zhongzisha/cluster_logs/LineSeg-job-train-%j-%N.out
#SBATCH -e /share/home/zhongzisha/cluster_logs/LineSeg-job-train-%j-%N.err

echo "job start `date`"
echo "job run at ${HOSTNAME}"
nvidia-smi

df -h
nvidia-smi
ls /usr/local
which nvcc
which gcc
which g++
nvcc --version
gcc --version
g++ --version

env

nvidia-smi

free -g
top -b -n 1

uname -a

source $HOME/anaconda3_py38/bin/activate   # here, opencv-4.5.2

export LINE_REFINE_SEG_TMP_DIR=$HOME/gd_line_seg/
export LINE_REFINE_SEG_DATA_ROOT=$HOME/refine_line_v1_512_512
export DATASET_TYPE=memory
GPU_IDS=0
#sleep 60

#bash run_ubuntu.sh train ${GPU_IDS} U_Net 8 0.0001
#bash run_ubuntu.sh train ${GPU_IDS} Dense_Unet 4 0.0001
bash run_ubuntu.sh train ${GPU_IDS} R2AttU_Net 4 0.0005
#bash run_ubuntu.sh train ${GPU_IDS} AttU_Net 8 0.0001   # g42 is OK
# bash run_ubuntu.sh train ${GPU_IDS} R2U_Net 8 0.0001
#bash run_ubuntu.sh train ${GPU_IDS} LadderNet 8 0.0001

