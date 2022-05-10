#!/bin/bash
##SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --account=owner-guest
#SBATCH --partition=q04
##SBATCH --gres=gpu:3
#SBATCH --nodelist=gg02
#SBATCH --gres=gpu:1
##SBATCH --mem-per-gpu=30G
##SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH -o /share/home/zhongzisha/cluster_logs/mmdet-towers-%j-%N.out
#SBATCH -e /share/home/zhongzisha/cluster_logs/mmdet-towers-%j-%N.err

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

sleep 500000000000000000


source /share/home/zhongzisha/venv_test/bin/activate

export LD_LIBRARY_PATH=$HOME/gcc-7.5.0/install/lib64:$LD_LIBRARY_PATH
export PATH=$HOME/gcc-7.5.0/install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/glibc2.30/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
# export CUDA_ROOT=$HOME/cuda-10.2-cudnn-7.6.5
# export LD_LIBRARY_PATH=$CUDA_ROOT/libs/lib64:$CUDA_ROOT/lib64:$CUDA_ROOT/lib64/stubs:$LD_LIBRARY_PATH
export CUDA_ROOT=$HOME/cuda-10.2-cudnn-8.2.2
export CUDA_PATH=$CUDA_ROOT
export LD_LIBRARY_PATH=$CUDA_ROOT/libs/lib64:$CUDA_ROOT/lib64:$CUDA_ROOT/lib64/stubs:$LD_LIBRARY_PATH
export PATH=$CUDA_ROOT/bin:$PATH
export CUDA_INSTALL_DIR=$HOME/cuda-10.2-cudnn-8.2.2
export CUDNN_INSTALL_DIR=$HOME/cuda-10.2-cudnn-8.2.2
export TRT_LIB_DIR=$HOME/cuda-10.2-cudnn-8.2.2/TensorRT-8.0.1.6/lib


if [ ${HOSTNAME} == "gg02" ]; then

ACTION=train   # train
GPUID=1   # 1
LR=0.001   # 0.0001
BS=8   # 4
IMG_SIZE=512   # 512
NETWORK=SMP_UnetPlusPlus    # U_Net
DATA_ROOT= `pwd`/data
OUTF=`pwd`/logs
SAVE_DIR=${NETWORK}_${IMG_SIZE}_${BS}_${LR}

if [ "$ACTION" == "train" ]; then
  CUDA_VISIBLE_DEVICES=$GPUID python train_mc_seg.py \
  --outf $OUTF \
  --data_root $DATA_ROOT \
  --batch_size $BS \
  --lr $LR \
  --N_patches 0 \
  --network $NETWORK \
  --save $SAVE_DIR \
  --train_subset train \
  --val_subset val \
  --test_subset val \
  --num_classes 4 \
  --img_size $IMG_SIZE \
  --action ${ACTION} \
  --N_epochs 50 \
  --num_workers 4 \
  --cached True
fi

fi



