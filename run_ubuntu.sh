RUN_TRAIN=$1
GPU_ID=$2
NETWORK=$3
BS=$4
LR=$5

SAVE_DIR=gd_line_seg_${NETWORK}_bs=${BS}_lr=${LR}

if [ $RUN_TRAIN == "1" ]; then

  echo "training ..."
  CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
  --outf /media/ubuntu/Temp/VesselSeg-Pytorch/gd/ \
  --data_root /media/ubuntu/Data/gd_newAug5_Rot0_4classes/refine_line_v1_512_512 \
  --batch_size $BS \
  --lr ${LR} \
  --N_patches 0 \
  --network ${NETWORK} \
  --save ${SAVE_DIR} \
  --dataset_type GdDataset

else

  echo "testing ..."
  CUDA_VISIBLE_DEVICES=$GPU_ID python test.py \
  --outf /media/ubuntu/Temp/VesselSeg-Pytorch/gd/ \
  --data_root /media/ubuntu/Data/gd_newAug5_Rot0_4classes/refine_line_v1_512_512 \
  --batch_size $BS \
  --subset val \
  --network $NETWORK \
  --save ${SAVE_DIR}

fi