RUN_TRAIN=$1   # 0 or 1
GPU_ID=$2      # 0 or 1
NETWORK=$3     # U_Net or LadderNet
BS=$4          # 4, 8, ...
LR=$5          # 0.001 or ...

SAVE_DIR=gd_line_seg_${NETWORK}_bs=${BS}_lr=${LR}

if [ ${HOSTNAME} == 'master' ]; then
LINE_REFINE_SEG_TMP_DIR=/media/ubuntu/Temp/VesselSeg-Pytorch/gd/
LINE_REFINE_SEG_DATA_ROOT=/media/ubuntu/Data/gd_newAug5_Rot0_4classes/refine_line_v1_512_512
DATASET_TYPE=GdDataset
fi

echo ${SAVE_DIR}
echo ${LINE_REFINE_SEG_TMP_DIR}
echo ${LINE_REFINE_SEG_DATA_ROOT}
echo "ok"

if [ $RUN_TRAIN == "1" ]; then

  echo "training ..."
  CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
  --outf ${LINE_REFINE_SEG_TMP_DIR} \
  --data_root ${LINE_REFINE_SEG_DATA_ROOT} \
  --batch_size $BS \
  --lr ${LR} \
  --N_patches 0 \
  --network ${NETWORK} \
  --save ${SAVE_DIR} \
  --dataset_type ${DATASET_TYPE}

else

  echo "testing ..."
  CUDA_VISIBLE_DEVICES=$GPU_ID python test.py \
  --outf ${LINE_REFINE_SEG_TMP_DIR} \
  --data_root ${LINE_REFINE_SEG_DATA_ROOT} \
  --batch_size $BS \
  --subset val \
  --network $NETWORK \
  --save ${SAVE_DIR}

fi