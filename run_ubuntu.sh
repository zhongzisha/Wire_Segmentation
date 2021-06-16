RUN_TYPE=$1   # train, val, test_big
GPU_ID=$2      # 0 or 1
NETWORK=$3     # U_Net or LadderNet
BS=$4          # 4, 8, ...
LR=$5          # 0.001 or ...
MODELTYPE=$6   # best or latest

SAVE_DIR=gd_line_seg_${NETWORK}_bs=${BS}_lr=${LR}_withSat

if [ ${HOSTNAME} == 'master' ]; then
LINE_REFINE_SEG_TMP_DIR=/media/ubuntu/Temp/VesselSeg-Pytorch/gd/
LINE_REFINE_SEG_DATA_ROOT=/media/ubuntu/Temp/gd_newAug5_Rot0_4classes/refine_line_v1_512_512
DATASET_TYPE=GdDataset
fi

echo ${SAVE_DIR}
echo ${LINE_REFINE_SEG_TMP_DIR}
echo ${LINE_REFINE_SEG_DATA_ROOT}
echo "ok"

if [ $RUN_TYPE == "train" ]; then

  echo "training ..."
  CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
  --outf ${LINE_REFINE_SEG_TMP_DIR} \
  --data_root ${LINE_REFINE_SEG_DATA_ROOT} \
  --batch_size $BS \
  --lr ${LR} \
  --N_patches 0 \
  --network ${NETWORK} \
  --save ${SAVE_DIR} \
  --dataset_type ${DATASET_TYPE} \
  --train_subset train1 \
  --val_subset val1 \
  --test_subset val1

elif [ $RUN_TYPE == "val" ]; then

  echo "validating ..."
  CUDA_VISIBLE_DEVICES=$GPU_ID python test.py \
  --outf ${LINE_REFINE_SEG_TMP_DIR} \
  --data_root ${LINE_REFINE_SEG_DATA_ROOT} \
  --batch_size $BS \
  --subset val \
  --network $NETWORK \
  --save ${SAVE_DIR} \
  --train_subset train1 \
  --val_subset val1 \
  --test_subset val1

elif [ $RUN_TYPE == "test" ]; then

  echo "testing ..."
  CUDA_VISIBLE_DEVICES=$GPU_ID python test.py \
  --outf ${LINE_REFINE_SEG_TMP_DIR} \
  --data_root ${LINE_REFINE_SEG_DATA_ROOT} \
  --batch_size 4 \
  --subset test2 \
  --network $NETWORK \
  --save ${SAVE_DIR}

elif [ $RUN_TYPE == "test_big" ] && [ $HOSTNAME == "master" ]; then

  echo "testing in big images ..."
  python detect_gd_line.py \
  --network ${NETWORK} \
  --source /media/ubuntu/Data/val_list.txt \
  --checkpoint ${LINE_REFINE_SEG_TMP_DIR}/${SAVE_DIR}/${MODELTYPE}_model.pth \
  --save-dir ${LINE_REFINE_SEG_TMP_DIR}/${SAVE_DIR}/${MODELTYPE}_big_results_Boxes0/ \
  --img-size 512 --gap 16 --batchsize 6 --device $GPU_ID

elif [ $RUN_TYPE == "test_big_with_boxes" ] && [ $HOSTNAME == "master" ]; then

  if [ ${NETWORK} == "U_Net" ]; then
    TEST_BS=12
  elif [ ${NETWORK} == "Dense_Unet" ]; then
    TEST_BS=4
  fi

  echo "testing in big images ..."
  python detect_gd_line.py \
  --network ${NETWORK} \
  --source /media/ubuntu/Data/val_list.txt \
  --checkpoint ${LINE_REFINE_SEG_TMP_DIR}/${SAVE_DIR}/${MODELTYPE}_model.pth \
  --save-dir ${LINE_REFINE_SEG_TMP_DIR}/${SAVE_DIR}/${MODELTYPE}_big_results_Boxes1/ \
  --img-size 512 --gap 8 --batchsize ${TEST_BS} --device $GPU_ID \
  --box_prediction_dir /media/ubuntu/Temp/gd/mmdetection/faster_rcnn_r50_fpn_dc5_2x_coco_lr0.001_newAug3_v2/outputs_val_1024_256_epoch_17 \
  --is_save_patches

elif [ $RUN_TYPE == "test_dataset" ]; then
  echo "test dataset ..."

fi