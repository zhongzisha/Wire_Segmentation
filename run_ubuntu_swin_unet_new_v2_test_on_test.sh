RUN_TYPE=do_test   # train, val, test_big
GPU_ID=1      # 0 or 1
NETWORK=Swin_Unet_V2     # U_Net or LadderNet
BS=24          # 4, 8, ...
LR=0.2          # 0.001 or ...
MODELTYPE=latest   # best or latest
SWIN_BACBONE=tiny   # tiny,small,base
IMG_SIZE=224    # 224
PATCH_SIZE=4    # 4

# swin-unet-v4: swin_${SWIN_BACBONE}_patch4_window7_224_lite_v4.yaml
# others: swin_${SWIN_BACBONE}_patch4_window7_224_lite_debug1.yaml
CFG_FILE=swin_${SWIN_BACBONE}_patch4_window7_224_lite_v4.yaml
CFG_FILE=swin_${SWIN_BACBONE}_patch4_window7_224_lite.yaml
# CFG_FILE=swin_${SWIN_BACBONE}_patch4_window7_224_lite_v1_3.yaml
# CFG_FILE=swin_${SWIN_BACBONE}_patch4_window7_224_lite_v1_4.yaml
POSTFIX=

SAVE_DIR=gd_lineseg_new_${NETWORK}_bs=${BS}_lr=${LR}_${SWIN_BACBONE}_${IMG_SIZE}_${PATCH_SIZE}_${POSTFIX}


if [ ${HOSTNAME} == 'master' ] || [ ${HOSTNAME} == 'slave' ]; then
LINE_REFINE_SEG_TMP_DIR=/media/ubuntu/Data/VesselSeg-Pytorch/gd/
LINE_REFINE_SEG_DATA_ROOT=/media/ubuntu/Data/gd_newAug5_Rot0_4classes_bak/refine_line_v1_512_512
DATASET_TYPE=GdDataset
fi

echo ${SAVE_DIR}
echo ${LINE_REFINE_SEG_TMP_DIR}
echo ${LINE_REFINE_SEG_DATA_ROOT}
echo "ok"

if [ $RUN_TYPE == "train" ]; then

  echo "training ..."
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_new.py \
  --outf ${LINE_REFINE_SEG_TMP_DIR} \
  --data_root ${LINE_REFINE_SEG_DATA_ROOT} \
  --batch_size $BS \
  --lr ${LR} \
  --N_patches 0 \
  --network ${NETWORK} \
  --save ${SAVE_DIR} \
  --dataset_type ${DATASET_TYPE} \
  --train_subset train \
  --val_subset val \
  --test_subset val \
  --cfg ${CFG_FILE} \
  --num_classes 2 \
  --img_size ${IMG_SIZE} \
  --patch_size ${PATCH_SIZE}

elif [ $RUN_TYPE == "do_test" ]; then

  echo "validating ..."
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_new.py \
  --outf ${LINE_REFINE_SEG_TMP_DIR} \
  --data_root ${LINE_REFINE_SEG_DATA_ROOT} \
  --batch_size $BS \
  --lr ${LR} \
  --N_patches 0 \
  --network ${NETWORK} \
  --save ${SAVE_DIR} \
  --dataset_type ${DATASET_TYPE} \
  --train_subset train \
  --val_subset val \
  --test_subset test \
  --cfg ${CFG_FILE} \
  --num_classes 2 \
  --action do_test \
  --test_images_dir /media/ubuntu/Data/gd_newAug5_Rot0_4classes_bak/refine_line_v1_512_512/test/images \
  --test_gts_dir /media/ubuntu/Data/gd_newAug5_Rot0_4classes_bak/refine_line_v1_512_512/test/annotations \
  --pth_filename epoch-50.pth

fi