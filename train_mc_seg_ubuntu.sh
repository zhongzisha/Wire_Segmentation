echo "training ..."
# bash train_mc_seg_ubuntu.sh train 1 0.0001 4 512 U_Net
# bash train_mc_seg_ubuntu.sh train 0 0.001 8 512 SMP_UnetPlusPlus


ACTION=$1   # train
GPUID=$2   # 1
LR=$3   # 0.0001
BS=$4   # 4
IMG_SIZE=$5   # 512
NETWORK=$6    # U_Net
DATA_ROOT=/media/ubuntu/Data/mc_seg/data
OUTF=/media/ubuntu/Data/mc_seg/logs
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
  --num_workers 4
fi

if [ "$ACTION" == "do_test" ]; then
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
  --action do_test \
  --pth_filename $OUTF/$SAVE_DIR/epoch-50.pth \
  --test_images_dir $DATA_ROOT%/val/images \
  --test_gts_dir $DATA_ROOT%/val/annotations
fi