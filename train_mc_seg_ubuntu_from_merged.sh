echo "training ..."
# bash train_mc_seg_ubuntu.sh train 1 0.0001 4 512 U_Net
# bash train_mc_seg_ubuntu.sh train 0 0.001 8 512 SMP_UnetPlusPlus
#
#time1=`date`              # 获取当前时间
#time2=$(date -d "-60 minute ago" +"%Y-%m-%d %H:%M:%S")  # 获取两个小时后的时间
#
#t1=`date -d "$time1" +%s`     # 时间转换成timestamp
#t2=`date -d "$time2" +%s`
#
#echo t1=$t1
#echo t2=$t2
#
#while [ $t1 -lt $t2 ]     # 循环，不断检查是否来到了未来时间
#do
#  echo "wait for 60 seconds .."
#  sleep 60
#  time1=`date`
#  t1=`date -d "$time1" +%s`
#  echo t1=$t1
#done
#
#echo "yes"       # 循环结束，开始执行任务
#echo $time1
#echo $time2
#
#sleep 60

ACTION=$1   # train
GPUID=0   # 1
LR=0.0001   # 0.0001
BS=4   # 4
IMG_SIZE=512   # 512
NETWORK=U_Net    # U_Net
DATA_ROOT=/media/ubuntu/Data/mc_seg/data_from_merged
OUTF=/media/ubuntu/Data/mc_seg/logs_from_merged
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
  --num_workers 2
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
  --test_images_dir $DATA_ROOT/images/val \
  --test_gts_dir $DATA_ROOT/annotations/val
fi