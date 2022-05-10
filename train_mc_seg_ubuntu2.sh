echo "training ..."
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


GPUID=1
LR=0.001
BS=8
IMG_SIZE=512
NETWORK=SMP_UnetPlusPlus
DATA_ROOT=/media/ubuntu/Data/mc_seg/data
OUTF=/media/ubuntu/Data/mc_seg/logs
SAVE_DIR=${NETWORK}_${IMG_SIZE}_${BS}_${LR}

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
--action train \
--N_epochs 50 \
--num_workers 2

#CUDA_VISIBLE_DEVICES=$GPUID python train_mc_seg.py \
#--outf $OUTF \
#--data_root $DATA_ROOT \
#--batch_size $BS \
#--lr $LR \
#--N_patches 0 \
#--network $NETWORK \
#--save $SAVE_DIR \
#--train_subset train \
#--val_subset val \
#--test_subset val \
#--num_classes 4 \
#--img_size $IMG_SIZE \
#--action do_test \
#--pth_filename epoch-50.pth \
#--test_images_dir $DATA_ROOT%/val/images \
#--test_gts_dir $DATA_ROOT%/val/annotations