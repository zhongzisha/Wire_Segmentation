echo "training ..."
chcp 65001
set LR=0.0001
@REM set LR=0.001
set BS=4
@REM set BS=8
set IMG_SIZE=512
set NETWORK=U_Net
@REM set NETWORK=SMP_UnetPlusPlus
set DATA_ROOT=E:/Downloads/mc_seg/data_using_shp_multi_random200
set OUTF=E:/Downloads/mc_seg/logs
set SAVE_DIR=%NETWORK%_%IMG_SIZE%_%BS%_%LR%
set TILED_TIFS_DIR=G:/gddata/tiled_tifs
set ACTION=%1%
set EPOCH=%2%

if [%ACTION%] == ["train"] (
python train_mc_seg.py ^
--outf %OUTF% ^
--data_root %DATA_ROOT% ^
--batch_size %BS% ^
--lr %LR% ^
--N_patches 0 ^
--network %NETWORK% ^
--save %SAVE_DIR% ^
--train_subset train ^
--val_subset val ^
--test_subset val ^
--num_classes 4 ^
--img_size %IMG_SIZE% ^
--action train ^
--N_epochs 50 ^
--num_workers 1
)


@REM python train_mc_seg.py ^
@REM --outf %OUTF% ^
@REM --data_root %DATA_ROOT% ^
@REM --batch_size %BS% ^
@REM --lr %LR% ^
@REM --N_patches 0 ^
@REM --network %NETWORK% ^
@REM --save %SAVE_DIR% ^
@REM --train_subset train ^
@REM --val_subset val ^
@REM --test_subset val ^
@REM --num_classes 4 ^
@REM --img_size %IMG_SIZE% ^
@REM --action do_test ^
@REM --pth_filename epoch-%EPOCH%.pth ^
@REM --test_images_dir %DATA_ROOT%/images/val ^
@REM --test_gts_dir %DATA_ROOT%/annotations/val

@REM python train_mc_seg.py ^
@REM --outf %OUTF% ^
@REM --data_root %DATA_ROOT% ^
@REM --batch_size %BS% ^
@REM --lr %LR% ^
@REM --N_patches 0 ^
@REM --network %NETWORK% ^
@REM --save %SAVE_DIR% ^
@REM --train_subset train ^
@REM --val_subset val ^
@REM --test_subset "2-WV03-在建杆塔" ^
@REM --num_classes 4 ^
@REM --img_size %IMG_SIZE% ^
@REM --action do_test ^
@REM --pth_filename epoch-%EPOCH%.pth ^
@REM --test_images_dir "E:/Downloads/mc_seg/tiles/2-WV03-在建杆塔" ^
@REM --test_gts_dir "E:/Downloads/mc_seg/tiles/2-WV03-在建杆塔" ^
@REM --test_image_postfix .tif

if [%ACTION%]==[do_test_tif] (
echo "yes, do test on tif files"

@REM using tif for testing
python train_mc_seg.py ^
--outf %OUTF% ^
--data_root %DATA_ROOT% ^
--batch_size %BS% ^
--lr %LR% ^
--N_patches 0 ^
--network %NETWORK% ^
--save %SAVE_DIR% ^
--train_subset train ^
--val_subset val ^
--test_subset test_tif ^
--num_classes 4 ^
--img_size %IMG_SIZE% ^
--action do_test_tif ^
--pth_filename epoch-%EPOCH%.pth ^
--tiled_tifs_dir %TILED_TIFS_DIR% ^
--test_tifs_dir G:/gddata/all

) else (
echo "no, what's that?"
)