echo "training ..."
set LR=0.001
set BS=2
set IMG_SIZE=512
set NETWORK=U_Net
set DATA_ROOT=E:/line_foreign_object_detection/augmented_data_v2
set OUTF=E:/line_foreign_object_detection/logs_v2
set SAVE_DIR=%NETWORK%_%IMG_SIZE%_%BS%_%LR%
set TILED_TIFS_DIR=E:/generated_big_test_images/tiled_tifs
set ACTION=%1%
set EPOCH=%2%

if [%ACTION%] == ["train"] (

@REM python train_line_foreign.py ^
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
@REM --num_classes 3 ^
@REM --img_size %IMG_SIZE% ^
@REM --action train ^
@REM --N_epochs 20

python train_line_foreign.py ^
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
--num_classes 3 ^
--img_size %IMG_SIZE% ^
--action do_test ^
--pth_filename epoch-20.pth ^
--test_images_dir %DATA_ROOT%/val/images ^
--test_gts_dir %DATA_ROOT%/val/annotations_with_foreign
)

if [%ACTION%]==[do_test_tif_line_foreign] (
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
--num_classes 3 ^
--img_size %IMG_SIZE% ^
--action do_test_tif_line_foreign ^
--pth_filename epoch-%EPOCH%.pth ^
--tiled_tifs_dir %TILED_TIFS_DIR% ^
--test_tifs_dir E:/generated_big_test_images/line_foreign ^
--min_blob_size 1

) else (
echo "no, what's that?"
)