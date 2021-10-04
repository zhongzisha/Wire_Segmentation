echo "training ..."
set LR=0.001
set BS=8
set IMG_SIZE=512
set NETWORK=SMP_UnetPlusPlus
set DATA_ROOT=E:/Downloads/mc_seg/data
set OUTF=E:/Downloads/mc_seg/logs
set SAVE_DIR=%NETWORK%_%IMG_SIZE%_%BS%_%LR%

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
@REM --action train ^
@REM --N_epochs 20

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
--action do_test ^
--pth_filename epoch-20.pth ^
--test_images_dir %DATA_ROOT%/images/val ^
--test_gts_dir %DATA_ROOT%/annotations/val