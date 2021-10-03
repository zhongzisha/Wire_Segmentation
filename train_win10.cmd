echo "training ..."
set LR=0.001
set BS=2
set IMG_SIZE=512
set NETWORK=U_Net
set DATA_ROOT=E:/line_foreign_object_detection/augmented_data
set OUTF=E:/line_foreign_object_detection/logs
set SAVE_DIR=%NETWORK%_%IMG_SIZE%_%BS%_%LR%

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
--action train ^
--N_epochs 20

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
--pth_filename epoch-50.pth ^
--test_images_dir %DATA_ROOT%/val/images ^
--test_gts_dir %DATA_ROOT%/val/annotations_with_foreign