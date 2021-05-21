
set NETWORK=U_Net
set MODEL_DIR=E:/VesselSeg-Pytorch/gd_line_seg_U_Net_bs=4_lr=0.0001
@REM set NETWORK=Dense_Unet
@REM set MODEL_DIR=E:/Vessel-Pytorch/gd_line_seg_Dense_Unet_bs=4_lr=0.0001
python detect_gd_line.py ^
  --network %NETWORK% ^
  --source E:/val_list.txt ^
  --checkpoint %MODEL_DIR%/best_model.pth ^
  --save-dir %MODEL_DIR%/big_results_noBoxes/ ^
  --img-size 512 --gap 16 --batchsize 4 --device 0
@REM   --box_prediction_dir E:/mmdetection/work_dirs/faster_rcnn_r50_fpn_dc5_2x_coco_lr0.001_newAug3_v2/outputs_val_1024_256_epoch_17











