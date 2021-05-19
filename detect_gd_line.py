from argparse import ArgumentParser

import glob, sys, os
import random
import numpy as np
import math
import gc

import torch
import torch.backends.cudnn as cudnn
import models

import cv2
from osgeo import gdal, osr
from natsort import natsorted
import psutil

try:
    from myutils import compute_offsets, LoadImages, load_gt_from_esri_xml
except ImportError:
    print('this script need the gd library, contact zzs.')
    sys.exit(-1)

"""
export PYTHONPATH=/media/ubuntu/Documents/gd/:$PYTHONPATH
python detect_gd_line.py \
--source /media/ubuntu/Data/val_list.txt \
--checkpoint /media/ubuntu/Temp/VesselSeg-Pytorch/gd/gd_line_seg_U_Net_bs=4_lr=0.0001/best_model.pth \ 
--save-dir /media/ubuntu/Temp/VesselSeg-Pytorch/gd/gd_line_seg_U_Net_bs=4_lr=0.0001/large_results/
--img-size 512 --gap 16 --batchsize 4 --device 1
"""


def line_detection(src, is_draw=True):
    # im is a HxW grayscale image
    # lines is [xmin, ymin, xmax, ymax, label] matrix
    lines = []

    # cdst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    cdst = np.stack([src, src, src], axis=-1)
    cdstP = np.copy(cdst)
    lines = cv2.HoughLines(src, 1, np.pi / 180, 150, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(cdst, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(src, 1, np.pi / 180, 200, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)

    return cdstP, lines


def main():
    parser = ArgumentParser()
    parser.add_argument('--network', type=str, default='U_Net', help='network type')
    parser.add_argument('--checkpoint', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--cls-weights', type=str, default='', help='cls_model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--gt-xml-dir', type=str, default='', help='gt xml dir')
    parser.add_argument('--gt-prefix', type=str, default='', help='gt prefix')
    parser.add_argument('--gt-subsize', type=int, default=5120, help='train image size for labeling')
    parser.add_argument('--gt-gap', type=int, default=128, help='train gap size for labeling')
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--big-subsize', type=int, default=51200, help='inference big-subsize (pixels)')
    parser.add_argument('--gap', type=int, default=128, help='overlap size')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--subset', type=str, default='test', help='train, val or test')

    parser.add_argument('--score-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--hw-thres', type=float, default=5, help='height or width threshold for box')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--save-dir', default='./', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--box_prediction_dir', type=str, default='', help='boxes prediction directory')

    args = parser.parse_args()

    source, view_img, save_txt, imgsz, gap, \
    gt_xml_dir, gt_prefix, gt_subsize, gt_gap, \
    big_subsize, batchsize, score_thr, hw_thr = \
        args.source, args.view_img, args.save_txt, args.img_size, args.gap, \
        args.gt_xml_dir, args.gt_prefix, int(args.gt_subsize), int(args.gt_gap), args.big_subsize, \
        args.batchsize, args.score_thres, args.hw_thres
    subset = args.subset
    box_prediction_dir = args.box_prediction_dir

    # Directories
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda:' + args.device)

    if args.network == 'U_Net':
        model = models.UNetFamily.U_Net(img_ch=3, output_ch=2).to(device)
    elif args.network == 'Dense_Unet':
        model = models.UNetFamily.Dense_Unet(in_chan=3, out_chan=2).to(device)
    elif args.network == 'R2AttU_Net':
        model = models.UNetFamily.R2AttU_Net(img_ch=3, output_ch=2).to(device)
    elif args.network == 'AttU_Net':
        model = models.UNetFamily.AttU_Net(img_ch=3, output_ch=2).to(device)
    elif args.network == 'R2U_Net':
        model = models.UNetFamily.R2U_Net(img_ch=3, output_ch=2).to(device)
    elif args.network == 'LadderNet':
        model = models.LadderNet(inplanes=3, num_classes=2, layers=3, filters=16).to(device)
    else:
        print('wrong network type. exit.')
        sys.exit(-1)

    cudnn.benchmark = True

    # Load checkpoint
    print('==> Loading checkpoint...')
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    stride = 32

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape([1, 3, 1, 1])
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape([1, 3, 1, 1])

    for ti in range(3,  4): #len(tiffiles)):
        tiffile = tiffiles[ti]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

        print(ti, '=' * 80)
        print(file_prefix)

        mask_savefilename = save_dir + '/' + subset + "_" + file_prefix + "_LineSeg_result.png"
        # if os.path.exists(mask_savefilename):
        #     continue

        ds = gdal.Open(tiffile, gdal.GA_ReadOnly)
        print("Driver: {}/{}".format(ds.GetDriver().ShortName,
                                     ds.GetDriver().LongName))
        print("Size is {} x {} x {}".format(ds.RasterXSize,
                                            ds.RasterYSize,
                                            ds.RasterCount))
        print("Projection is {}".format(ds.GetProjection()))
        projection = ds.GetProjection()
        projection_sr = osr.SpatialReference(wkt=projection)
        projection_esri = projection_sr.ExportToWkt(["FORMAT=WKT1_ESRI"])
        geotransform = ds.GetGeoTransform()
        xOrigin = geotransform[0]
        yOrigin = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        orig_height, orig_width = ds.RasterYSize, ds.RasterXSize
        if geotransform:
            print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
            print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
            print("IsNorth = ({}, {})".format(geotransform[2], geotransform[4]))

        # 先计算可用内存，如果可以放得下，就不用分块了
        avaialble_mem_bytes = psutil.virtual_memory().available
        if False:  # orig_width * orig_height * ds.RasterCount < 0.8 * avaialble_mem_bytes:
            offsets = [[0, 0, orig_width, orig_height]]
        else:
            # 根据big_subsize计算子块的起始偏移
            offsets = compute_offsets(height=orig_height, width=orig_width, subsize=big_subsize, gap=2 * gt_gap)
        print('offsets: ', offsets)

        final_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)

        for oi, (xoffset, yoffset, sub_width, sub_height) in enumerate(offsets):  # left, up

            print('processing sub image %d' % oi, xoffset, yoffset, sub_width, sub_height)
            dataset = LoadImages(gdal_ds=ds, xoffset=xoffset, yoffset=yoffset,
                                 width=sub_width, height=sub_height,
                                 batchsize=batchsize, subsize=imgsz, gap=gap, stride=stride,
                                 return_list=False, is_nchw=True, return_positions=True)
            if len(dataset) == 0:
                continue

            print('forward inference')
            for idx, (imgs, poss) in enumerate(dataset):
                inputs = torch.from_numpy(imgs.astype(np.float32) / 255).to(device)
                # forward the model
                with torch.no_grad():
                    outputs = model(inputs)
                    results = outputs[:, 1].data.cpu().numpy()

                for (x, y), result in zip(poss, results):
                    if np.any(result):
                        x += xoffset
                        y += yoffset
                        y2 = min(orig_height - 1, y + result.shape[0])
                        x2 = min(orig_width - 1, x + result.shape[1])
                        w = int(x2 - x)
                        h = int(y2 - y)

                        if True:
                            result = result[:h, :w] * 255
                            result = result.astype(np.uint8)
                            result, lines = line_detection(result)    # do line detection
                            if len(result.shape) == 3:
                                final_mask[y:y2, x:x2] = result[:, :, 0]
                            else:
                                final_mask[y:y2, x:x2] = result
                        else:
                            final_mask[y:y2, x:x2] = result[:h, :w] * 255

            del dataset.img0
            del dataset
            gc.collect()

        # draw box predictions
        if box_prediction_dir != '':

            box_preds_xml_filename = box_prediction_dir + '/' + file_prefix + '.xml'
            pred_boxes, pred_labels = load_gt_from_esri_xml(box_preds_xml_filename,
                                                            gdal_trans_info=geotransform)
            if len(pred_boxes) > 0:
                print('num of predicted boxes: ', len(pred_boxes))
                for j, (box, label) in enumerate(zip(pred_boxes, pred_labels)):  # per item
                    xmin, ymin, xmax, ymax = box.astype(np.int32)
                    cv2.rectangle(final_mask, (xmin, ymin), (xmax, ymax), color=(255, 0, 0),
                                  thickness=3)
                    cv2.putText(final_mask, str(label), ((xmin+xmax)//2, (ymin+ymax)//2), fontFace=1, fontScale=1,
                                color=(255, 0, 0), thickness=3)

        # cv2.imwrite(mask_savefilename, mask)
        cv2.imencode('.png', final_mask)[1].tofile(mask_savefilename)

        del final_mask
        del ds


if __name__ == '__main__':
    main()
