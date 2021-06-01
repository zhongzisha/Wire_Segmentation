from argparse import ArgumentParser

import glob, sys, os, shutil
import random
import numpy as np
import math
import gc
import time
import torch
import torch.backends.cudnn as cudnn
import models

import cv2
from osgeo import gdal, osr
from natsort import natsorted
import psutil

try:
    from myutils import compute_offsets, LoadImages, load_gt_from_esri_xml, LoadMasks
    from myutils import save_predictions_to_envi_xml_and_shp as save_predictions_to_envi_xml
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


def line_detection_bak(src, is_draw=True):
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

    return src, lines


def line_detection(src, xoffset=0, yoffset=0, is_draw=True):
    # im is a HxW grayscale image
    # lines is [xmin, ymin, xmax, ymax, label] matrix
    lines = []

    # cdst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    cdst = np.stack([src, src, src], axis=-1)
    cdstP = np.copy(cdst)
    # lines = cv2.HoughLines(src, 1, np.pi / 180, 150, None, 0, 0)
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
    #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
    #         cv2.line(cdst, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(src, 1, np.pi / 180, 200, None, 50, 10)

    radians = []
    dists = []
    if linesP is not None:
        # for i in range(len(linesP)):
        #     x1, y1, x2, y2 = linesP[i][0]
        #     r = np.arctan((y1 - y2) / (x2 - x1 + 1e-10))
        #     dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        #     radians.append(r)
        #     dists.append(dist)
        #
        # print('radians', radians)
        # print('dists', dists)
        # radian_hist, radian_bin_edges = np.histogram(radians, bins=np.arange(-np.pi / 2, np.pi / 2, 5 * np.pi / 180))
        # print('radian_hist', radian_hist)
        # print('radian_bin_edges', radian_bin_edges)
        # max_hist = np.argmax(radian_hist)
        # max_radian = radian_bin_edges[max_hist]
        # print('max_hist', max_hist)
        # print('max_radian', max_radian)

        for i in range(0, len(linesP)):
            x1, y1, x2, y2 = linesP[i][0]

            dis = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dis < 100:
                continue

            cv2.line(cdstP, (x1, y1), (x2, y2), (255, 255, 255), 3, cv2.LINE_AA)
            lines.append([x1 + xoffset, y1 + yoffset,
                          x2 + xoffset, y2 + yoffset])

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
    parser.add_argument('--is_save_patches', action='store_true', help='if save image patches for debug')

    args = parser.parse_args()

    source, view_img, save_txt, imgsz, gap, \
    gt_xml_dir, gt_prefix, gt_subsize, gt_gap, \
    big_subsize, batchsize, score_thr, hw_thr = \
        args.source, args.view_img, args.save_txt, args.img_size, args.gap, \
        args.gt_xml_dir, args.gt_prefix, int(args.gt_subsize), int(args.gt_gap), args.big_subsize, \
        args.batchsize, args.score_thres, args.hw_thres
    subset = args.subset
    box_prediction_dir = args.box_prediction_dir
    is_save_patches = args.is_save_patches

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

    for ti in range(5, len(tiffiles)):
        tiffile = tiffiles[ti]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

        print(ti, '=' * 80)
        print(file_prefix)

        mask_savefilename = os.path.join(save_dir, subset + "_" + file_prefix + "_LineSeg_result.png")
        # if os.path.exists(mask_savefilename):
        #     continue

        if is_save_patches:
            patches_save_dir = os.path.join(save_dir, file_prefix)
            print('patches_save_dir', patches_save_dir)
            if os.path.exists(patches_save_dir):
                shutil.rmtree(patches_save_dir, ignore_errors=True)
            os.makedirs(patches_save_dir)

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

        final_mask = None
        if not os.path.exists(mask_savefilename):
            # dir_path = os.path.dirname(os.path.abspath(mask_savefilename))
            # tmp_filename = os.path.join(dir_path, 'tmp.png')
            # shutil.move(mask_savefilename, tmp_filename)
            # final_mask = cv2.imread(tmp_filename, 0)
            # shutil.move(tmp_filename, mask_savefilename)

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

                            result = result[:h, :w] * 255
                            result = result.astype(np.uint8)
                            final_mask[y:y2, x:x2] = result * 255

                del dataset.img0
                del dataset
                gc.collect()
            # cv2.imwrite(mask_savefilename, mask)
            cv2.imencode('.png', final_mask)[1].tofile(mask_savefilename)

        del ds

        time.sleep(3)

        mask_ds = gdal.Open(mask_savefilename, gdal.GA_ReadOnly)
        # conduct line detection in final_mask
        imgsz_mask = 5120
        final_mask2 = np.zeros((orig_height, orig_width), dtype=np.uint8)
        all_lines = []
        for oi, (xoffset, yoffset, sub_width, sub_height) in enumerate(offsets):  # left, up

            print('processing sub mask %d' % oi, xoffset, yoffset, sub_width, sub_height)
            dataset = LoadMasks(gdal_ds=mask_ds, xoffset=xoffset, yoffset=yoffset,
                                width=sub_width, height=sub_height,
                                batchsize=batchsize, subsize=imgsz_mask, gap=gap, stride=stride,
                                return_list=False, is_nchw=True, return_positions=True)
            if len(dataset) == 0:
                continue

            print('forward inference')
            for idx, (imgs, poss) in enumerate(dataset):
                for (x, y), result in zip(poss, imgs):
                    if np.any(result):
                        x += xoffset
                        y += yoffset
                        y2 = min(orig_height - 1, y + result.shape[0])
                        x2 = min(orig_width - 1, x + result.shape[1])
                        w = int(x2 - x)
                        h = int(y2 - y)

                        result = result[:h, :w]

                        if is_save_patches:
                            cv2.imwrite(os.path.join(patches_save_dir, "patch_%05d_%d_%d_before.png" % (idx, x, y)),
                                        result)

                        result, lines = line_detection(result, x, y)  # do line detection  lines:[x1,y1,x2,y2]
                        all_lines += lines

                        if is_save_patches:
                            cv2.imwrite(os.path.join(patches_save_dir, "patch_%05d_%d_%d_after.png" % (idx, x, y)),
                                        result)

                        if len(result.shape) == 3:
                            final_mask2[y:y2, x:x2] = result[:, :, 0]
                        else:
                            final_mask2[y:y2, x:x2] = result

            del dataset.img0
            del dataset
            gc.collect()

        del mask_ds
        cv2.imencode('.png', final_mask2)[1].tofile(mask_savefilename.replace('_LineSeg_result', '_LineSeg_result2'))
        del final_mask2

        if len(all_lines) > 0:
            radians = []
            dists = []
            valid_lines = []
            for x1, y1, x2, y2 in all_lines:
                radian = np.arctan((y1 - y2) / (x2 - x1 + 1e-10))
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                radians.append(radian)
                dists.append(dist)

                if dist > 50:
                    valid_lines.append([x1, y1, x2, y2])

            # print('radians', radians)
            # print('dists', dists)
            radian_hist, radian_bin_edges = np.histogram(radians,
                                                         bins=np.arange(-np.pi / 2, np.pi / 2, 5 * np.pi / 180))
            print('radian_hist', radian_hist)
            print('radian_bin_edges', radian_bin_edges)
            max_hist = np.argmax(radian_hist)
            max_radian = radian_bin_edges[max_hist]
            print('max_hist', max_hist)
            print('max_radian', max_radian)

            if len(valid_lines) > 0:
                all_preds = np.concatenate([np.array(valid_lines).reshape([-1, 4]),
                                            np.ones((len(valid_lines), 2), dtype=np.float32)], axis=1)

                save_predictions_to_envi_xml(preds=all_preds,
                                             save_xml_filename=os.path.join(save_dir, file_prefix + '.xml'),
                                             gdal_proj_info=projection_esri,
                                             gdal_trans_info=geotransform,
                                             names={1: "Line"},
                                             colors={1: [255, 0, 0]},
                                             is_line=True,
                                             spatialreference=projection_sr,
                                             is_save_xml=False)

        if final_mask is not None:
            # draw box predictions
            if box_prediction_dir != '':

                box_preds_xml_filename = os.path.join(box_prediction_dir, file_prefix + '.xml')
                pred_boxes, pred_labels, pred_scores = load_gt_from_esri_xml(box_preds_xml_filename,
                                                                             gdal_trans_info=geotransform,
                                                                             has_scores=True)
                print(len(pred_boxes), len(pred_labels), len(pred_scores))
                if len(pred_boxes) > 0:
                    print('num of predicted boxes: ', len(pred_boxes))
                    for j, (box, label) in enumerate(zip(pred_boxes, pred_labels)):  # per item
                        xmin, ymin, xmax, ymax = box.astype(np.int32)
                        cv2.rectangle(final_mask, (xmin, ymin), (xmax, ymax), color=(255, 0, 0),
                                      thickness=3)
                        cv2.putText(final_mask, str(label), ((xmin + xmax) // 2, (ymin + ymax) // 2), fontFace=1,
                                    fontScale=1,
                                    color=(255, 0, 0), thickness=3)

            # cv2.imwrite(mask_savefilename, mask)
            cv2.imencode('.png', final_mask)[1].tofile(
                mask_savefilename.replace('_LineSeg_result', '_LineSeg_result_withBoxes'))
            del final_mask


if __name__ == '__main__':
    main()
