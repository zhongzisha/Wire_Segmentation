from argparse import ArgumentParser

import glob,sys,os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import models

import cv2
from osgeo import gdal, osr
from natsort import natsorted
import psutil
from myutils import compute_offsets, LoadImages


"""
export PYTHONPATH=/media/ubuntu/Documents/gd/:$PYTHONPATH
python detect_gd_line.py \
--source /media/ubuntu/Data/val_list.txt \
--checkpoint /media/ubuntu/Temp/VesselSeg-Pytorch/gd/gd_line_seg_U_Net_bs=4_lr=0.0001/best_model.pth \ 
--save-dir /media/ubuntu/Temp/VesselSeg-Pytorch/gd/gd_line_seg_U_Net_bs=4_lr=0.0001/large_results/
--img-size 512 --gap 16 --batchsize 4 --device 1
"""


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

    args = parser.parse_args()

    source, view_img, save_txt, imgsz, gap, \
    gt_xml_dir, gt_prefix, gt_subsize, gt_gap, \
    big_subsize, batchsize, score_thr, hw_thr = \
        args.source, args.view_img, args.save_txt, args.img_size, args.gap, \
        args.gt_xml_dir, args.gt_prefix, int(args.gt_subsize), int(args.gt_gap), args.big_subsize, \
        args.batchsize, args.score_thres, args.hw_thres
    subset = args.subset

    # Directories
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    names = {0: 'nonline', 1: 'line'}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    shown_labels = [0, 1]  # 只显示中大型杆塔和绝缘子

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

    stride = 32

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    seen = 0
    nc = len(names)
    inst_count = 1

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape([1,3,1,1])
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape([1,3,1,1])

    for ti in range(len(tiffiles)):
        image_id = ti + 1
        tiffile = tiffiles[ti]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

        print(ti, '=' * 80)
        print(file_prefix)

        mask_savefilename = save_dir + '/' + subset + "_" + file_prefix + "_LineSeg_result.png"
        if os.path.exists(mask_savefilename):
            continue

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
        if False: #orig_width * orig_height * ds.RasterCount < 0.8 * avaialble_mem_bytes:
            offsets = [[0, 0, orig_width, orig_height]]
        else:
            # 根据big_subsize计算子块的起始偏移
            offsets = compute_offsets(height=orig_height, width=orig_width, subsize=big_subsize, gap=2 * gt_gap)
        print('offsets: ', offsets)

        final_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        all_preds_filename = str(save_dir) + '/' + file_prefix + '_all_preds.pt'

        if True:

            all_preds = []
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
                    # img: BS x 3 x 224 x 224
                    # build the data pipeline
                    # prepare data

                    # results['filename'] = None
                    # results['ori_filename'] = None
                    # img = mmcv.imread(results['img'])
                    # results['img'] = img
                    # results['img_shape'] = img.shape
                    # results['ori_shape'] = img.shape

                    # img = img.astype(np.float32)
                    # img -= mean
                    # img /= std
                    # img = torch.from_numpy(img)
                    # img = img.to(device)
                    #
                    # data={'img':[img],
                    #       'img_metas':[[{'filename': [None],
                    #                      'ori_filename': [None],
                    #                      'img_shape': [[imgsz, imgsz, 3]],
                    #                      'ori_shape': [[imgsz, imgsz, 3]],
                    #                      'scale_factor': [[np.array([1.0, 1.0, 1.0, 1.0],
                    #                               dtype=np.float32)]],
                    #                      'pad_shape': [[imgsz, imgsz, 3]],
                    #                      'keep_ratio': [[True]]
                    #                      } for _ in range(img.shape[0])]]
                    #       }
                    inputs = torch.from_numpy(imgs.astype(np.float32)/255).to(device)
                    # forward the model
                    with torch.no_grad():
                        outputs = model(inputs)
                        results = outputs[:, 1].data.cpu().numpy()

                    for (x, y), result in zip(poss, results):
                        if np.any(result):
                            x += xoffset
                            y += yoffset
                            y2 = min(orig_height-1, y+result.shape[0])
                            x2 = min(orig_width-1, x+result.shape[1])
                            w = int(x2 - x)
                            h = int(y2 - y)
                            final_mask[y:y2, x:x2] = result[:h, :w] * 255

                # import pdb
                # pdb.set_trace()

                # pdb.set_trace()
                del dataset.img0
                del dataset
                import gc
                gc.collect()

        # cv2.imwrite(mask_savefilename, mask)
        cv2.imencode('.png', final_mask)[1].tofile(mask_savefilename)


if __name__ == '__main__':
    main()
