import sys, os, glob, shutil

import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time
from os.path import join
from lib.losses.loss import *
from lib.common import *
from lib.dataset import McSegDataset
from torch.utils.data import DataLoader
from config import parse_args
from lib.logger import Logger, Print_Logger
from collections import OrderedDict
import models
import segmentation_models_pytorch as smp
from osgeo import gdal, osr, ogr
import subprocess
import gc
import math


def CE_loss(input_logits, target_targets, ignore_index, temperature=1):
    # print(input_logits.shape, target_targets.shape, torch.unique(target_targets))
    if torch.all(target_targets == ignore_index):
        return torch.autograd.Variable(torch.zeros(1))
    return F.cross_entropy(input_logits / temperature, target_targets, ignore_index=ignore_index)


#  Load the data and extract patches
def get_dataloader(args):
    mean = None
    std = None
    if 'SMP_' in args.network:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    crop_shape = (args.img_size, args.img_size)
    print('crop_shape: ', crop_shape)
    train_set = McSegDataset(data_root=args.data_root,
                             subset=args.train_subset,
                             crop_shape=crop_shape,
                             mean=mean, std=std, cached=args.cached)
    val_set = McSegDataset(data_root=args.data_root, subset=args.val_subset, crop_shape=None,
                           mean=mean, std=std, cached=args.cached)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader


def test(test_images_dir, test_gts_dir, net,
         device=None, patch_size=None, save_path=None,
         num_classes=2, args=None):
    import glob
    img_filenames = glob.glob(os.path.join(test_images_dir, '*.jpg'))
    print(test_images_dir)
    print(img_filenames)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if 'SMP_' in args.network:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean, std = None, None
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255]])
    # 'landslide', 'water', 'tree', 'building'
    net.eval()
    val_loss = AverageMeter()
    if patch_size is not None:  # sliding window inference, from mmsegmentation
        with torch.no_grad():
            for batch_idx, img_filename in tqdm(enumerate(img_filenames), total=len(img_filenames)):
                image_np = cv2.imread(img_filename).astype(np.float32)[:, :, ::-1] / 255  # bgr -> rgb
                image = image_np.copy()
                if mean is not None:
                    image -= np.array(mean)
                if std is not None:
                    image /= np.array(std)
                image = np.transpose(image, [2, 0, 1])

                image = torch.from_numpy(image).unsqueeze(0).to(device)  # 1CHW
                h_stride, w_stride = patch_size // 2, patch_size // 2
                h_crop, w_crop = patch_size, patch_size
                batch_size, _, h_img, w_img = image.size()
                h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
                w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
                preds = image.new_zeros((batch_size, num_classes, h_img, w_img))
                count_mat = image.new_zeros((batch_size, 1, h_img, w_img))
                for h_idx in range(h_grids):
                    for w_idx in range(w_grids):
                        y1 = h_idx * h_stride
                        x1 = w_idx * w_stride
                        y2 = min(y1 + h_crop, h_img)
                        x2 = min(x1 + w_crop, w_img)
                        y1 = max(y2 - h_crop, 0)
                        x1 = max(x2 - w_crop, 0)
                        print(h_idx, w_idx, y1, y2, x1, x2)
                        crop_img = image[:, :, y1:y2, x1:x2]
                        crop_seg_logit = net(crop_img)
                        preds += F.pad(crop_seg_logit,
                                       (int(x1), int(preds.shape[3] - x2), int(y1),
                                        int(preds.shape[2] - y2)))

                        count_mat[:, :, y1:y2, x1:x2] += 1
                assert (count_mat == 0).sum() == 0
                if torch.onnx.is_in_onnx_export():
                    # cast count_mat to constant while exporting to ONNX
                    count_mat = torch.from_numpy(
                        count_mat.cpu().detach().numpy()).to(device=image.device)
                preds = preds / count_mat
                preds = torch.softmax(preds, dim=1)

                outputs = preds.data.cpu().numpy()
                pred = np.argmax(outputs, axis=1)[0] + 1
                pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                for label, color in enumerate(palette):
                    pred_color[pred == label, :] = color

                file_prefix = img_filename.split(os.sep)[-1].replace('.jpg', '')
                gt_filename = os.path.join(test_gts_dir, file_prefix + '.png')
                if os.path.exists(gt_filename):
                    gt = cv2.imread(gt_filename)[:, :, 0]
                else:
                    gt = np.zeros((h_img, w_img), dtype=np.uint8)
                gt_color = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
                for label, color in enumerate(palette):
                    gt_color[gt == label, :] = color

                if save_path is not None:
                    final_img = np.concatenate([
                        (image_np * 255).astype(np.uint8),
                        gt_color,
                        pred_color
                    ], axis=1)
                    cv2.imwrite(os.path.join(save_path, file_prefix + '.png'), final_img[:, :, ::-1])
                    cv2.imwrite(os.path.join(save_path, file_prefix + '_binary.png'), pred)
    else:
        with torch.no_grad():
            for batch_idx, img_filename in tqdm(enumerate(img_filenames), total=len(img_filenames)):
                image_np = cv2.imread(img_filename)
                h_img, w_img = image_np.shape[:2]
                image_np = cv2.imread(img_filename).astype(np.float32)[:, :, ::-1] / 255  # bgr -> rgb
                image = image_np.copy()
                if mean is not None:
                    image -= np.array(mean)
                if std is not None:
                    image /= np.array(std)
                image = np.transpose(image, [2, 0, 1])

                image = torch.from_numpy(image).unsqueeze(0).to(device)  # 1CHW
                outputs = net(image)

                outputs = outputs.data.cpu().numpy()
                pred = np.argmax(outputs, axis=1)[0] + 1
                pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                for label, color in enumerate(palette):
                    pred_color[pred == label, :] = color

                file_prefix = img_filename.split(os.sep)[-1].replace('.jpg', '')
                gt_filename = os.path.join(test_gts_dir, file_prefix + '.png')
                if os.path.exists(gt_filename):
                    gt = cv2.imread(gt_filename)[:, :, 0]
                else:
                    gt = np.zeros((h_img, w_img), dtype=np.uint8)
                gt_color = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
                for label, color in enumerate(palette):
                    gt_color[gt == label, :] = color

                if save_path is not None:
                    final_img = np.concatenate([
                        (image_np * 255).astype(np.uint8),
                        gt_color,
                        pred_color
                    ], axis=1)
                    cv2.imwrite(os.path.join(save_path, file_prefix + '.png'), final_img[:, :, ::-1])
                    cv2.imwrite(os.path.join(save_path, file_prefix + '_binary.png'), pred)

    log = OrderedDict([('val_loss', val_loss.avg)])
    return log


def line_detection(src, xoffset=0, yoffset=0, is_draw=True, save_path=None):
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
            print(dis, np.degrees(np.arctan((y1-y2)/(x1-x2))))
            if dis < 100:
                continue

            cv2.line(cdstP, (x1, y1), (x2, y2), (255, 255, 0), 3, cv2.LINE_AA)
            lines.append([x1 + xoffset, y1 + yoffset,
                          x2 + xoffset, y2 + yoffset])

    return cdstP, lines


def save_numpy_array_to_tif(arr, reference_tif, label_maps, save_path, min_blob_size=50,
                            tiled_width=5120, tiled_height=5120, tiled_overlap=64):
    ds = gdal.Open(reference_tif, gdal.GA_ReadOnly)
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
    ds = None

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(save_path, orig_width, orig_height, len(label_maps.keys()), gdal.GDT_Byte)
    # options=['COMPRESS=LZW', 'BIGTIFF=YES', 'INTERLEAVE=PIXEL'])
    # outdata = driver.CreateCopy(save_path, ds, 0, ['COMPRESS=LZW', 'BIGTIFF=YES', 'INTERLEAVE=PIXEL'])
    outdata.SetGeoTransform(geotransform)  # sets same geotransform as input
    outdata.SetProjection(projection)  # sets same projection as input

    all_lines = []
    for b, (label, label_name) in enumerate(label_maps.items()):
        print('write image data', b, label, label_name)
        band = outdata.GetRasterBand(b + 1)
        mask = (arr == label).astype(np.uint8)

        if label_name == 'line':
            yoffset, xoffset = [int(float(val)) for val in save_path.replace('.tif', '').split('_')[-2:]]
            xoffset = int((tiled_width - tiled_overlap) * (xoffset - 1))
            yoffset = int((tiled_height - tiled_overlap) * (yoffset - 1))
            print(save_path, xoffset, yoffset)

            if np.any(mask):
                # morph operators
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

                mask, lines = line_detection(mask.copy() * 255, xoffset, yoffset, save_path=save_path)  # do line detection  lines:[x1,y1,x2,y2]
                all_lines += lines

                # if len(result.shape) == 3:
                #     final_mask2[y:y2, x:x2] = result[:, :, 0]
                # else:
                #     final_mask2[y:y2, x:x2] = result
            if len(mask.shape) == 3:
                band.WriteArray(mask[:, :, 0], xoff=0, yoff=0)
            else:
                band.WriteArray(mask, xoff=0, yoff=0)
        else:
            mask = cv2.medianBlur(mask, 5)

            # morph operators
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # remove small connected blobs
            # find connected components
            n_components, output, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8)
            # remove background class
            sizes = stats[1:, -1]
            n_components = n_components - 1

            # remove blobs
            mask_clean = np.zeros(output.shape, dtype=np.uint8)
            # for every component in the image, keep it only if it's above min_blob_size
            for i in range(0, n_components):
                if sizes[i] >= min_blob_size:
                    mask_clean[output == i + 1] = 255

            # current_pred[current_pred == 0] = no_data_value
            print(np.min(mask_clean), np.max(mask_clean))
            band.WriteArray(mask_clean, xoff=0, yoff=0)

        # band.SetNoDataValue(no_data_value)
        band.FlushCache()
        del band
    outdata.FlushCache()
    del outdata
    del driver

    return all_lines


# numpy array to envi shapefile using gdal
def save_predictions_to_envi_xml_and_shp(preds, save_xml_filename, gdal_proj_info, gdal_trans_info,
                                         names=None, colors=None, is_line=False, spatialreference=None,
                                         is_save_xml=True, save_shp_filename=None):
    if names is None:
        names = {0: 'tower', 1: 'insulator'}
    if colors is None:
        colors = {0: [255, 0, 0], 1: [0, 0, 255]}

    # print('names', names)
    # print('colors', colors)
    # print('spatialreference', spatialreference)
    # print('gdal_proj_info', gdal_proj_info)
    # print('gdal_trans_info', gdal_trans_info)

    def get_coords(xmin, ymin, xmax, ymax):
        # [xmin, ymin]
        x1 = gdal_trans_info[0] + (xmin + 0.5) * gdal_trans_info[1] + (ymin + 0.5) * gdal_trans_info[2]
        y1 = gdal_trans_info[3] + (xmin + 0.5) * gdal_trans_info[4] + (ymin + 0.5) * gdal_trans_info[5]

        # [xmax, ymax]
        x3 = gdal_trans_info[0] + (xmax + 0.5) * gdal_trans_info[1] + (ymax + 0.5) * gdal_trans_info[2]
        y3 = gdal_trans_info[3] + (xmax + 0.5) * gdal_trans_info[4] + (ymax + 0.5) * gdal_trans_info[5]

        if is_line:
            return [x1, y1, x3, y3]
        else:
            # [xmax, ymin]
            x2 = gdal_trans_info[0] + (xmax + 0.5) * gdal_trans_info[1] + (ymin + 0.5) * gdal_trans_info[2]
            y2 = gdal_trans_info[3] + (xmax + 0.5) * gdal_trans_info[4] + (ymin + 0.5) * gdal_trans_info[5]
            # [xmin, ymax]
            x4 = gdal_trans_info[0] + (xmin + 0.5) * gdal_trans_info[1] + (ymax + 0.5) * gdal_trans_info[2]
            y4 = gdal_trans_info[3] + (xmin + 0.5) * gdal_trans_info[4] + (ymax + 0.5) * gdal_trans_info[5]

            return [x1, y1, x2, y2, x3, y3, x4, y4, x1, y1]

    lines = ['<?xml version="1.0" encoding="UTF-8"?>\n<RegionsOfInterest version="1.0">\n']
    # names = {0: '1', 1: '2'}
    # create output file
    # save_path = os.path.dirname(os.path.abspath(save_xml_filename))
    # print('save_path', save_path)
    # file_prefix = save_xml_filename.split(os.sep)[-1].replace('.xml', '')
    # print('file_prefix', file_prefix)
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    # shp_save_dir = os.path.join(save_path, file_prefix)
    # # print('shp_save_dir', shp_save_dir)
    # # if os.path.exists(shp_save_dir):
    # #     shutil.rmtree(shp_save_dir, ignore_errors=True)
    # # os.makedirs(shp_save_dir)
    # if not os.path.exists(shp_save_dir):
    #     os.makedirs(shp_save_dir)

    is_gt = False
    if len(preds) > 0:
        is_gt = len(preds[0]) == 5

    scores = []

    outDataSource = outDriver.CreateDataSource(save_shp_filename)
    for current_label, label_name in names.items():
        # print(current_label, label_name, colors[current_label])
        # save_shp_filename = os.path.join(shp_save_dir, label_name + '.shp')
        # if os.path.exists(save_shp_filename):
        #     os.remove(save_shp_filename)

        if is_line:
            outLayer = outDataSource.CreateLayer(label_name, spatialreference, geom_type=ogr.wkbLineString)
        else:
            outLayer = outDataSource.CreateLayer(label_name, spatialreference, geom_type=ogr.wkbPolygon)
        featureDefn = outLayer.GetLayerDefn()

        lines1 = [
            '<Region name="%s" color="%s">\n' % (label_name, ','.join([str(val) for val in colors[current_label]])),
            '<GeometryDef>\n<CoordSysStr>%s</CoordSysStr>\n' % (
                gdal_proj_info if gdal_proj_info != '' else 'none')]  # 这里不能有换行符

        count = 0
        for i, pred in enumerate(preds):
            if is_gt:
                xmin, ymin, xmax, ymax, label = pred
                score = 0.999
                label = int(label) - 1  # label==0: 杆塔, label==1: 绝缘子
            else:
                xmin, ymin, xmax, ymax, score, label = pred
                label = int(label)  # label==0: 杆塔, label==1: 绝缘子

            if label == current_label:
                coords = get_coords(xmin, ymin, xmax, ymax)

                if is_line:
                    lines1.append('<LineString>\n<Coordinates>\n')
                    lines1.append('%s\n' % (" ".join(['%.6f' % val for val in coords])))
                    lines1.append('</Coordinates>\n</LineString>\n')

                    coords = np.array(coords).reshape((-1, 2)).astype(np.float64)
                    if gdal_proj_info is None:
                        coords *= 0.013888889

                    poly = ogr.Geometry(ogr.wkbLineString)
                    for xx, yy in coords:
                        poly.AddPoint(xx, yy)

                else:
                    lines1.append('<Polygon>\n<Exterior>\n<LinearRing>\n<Coordinates>\n')
                    lines1.append('%s\n' % (" ".join(['%.6f' % val for val in coords])))
                    lines1.append('</Coordinates>\n</LinearRing>\n</Exterior>\n</Polygon>\n')

                    coords = np.array(coords).reshape((-1, 2)).astype(np.float64)
                    if gdal_proj_info is None:
                        coords *= 0.013888889

                    ring = ogr.Geometry(ogr.wkbLinearRing)
                    for xx, yy in coords:
                        ring.AddPoint(xx, yy)
                    poly = ogr.Geometry(ogr.wkbPolygon)
                    poly.AddGeometry(ring)

                # add new geom to layer
                outFeature = ogr.Feature(featureDefn)
                outFeature.SetGeometry(poly)
                outLayer.CreateFeature(outFeature)
                outFeature = None

                scores.append(score)
                count += 1
        lines1.append('</GeometryDef>\n</Region>\n')

        if count > 0:
            lines.append(''.join(lines1))

        featureDefn = None
        outLayer = None
    outDataSource = None

    lines.append('</RegionsOfInterest>\n')

    # if len(lines) > 0 and is_save_xml:
    #     with open(save_xml_filename, 'w') as fp:
    #         fp.writelines(lines)
    #     np.savetxt(save_xml_filename.replace('.xml', '_scores.txt'),
    #                np.array(scores, dtype=np.float32),
    #                fmt='%.6f', delimiter=',', encoding='utf-8')


def test_tif(tiffiles, net, device=None, patch_size=None, save_root=None, num_classes=2, args=None,
             label_maps=None, tiled_width=5120, tiled_height=5120, tiled_overlap=64):
    bands_info_txt = "E:\\Downloads\\mc_seg\\tifs\\bands_info.txt"
    invalid_tifs_txt = "E:\\Downloads\\mc_seg\\tifs\\invalid_tifs.txt"
    bands_info = []
    if os.path.exists(bands_info_txt):
        with open(bands_info_txt, 'r', encoding='utf-8-sig') as fp:
            bands_info = [line.strip() for line in fp.readlines()]
    invalid_tifs = []
    if os.path.exists(invalid_tifs_txt):
        with open(invalid_tifs_txt, 'r', encoding='utf-8-sig') as fp:
            invalid_tifs = [line.strip() for line in fp.readlines()]

    import glob
    from PIL import Image

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if 'SMP_' in args.network:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean, std = None, None
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255]])
    # 'landslide', 'water', 'tree', 'building'
    # label_maps = {
    #     1: 'landslide',
    #     2: 'water',
    #     3: 'tree',
    #     4: 'building',
    # }
    net.eval()

    all_tiffiles = []
    for tiffile in tiffiles:
        all_tiffiles += tiffile.split(',')

    for tiffile in all_tiffiles:
        tiffile_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
        if tiffile_prefix in invalid_tifs:
            continue

        save_path = os.path.join(save_root, tiffile_prefix)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        shp_exists = [os.path.exists(os.path.join(save_path, label_name + '.shp'))
                      for label, label_name in label_maps.items()]
        if np.all(shp_exists):
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
        ds = None

        # split into small tif files
        if args.tiled_tifs_dir != '':
            test_images_dir = os.path.join(args.tiled_tifs_dir, tiffile_prefix, 'tifs')
        else:
            test_images_dir = os.path.join(save_path, 'tifs')
        if not os.path.exists(test_images_dir):
            os.makedirs(test_images_dir)
            command = r'gdal_retile.py -of GTiff -ps %d %d -overlap %d -ot Byte -r cubic -targetDir %s %s' % (
                tiled_width, tiled_height, tiled_overlap, test_images_dir, tiffile
            )
            print(command)
            os.system(command)

        results_dir = os.path.join(save_path, 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        tmp_dir = os.path.join(save_path, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        img_filenames = glob.glob(os.path.join(test_images_dir, '*.tif'))
        if patch_size is not None:  # sliding window inference, from mmsegmentation
            merge_tif_filenames = []
            all_lines = []
            with torch.no_grad():
                for batch_idx, img_filename in tqdm(enumerate(img_filenames), total=len(img_filenames)):
                    file_prefix = img_filename.split(os.sep)[-1].replace('.tif', '')
                    save_tif_filename = os.path.join(results_dir, file_prefix + '.tif')
                    if os.path.exists(save_tif_filename):
                        merge_tif_filenames.append(save_tif_filename)
                        continue
                    # image_np = cv2.imread(img_filename).astype(np.float32)[:,:,::-1] / 255  # bgr -> rgb
                    image_np = np.array(Image.open(img_filename)).astype(np.float32) / 255
                    all_zero_im = np.sum(image_np, axis=2)
                    hh, ww = image_np.shape[:2]
                    if len(np.unique(image_np)) == 1:
                        pred = np.zeros((hh, ww), dtype=np.uint8)
                    else:
                        if tiffile_prefix in bands_info:
                            image_np = image_np[:, :, ::-1]

                        image = image_np.copy()
                        if mean is not None:
                            image -= np.array(mean)
                        if std is not None:
                            image /= np.array(std)
                        image = np.transpose(image, [2, 0, 1])

                        image = torch.from_numpy(image).unsqueeze(0).to(device)  # 1CHW
                        h_stride, w_stride = patch_size // 2, patch_size // 2
                        h_crop, w_crop = patch_size, patch_size
                        batch_size, _, h_img, w_img = image.size()
                        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
                        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
                        preds = image.new_zeros((batch_size, num_classes, h_img, w_img))
                        count_mat = image.new_zeros((batch_size, 1, h_img, w_img))
                        for h_idx in range(h_grids):
                            for w_idx in range(w_grids):
                                y1 = h_idx * h_stride
                                x1 = w_idx * w_stride
                                y2 = min(y1 + h_crop, h_img)
                                x2 = min(x1 + w_crop, w_img)
                                y1 = max(y2 - h_crop, 0)
                                x1 = max(x2 - w_crop, 0)
                                print(h_idx, w_idx, y1, y2, x1, x2)
                                crop_img = image[:, :, y1:y2, x1:x2]
                                crop_img_h, crop_img_w = crop_img.shape[2:]
                                if crop_img.shape[2:] != (patch_size, patch_size):
                                    crop_img = F.pad(crop_img, (0, int(patch_size - crop_img.shape[3]),
                                                                0, int(patch_size - crop_img.shape[2])))
                                crop_seg_logit = net(crop_img)
                                preds += F.pad(crop_seg_logit[:, :, :crop_img_h, :crop_img_w],
                                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                                int(preds.shape[2] - y2)))

                                count_mat[:, :, y1:y2, x1:x2] += 1
                        assert (count_mat == 0).sum() == 0
                        if torch.onnx.is_in_onnx_export():
                            # cast count_mat to constant while exporting to ONNX
                            count_mat = torch.from_numpy(
                                count_mat.cpu().detach().numpy()).to(device=image.device)
                        preds = preds / count_mat
                        preds = torch.softmax(preds, dim=1)

                        outputs = preds.data.cpu().numpy()
                        pred = np.argmax(outputs, axis=1)[0] + 1
                        del preds, outputs, count_mat, image
                        gc.collect()

                    pred[all_zero_im == 0] = 0
                    pred[all_zero_im == 3] = 0
                    pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                    for label, color in enumerate(palette):
                        pred_color[pred == label, :] = color

                    gt = np.zeros((pred.shape[0], pred.shape[1]), dtype=np.uint8)
                    gt_color = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
                    for label, color in enumerate(palette):
                        gt_color[gt == label, :] = color

                    if results_dir is not None:
                        final_img = np.concatenate([
                            (image_np * 255).astype(np.uint8),
                            gt_color,
                            pred_color
                        ], axis=1)
                        # cv2.imwrite(os.path.join(save_path, file_prefix + '.png'), final_img[:, :, ::-1])
                        # cv2.imwrite(os.path.join(save_path, file_prefix + '_binary.png'), pred)
                        Image.fromarray(final_img).save(os.path.join(results_dir, file_prefix + '.png'))

                        # save_tif_filename = os.path.join(save_path, file_prefix + args.test_image_postfix)
                        lines = save_numpy_array_to_tif(pred, img_filename, label_maps, save_tif_filename,
                                                        args.min_blob_size,
                                                        tiled_width=tiled_width,
                                                        tiled_height=tiled_height,
                                                        tiled_overlap=tiled_overlap)
                        all_lines += lines
                        merge_tif_filenames.append(save_tif_filename)
                        # save_path1 = os.path.join(save_path, file_prefix)
                        # if not os.path.exists(save_path1):
                        #     os.makedirs(save_path1)
                        # for label in np.unique(pred):
                        #     # cv2.imwrite(os.path.join(save_path, file_prefix + '_%s.png' % label_maps[int(label)]), pred)
                        #     current_pred = (pred == label).astype(np.uint8) * 255
                        #     Image.fromarray(current_pred).save(
                        #         os.path.join(save_path1, '%s.png' % label_maps[int(label)])
                        #     )
            if len(merge_tif_filenames) > 0:
                current_dir = os.getcwd()
                os.chdir(results_dir)
                command = r'gdal_merge.py -of GTiff -co "TILED=YES" -co "COMPRESS=LZW" -co "BIGTIFF=YES" -n 0 -o %s ' \
                          r'%s' % (
                              os.path.join(save_path, 'merged.tif'),
                              ' '.join([os.path.basename(name) for name in merge_tif_filenames])
                          )
                print(command)
                # os.system(command)
                process = subprocess.Popen(command, shell=True)
                output = process.communicate()[0]
                os.chdir(current_dir)

                for b, (label, label_name) in enumerate(label_maps.items()):
                    command = r'gdal_translate -of GTiff -co "TILED=YES" -co "COMPRESS=LZW" -co "BIGTIFF=YES" -b %d %s %s' % (
                        b + 1,
                        os.path.join(save_path, 'merged.tif'),
                        os.path.join(save_path, '%s.tif' % label_name),
                    )
                    print(command)
                    os.system(command)

                    if label_name != 'line':
                        command = r'gdal_polygonize.py %s -b 1 -f "ESRI Shapefile" %s' % (
                            os.path.join(save_path, '%s.tif' % label_name),
                            os.path.join(tmp_dir, '%s_tmp.shp' % label_name)
                        )
                        print(command)
                        os.system(command)
                    else:
                        save_absfilepath = os.path.join(save_path, '%s.shp' % label_name)
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
                                                                         bins=np.arange(-np.pi / 2, np.pi / 2,
                                                                                        5 * np.pi / 180))
                            # print('radian_hist', radian_hist)
                            # print('radian_bin_edges', radian_bin_edges)
                            max_hist = np.argmax(radian_hist)
                            max_radian = radian_bin_edges[max_hist]
                            # print('max_hist', max_hist)
                            # print('max_radian', max_radian)

                            if len(valid_lines) > 0:
                                all_preds = np.concatenate([np.array(valid_lines).reshape([-1, 4]),
                                                            np.ones((len(valid_lines), 2), dtype=np.float32)], axis=1)
                            else:
                                all_preds = []

                                # self.names = {1: 'Line'}
                                # self.colors = {1: [255, 255, 255]}
                            save_predictions_to_envi_xml_and_shp(preds=all_preds,
                                                                 save_xml_filename=None,
                                                                 gdal_proj_info=projection_sr,  # projection_esri,
                                                                 gdal_trans_info=geotransform,
                                                                 names={1: 'line'},
                                                                 colors={1: [255, 255, 255]},
                                                                 is_line=True,
                                                                 spatialreference=projection_sr,
                                                                 is_save_xml=False,
                                                                 save_shp_filename=save_absfilepath)

                    if label_name != 'line':
                        command = r'ogr2ogr -where "\"DN\"=255" -f "ESRI Shapefile" %s %s' % (
                            os.path.join(save_path, '%s.shp' % label_name),
                            os.path.join(tmp_dir, '%s_tmp.shp' % label_name)
                        )
                        print(command)
                        os.system(command)

        if True:
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir, ignore_errors=True)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)


def merge_mc_seg_results(args=None):
    tiffiles = [
        'G:\\gddata\\all\\2-WV03-在建杆塔.tif',
        'G:\\gddata\\all\\3-wv02-在建杆塔.tif',
        'G:\\gddata\\all\\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif',
        # 'G:\\gddata\\all\\220kvqinshunxiann39-n42.tif',
        'G:\\gddata\\all\\220kvqinshunxiann53-n541.tif',
        'G:\\gddata\\all\\220kvqinshunxiann70-n71.tif',
        'G:\\gddata\\all\\po008535_gd33.tif',
        'G:\\gddata\\all\\WV03-曲花甲线-20170510.tif',
        'G:\\gddata\\all\\WV03-英连线-20170206.tif',
        # 'G:\\gddata\\all\\候村250m_mosaic.tif',
    ]
    # tiffiles = glob.glob(os.path.join(args.test_tifs_dir, '*.tif'))
    epochs = [20, 30, 40, 50]
    log_roots = [
        r'E:\Downloads\mc_seg\logs\U_Net_512_4_0.0001',
        r'E:\Downloads\mc_seg\logs\SMP_UnetPlusPlus_512_8_0.001',
    ]
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255]])
    # 'landslide', 'water', 'tree', 'building'
    label_maps = {
        1: 'landslide',
        2: 'water',
        3: 'tree',
        4: 'building',
    }

    save_root = r'E:\Downloads\mc_seg\logs\merged_results'

    all_tiffiles = []
    for tiffile in tiffiles:
        all_tiffiles += tiffile.split(',')

    for tiffile in all_tiffiles:
        tiffile_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

        save_path = os.path.join(save_root, tiffile_prefix)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tmp_dir = os.path.join(save_root, tiffile_prefix, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        shp_exists = [os.path.exists(os.path.join(save_path, label_name + '.shp'))
                      for label, label_name in label_maps.items()]
        if np.all(shp_exists):
            continue

        print(tiffile)
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
        ds = None

        merged_tif_filename = os.path.join(save_path, 'merged.tif')
        if not os.path.exists(merged_tif_filename):
            img = np.zeros((orig_height, orig_width, 4), dtype=np.uint8)  # RGB format
            for log_root in log_roots:
                for epoch in epochs:
                    result_tif_filename = os.path.join(
                        log_root, 'epoch-%d' % epoch, 'test_tif', tiffile_prefix, 'merged.tif'
                    )
                    if not os.path.exists(result_tif_filename):
                        continue
                    print(result_tif_filename)
                    ds = gdal.Open(result_tif_filename, gdal.GA_ReadOnly)
                    for b in range(4):
                        band = ds.GetRasterBand(b + 1)
                        img[:, :, b] += \
                            (band.ReadAsArray(0, 0, win_xsize=orig_width, win_ysize=orig_height) > 0).astype(np.uint8)
                    ds = None
            print('get pred', np.unique(img))
            pred = np.argmax(img, axis=2) + 1
            save_numpy_array_to_tif(pred, tiffile, label_maps, merged_tif_filename, args.min_blob_size)
            del img, pred

        for b, (label, label_name) in enumerate(label_maps.items()):
            command = r'gdal_translate -of GTiff -co "TILED=YES" -co "COMPRESS=LZW" -co "BIGTIFF=YES" -b %d %s %s' % (
                b + 1,
                merged_tif_filename,
                os.path.join(save_path, '%s.tif' % label_name),
            )
            print(command)
            os.system(command)

            command = r'gdal_polygonize.py %s -b 1 -f "ESRI Shapefile" %s' % (
                os.path.join(save_path, '%s.tif' % label_name),
                os.path.join(tmp_dir, '%s_tmp.shp' % label_name)
            )
            print(command)
            os.system(command)

            command = r'ogr2ogr -where "\"DN\"=255" -f "ESRI Shapefile" %s %s' % (
                os.path.join(save_path, '%s.shp' % label_name),
                os.path.join(tmp_dir, '%s_tmp.shp' % label_name)
            )
            print(command)
            os.system(command)
        # break

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def refine_result(result, reference_tif, label_name, save_path, tmp_dir):
    H, W = result.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_ERODE, kernel)
    # morph operators
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    # remove small connected blobs
    # find connected components
    n_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        result, connectivity=8)
    # remove background class
    sizes = stats[1:, -1]
    n_components = n_components - 1

    sizes_inds = np.argsort(sizes)
    ind = int(np.floor(0.98*len(sizes)))
    blob_size = sizes[sizes_inds[ind]]

    # remove blobs
    mask_clean = np.zeros(output.shape, dtype=np.uint8)
    # for every component in the image, keep it only if it's above min_blob_size
    for i in range(0, n_components):
        if sizes[i] >= blob_size:
            mask_clean[output == i + 1] = 255

    # current_pred[current_pred == 0] = no_data_value
    print(np.min(mask_clean), np.max(mask_clean))


    ds = gdal.Open(reference_tif, gdal.GA_ReadOnly)
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
    ds = None

    save_tiffilename = os.path.join(tmp_dir, '%s_tmp.tif' % label_name)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(save_tiffilename, orig_width, orig_height, 1, gdal.GDT_Byte)
    # options=['COMPRESS=LZW', 'BIGTIFF=YES', 'INTERLEAVE=PIXEL'])
    # outdata = driver.CreateCopy(save_path, ds, 0, ['COMPRESS=LZW', 'BIGTIFF=YES', 'INTERLEAVE=PIXEL'])
    outdata.SetGeoTransform(geotransform)  # sets same geotransform as input
    outdata.SetProjection(projection)  # sets same projection as input

    band = outdata.GetRasterBand(1)
    band.WriteArray(mask_clean, xoff=0, yoff=0)

    # band.SetNoDataValue(no_data_value)
    band.FlushCache()
    del band
    outdata.FlushCache()
    del outdata
    del driver

    command = r'gdal_translate -of GTiff -co "TILED=YES" -co "COMPRESS=LZW" -co "BIGTIFF=YES" %s %s' % (
        save_tiffilename,
        os.path.join(save_path, '%s.tif' % label_name),
    )
    print(command)
    os.system(command)

    command = r'gdal_polygonize.py %s -b 1 -f "ESRI Shapefile" %s' % (
        os.path.join(save_path, '%s.tif' % label_name),
        os.path.join(tmp_dir, '%s_tmp.shp' % label_name)
    )
    print(command)
    os.system(command)

    command = r'ogr2ogr -where "\"DN\"=255" -f "ESRI Shapefile" %s %s' % (
        os.path.join(save_path, '%s.shp' % label_name),
        os.path.join(tmp_dir, '%s_tmp.shp' % label_name)
    )
    print(command)
    os.system(command)


def generate_new_labeling_from_merged_results(args):
    tiffiles = [
        # 'G:\\gddata\\all\\2-WV03-在建杆塔.tif',
        'G:\\gddata\\all\\3-wv02-在建杆塔.tif',
        'G:\\gddata\\all\\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif',
        # 'G:\\gddata\\all\\220kvqinshunxiann39-n42.tif',
        'G:\\gddata\\all\\220kvqinshunxiann53-n541.tif',
        'G:\\gddata\\all\\220kvqinshunxiann70-n71.tif',
        'G:\\gddata\\all\\po008535_gd33.tif',
        'G:\\gddata\\all\\WV03-曲花甲线-20170510.tif',
        'G:\\gddata\\all\\WV03-英连线-20170206.tif',
        # 'G:\\gddata\\all\\候村250m_mosaic.tif',
    ]
    # tiffiles = glob.glob(os.path.join(args.test_tifs_dir, '*.tif'))
    epochs = [20, 30, 40, 50]
    log_roots = [
        r'E:\Downloads\mc_seg\logs\U_Net_512_4_0.0001',
        r'E:\Downloads\mc_seg\logs\SMP_UnetPlusPlus_512_8_0.001',
    ]
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255]])
    # 'landslide', 'water', 'tree', 'building'
    label_maps = {
        1: 'landslide',
        2: 'water',
        3: 'tree',
        4: 'building',
    }

    merged_root = r'E:\Downloads\mc_seg\logs\merged_results'
    save_root = r'E:\Downloads\mc_seg\logs\merged_results_refined'

    all_tiffiles = []
    for tiffile in tiffiles:
        all_tiffiles += tiffile.split(',')

    for tiffile in all_tiffiles:
        tiffile_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

        merged_path = os.path.join(merged_root, tiffile_prefix)
        save_path = os.path.join(save_root, tiffile_prefix)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        tmp_dir = os.path.join(save_root, tiffile_prefix, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        shp_exists = [os.path.exists(os.path.join(save_path, label_name + '.shp'))
                      for label, label_name in label_maps.items()]
        if np.all(shp_exists):
            continue

        print(tiffile)
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

        # get all_zero_pixels
        img = np.zeros((orig_height, orig_width), dtype=np.uint8)  # RGB format
        ds = gdal.Open(tiffile, gdal.GA_ReadOnly)
        for b in range(3):
            band = ds.GetRasterBand(b + 1)
            img |= band.ReadAsArray(0, 0, win_xsize=orig_width, win_ysize=orig_height)
        y0s, x0s = np.where(img == 0)
        del img
        ds = None
        print(len(y0s), len(x0s))

        for b, (label, label_name) in enumerate(label_maps.items()):
            print(label_name)
            label_filename = os.path.join(merged_path, '%s.tif' % label_name)
            merged_result = np.array((orig_height, orig_width), dtype=np.uint8)
            ds = gdal.Open(label_filename, gdal.GA_ReadOnly)
            for b in range(1):
                band = ds.GetRasterBand(b + 1)
                merged_result = band.ReadAsArray(0, 0, win_xsize=orig_width, win_ysize=orig_height)
            ds = None
            merged_result[y0s, x0s] = 0
            print('refine result')
            refine_result(merged_result, tiffile, label_name, save_path, tmp_dir)
        # break

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    setpu_seed(2021)
    args = parse_args()

    if args.action == 'merge_results':
        merge_mc_seg_results(args)
        sys.exit(-1)
    if args.action == 'generate_new_labeling_from_merged_results':
        generate_new_labeling_from_merged_results(args)
        sys.exit(-1)

    save_path = join(args.outf, args.save)
    save_args(args, save_path)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    cudnn.benchmark = True

    log = Logger(save_path)
    sys.stdout = Print_Logger(os.path.join(save_path, 'train_log.txt'))
    print('The computing device used is: ', 'GPU' if device.type == 'cuda' else 'CPU')

    network_name = args.network
    if network_name == 'U_Net':
        net = models.UNetFamily.U_Net(img_ch=3, output_ch=args.num_classes,
                                      has_softmax=False).to(device)
    elif network_name == 'SMP_UnetPlusPlus':
        net = smp.UnetPlusPlus(in_channels=3, classes=args.num_classes).to(device)
    elif network_name == 'SMP_Unet':
        net = smp.Unet(in_channels=3, classes=args.num_classes, activation='softmax2d').to(device)
    elif network_name == 'Dense_Unet':
        net = models.UNetFamily.Dense_Unet(in_chan=3, out_chan=args.num_classes).to(device)
    elif network_name == 'R2AttU_Net':
        net = models.UNetFamily.R2AttU_Net(img_ch=3, output_ch=args.num_classes).to(device)
    elif network_name == 'AttU_Net':
        net = models.UNetFamily.AttU_Net(img_ch=3, output_ch=args.num_classes).to(device)
    elif network_name == 'R2U_Net':
        net = models.UNetFamily.R2U_Net(img_ch=3, output_ch=args.num_classes).to(device)
    elif network_name == 'LadderNet':
        net = models.LadderNet(inplanes=3, num_classes=args.num_classes, layers=3, filters=16).to(device)
    else:
        print('wrong network type. exit.')
        sys.exit(-1)
    print("Total number of parameters: " + str(count_parameters(net)))

    if args.action == 'train':
        log.save_graph(net, torch.randn((1, 3, 512, 512)).to(device).to(
            device=device))  # Save the model structure to the tensorboard file

    if args.action == 'do_test':
        # Load checkpoint
        print('==> Loading checkpoint...')
        checkpoint = torch.load(join(save_path, args.pth_filename))
        net.load_state_dict(checkpoint['net'])

        save_dir = os.path.join(save_path, os.path.splitext(os.path.basename(args.pth_filename))[0], args.test_subset)
        test(args.test_images_dir, args.test_gts_dir, net, device,
             patch_size=args.img_size,
             save_path=save_dir,
             num_classes=args.num_classes,
             args=args)
        sys.exit(-1)
    if args.action == 'do_test_tif':
        # Load checkpoint
        print('==> Loading checkpoint...')
        checkpoint = torch.load(join(save_path, args.pth_filename))
        net.load_state_dict(checkpoint['net'])

        tiffiles = [
            # 'G:\\gddata\\all\\2-WV03-在建杆塔.tif',
            'G:\\gddata\\all\\3-wv02-在建杆塔.tif',
            'G:\\gddata\\all\\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif',
            'G:\\gddata\\all\\220kvqinshunxiann39-n42.tif',
            'G:\\gddata\\all\\220kvqinshunxiann53-n541.tif',
            'G:\\gddata\\all\\220kvqinshunxiann70-n71.tif',
            'G:\\gddata\\all\\po008535_gd33.tif',
            'G:\\gddata\\all\\WV03-曲花甲线-20170510.tif',
            'G:\\gddata\\all\\WV03-英连线-20170206.tif',
            # 'G:\\gddata\\all\\候村250m_mosaic.tif',
        ]
        # tiffiles = glob.glob(os.path.join(args.test_tifs_dir, '*.tif'))
        label_maps = {
            1: 'landslide',
            2: 'water',
            3: 'tree',
            4: 'building',
        }

        save_dir = os.path.join(save_path, os.path.splitext(os.path.basename(args.pth_filename))[0], args.test_subset)
        test_tif(tiffiles, net, device,
                 patch_size=args.img_size,
                 save_root=save_dir,
                 num_classes=args.num_classes,
                 args=args,
                 label_maps=label_maps)
        sys.exit(-1)
    if args.action == 'do_test_tif_line_foreign':
        # Load checkpoint
        print('==> Loading checkpoint...')
        checkpoint = torch.load(join(save_path, args.pth_filename))
        net.load_state_dict(checkpoint['net'])

        tiffiles = [
            'E:\\generated_big_test_images\\line_foreign\\2-WV03-在建杆塔.tif',
            'E:\\generated_big_test_images\\line_foreign\\2-WV03-在建杆塔_line_region.tif',
        ]
        tiffiles = glob.glob(os.path.join(args.test_tifs_dir, '*.tif'))
        label_maps = {
            # 1: 'bg',
            2: 'line',
            3: 'line_foreign',
        }

        save_dir = os.path.join(save_path, os.path.splitext(os.path.basename(args.pth_filename))[0], args.test_subset)
        test_tif(tiffiles, net, device,
                 patch_size=args.img_size,
                 save_root=save_dir,
                 num_classes=args.num_classes,
                 args=args,
                 label_maps=label_maps)
        sys.exit(-1)

    # torch.nn.init.kaiming_normal(net, mode='fan_out')      # Modify default initialization method
    # net.apply(weight_init)

    # criterion = LossMulti(jaccard_weight=0,class_weights=np.array([0.5,0.5]))

    # create a list of learning rate with epochs
    # lr_epoch = np.array([50, args.N_epochs])
    # lr_value = np.array([0.001, 0.0001])
    # lr_schedule = make_lr_schedule(lr_epoch,lr_value)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    # optimizer = optim.SGD(net.parameters(),lr=lr_schedule[0], momentum=0.9, weight_decay=5e-4, nesterov=True)
    base_lr = args.lr
    # criterion = DiceLoss(n_classes=args.num_classes)  # Initialize loss function
    optimizer = optim.Adam(net.parameters(), lr=base_lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0)

    print(args)
    train_loader, val_loader = get_dataloader(args)  # create dataloader

    best = {'epoch': 0, 'AUC_roc': 0.5}  # Initialize the best epoch and performance(AUC of ROC)
    trigger = 0  # Early stop Counter

    iter_num = 0
    for epoch in range(args.start_epoch, args.N_epochs + 1):
        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
              (epoch, args.N_epochs, optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))

        # train stage
        # train_log = train(train_loader, net, criterion, optimizer, device)
        net.train()
        train_loss = AverageMeter()
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # print(batch_idx, inputs.shape, targets.shape)
            # print(type(inputs), inputs.min(), inputs.max())
            # print(type(targets), targets.min(), targets.max())
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)  # here, for Swin_Unet, outputs is logits, others are probs
            loss = CE_loss(outputs, targets, ignore_index=255)
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1

            train_loss.update(loss.item(), inputs.size(0))
        train_log = OrderedDict([('train_loss', train_loss.avg)])

        # val stage
        # if 'Swin_Unet' in network_name:
        #     val_log = val(val_loader, net, criterion, device, patch_size=args.img_size)
        # else:
        #     val_log = val(val_loader, net, criterion, device)

        val_log = {}
        log.update(epoch, train_log, val_log)  # Add log information

        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch == 1 or epoch % 10 == 0:
        # Save checkpoint of latest and best model.
            state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, join(save_path, 'epoch-%d.pth' % epoch))
        trigger += 1
        # if val_log['val_auc_roc'] > best['AUC_roc']:
        #     print('\033[0;33mSaving best model!\033[0m')
        #     torch.save(state, join(save_path, 'best_model.pth'))
        #     best['epoch'] = epoch
        #     best['AUC_roc'] = val_log['val_auc_roc']
        #     trigger = 0
        # print('Best performance at Epoch: {} | AUC_roc: {}'.format(best['epoch'], best['AUC_roc']))
        # # early stopping
        # if not args.early_stop is None:
        #     if trigger >= args.early_stop:
        #         print("=> early stopping")
        #         break
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
