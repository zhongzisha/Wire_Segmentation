import sys, os, glob, shutil

import cv2
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
                    # cv2.imwrite(os.path.join(save_path, file_prefix + '_binary.png'), pred)
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
                    # cv2.imwrite(os.path.join(save_path, file_prefix + '_binary.png'), pred)

    log = OrderedDict([('val_loss', val_loss.avg)])
    return log


def save_numpy_array_to_tif(arr, reference_tif, label_maps, save_path, min_blob_size=50):
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

    for b, (label, label_name) in enumerate(label_maps.items()):
        print('write image data', b, label, label_name)
        band = outdata.GetRasterBand(b + 1)
        mask = (arr == label).astype(np.uint8)
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


def test_tif(tiffiles, net, device=None, patch_size=None, save_root=None, num_classes=2, args=None):
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
    label_maps = {
        1: 'landslide',
        2: 'water',
        3: 'tree',
        4: 'building',
    }
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

        # split into small tif files
        if args.tiled_tifs_dir != '':
            test_images_dir = os.path.join(args.tiled_tifs_dir, tiffile_prefix, 'tifs')
        else:
            test_images_dir = os.path.join(save_path, 'tifs')
        if not os.path.exists(test_images_dir):
            os.makedirs(test_images_dir)
            command = r'gdal_retile.py -of GTiff -ps 2048 2048 -overlap 64 -ot Byte -r cubic -targetDir %s %s' % (
                test_images_dir, tiffile
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
                        save_numpy_array_to_tif(pred, img_filename, label_maps, save_tif_filename, args.min_blob_size)
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

        if os.path.exists(results_dir):
            shutil.rmtree(results_dir, ignore_errors=True)
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    setpu_seed(2021)
    args = parse_args()
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

        save_dir = os.path.join(save_path, args.pth_filename, args.test_subset)
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

        save_dir = os.path.join(save_path, os.path.splitext(os.path.basename(args.pth_filename))[0], args.test_subset)
        test_tif(tiffiles, net, device,
                 patch_size=args.img_size,
                 save_root=save_dir,
                 num_classes=args.num_classes,
                 args=args)
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
