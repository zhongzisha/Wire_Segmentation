import sys,os

import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import time
from os.path import join
from lib.losses.loss import *
from lib.common import *
from lib.dataset import LineForeignDataset
from torch.utils.data import DataLoader
from config import parse_args
from lib.logger import Logger, Print_Logger
from collections import OrderedDict
import models
import segmentation_models_pytorch as smp


#  Load the data and extract patches
def get_dataloader(args):
    divide255 = True
    mean = None
    std = None
    if 'SMP_' in args.network:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    crop_shape = (args.img_size, args.img_size)
    print('crop_shape: ', crop_shape)
    train_set = LineForeignDataset(data_root=args.data_root,
                                   subset=args.train_subset,
                                   crop_shape=crop_shape,
                                   mean=mean, std=std)
    val_set = LineForeignDataset(data_root=args.data_root, subset=args.val_subset, crop_shape=None,
                                 mean=mean, std=std)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader


def test(test_images_dir, test_gts_dir, net,
         device=None, patch_size=None, save_path=None,
         num_classes=2):
    import glob
    img_filenames = glob.glob(os.path.join(test_images_dir, '*.jpg'))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    palette = np.array([[0, 0, 0], [255, 255, 255], [255, 255, 0], [0, 255, 255]])
    net.eval()
    val_loss = AverageMeter()
    if patch_size is not None:  # sliding window inference, from mmsegmentation
        with torch.no_grad():
            for batch_idx, img_filename in tqdm(enumerate(img_filenames), total=len(img_filenames)):
                image_np = cv2.imread(img_filename)
                image = np.transpose(image_np, [2, 0, 1]).astype(np.float32) / 255

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
                pred = np.argmax(outputs, axis=1)[0]
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
                        image_np,
                        gt_color,
                        pred_color
                    ], axis=1)
                    cv2.imwrite(os.path.join(save_path, file_prefix + '.png'), final_img)
                    cv2.imwrite(os.path.join(save_path, file_prefix + '_binary.png'), pred)
    else:
        with torch.no_grad():
            for batch_idx, img_filename in tqdm(enumerate(img_filenames), total=len(img_filenames)):
                image_np = cv2.imread(img_filename)
                h_img, w_img = image_np.shape[:2]
                image = np.transpose(image_np, [2, 0, 1]).astype(np.float32) / 255

                image = torch.from_numpy(image).unsqueeze(0).to(device)  # 1CHW
                outputs = net(image)

                outputs = outputs.data.cpu().numpy()
                pred = np.argmax(outputs, axis=1)[0]
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
                        image_np,
                        gt_color,
                        pred_color
                    ], axis=1)
                    cv2.imwrite(os.path.join(save_path, file_prefix + '.png'), final_img)
                    cv2.imwrite(os.path.join(save_path, file_prefix + '_binary.png'), pred)

    log = OrderedDict([('val_loss', val_loss.avg)])
    return log


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
        net = models.UNetFamily.U_Net(img_ch=3, output_ch=args.num_classes).to(device)
    elif network_name == 'SMP_UnetPlusPlus':
        net = smp.UnetPlusPlus(in_channels=3, classes=args.num_classes, activation='softmax2d').to(device)
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

    log.save_graph(net, torch.randn((1, 3, 512, 512)).to(device).to(
        device=device))  # Save the model structure to the tensorboard file

    if args.action == 'do_test':
        # Load checkpoint
        print('==> Loading checkpoint...')
        checkpoint = torch.load(join(save_path, args.pth_filename))
        net.load_state_dict(checkpoint['net'])

        save_dir = os.path.join(save_path, args.test_subset)
        test(args.test_images_dir, args.test_gts_dir, net, device,
             patch_size=args.img_size,
             save_path=save_dir,
             num_classes=args.num_classes)
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
    criterion = DiceLoss(n_classes=args.num_classes)  # Initialize loss function
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
            loss = criterion(outputs, targets)
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
