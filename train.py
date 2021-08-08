import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import random, sys, time
from os.path import join
import torch
from lib.extract_patches import get_data
from lib.losses.loss import *
from lib.visualize import group_images, save_img
from lib.common import *
from lib.dataset import TrainDataset, GdDataset
from torch.utils.data import DataLoader
from config import parse_args
from lib.logger import Logger, Print_Logger
from collections import OrderedDict
from lib.metrics import Evaluate
import models
from test import Test
from swin_unet_config import get_config


#  Load the data and extract patches
def get_dataloader(args):
    if args.dataset_type == 'GdDataset':
        crop_shape = (224, 224) if args.network == 'Swin_Unet' else None
        print('crop_shape: ', crop_shape)
        train_set = GdDataset(data_root=args.data_root, subset=args.train_subset, crop_shape=crop_shape)
        val_set = GdDataset(data_root=args.data_root, subset=args.val_subset, crop_shape=crop_shape)
    else:
        patches_imgs_train, patches_masks_train = get_data(data_root=args.data_root, subset=args.train_subset)
        patches_imgs_val, patches_masks_val = get_data(data_root=args.data_root, subset=args.val_subset)

        train_set = TrainDataset(patches_imgs_train, patches_masks_train, mode=args.train_subset)
        val_set = TrainDataset(patches_imgs_val, patches_masks_val, mode=args.val_subset)

        # Save some samples of feeding to the neural network
        N_sample = min(patches_imgs_train.shape[0], 50)
        save_img(group_images((patches_imgs_train[0:N_sample, :, :, :] * 255).astype(np.uint8), 10),
                 join(args.outf, args.save, "sample_input_imgs.png"))
        save_img(group_images((patches_masks_train[0:N_sample, :, :, :] * 255).astype(np.uint8), 10),
                 join(args.outf, args.save, "sample_input_masks.png"))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader


# train 
def train(train_loader, net, criterion, optimizer, device):
    net.train()
    train_loss = AverageMeter()

    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
    log = OrderedDict([('train_loss', train_loss.avg)])
    return log


# val 
def val(val_loader, net, criterion, device):
    net.eval()
    val_loss = AverageMeter()
    evaluater = Evaluate()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss.update(loss.item(), inputs.size(0))

            outputs = outputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            evaluater.add_batch(targets, outputs[:, 1])
    log = OrderedDict([('val_loss', val_loss.avg),
                       ('val_acc', evaluater.confusion_matrix()[1]),
                       ('val_f1', evaluater.f1_score()),
                       ('val_auc_roc', evaluater.auc_roc())])
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

    if args.network == 'U_Net':
        net = models.UNetFamily.U_Net(img_ch=3, output_ch=2).to(device)
    elif args.network == 'Dense_Unet':
        net = models.UNetFamily.Dense_Unet(in_chan=3, out_chan=2).to(device)
    elif args.network == 'R2AttU_Net':
        net = models.UNetFamily.R2AttU_Net(img_ch=3, output_ch=2).to(device)
    elif args.network == 'AttU_Net':
        net = models.UNetFamily.AttU_Net(img_ch=3, output_ch=2).to(device)
    elif args.network == 'R2U_Net':
        net = models.UNetFamily.R2U_Net(img_ch=3, output_ch=2).to(device)
    elif args.network == 'LadderNet':
        net = models.LadderNet(inplanes=3, num_classes=2, layers=3, filters=16).to(device)
    elif args.network == 'Swin_Unet':
        net_config = get_config(args)
        net = models.Swin_Unet(config=net_config,
                               img_size=args.img_size,
                               num_classes=args.num_classes).to(device)
        net.load_from(net_config)
    else:
        print('wrong network type. exit.')
        sys.exit(-1)
    print("Total number of parameters: " + str(count_parameters(net)))

    if args.network == 'Swin_Unet':
        log.save_graph(net, torch.randn((1, 3, 224, 224)).to(device).to(
            device=device))  # Save the model structure to the tensorboard file
    else:
        log.save_graph(net, torch.randn((1, 3, 512, 512)).to(device).to(
            device=device))  # Save the model structure to the tensorboard file

    # torch.nn.init.kaiming_normal(net, mode='fan_out')      # Modify default initialization method
    # net.apply(weight_init)

    # criterion = LossMulti(jaccard_weight=0,class_weights=np.array([0.5,0.5]))
    criterion = CrossEntropyLoss2d()  # Initialize loss function

    # create a list of learning rate with epochs
    # lr_epoch = np.array([50, args.N_epochs])
    # lr_value = np.array([0.001, 0.0001])
    # lr_schedule = make_lr_schedule(lr_epoch,lr_value)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    # optimizer = optim.SGD(net.parameters(),lr=lr_schedule[0], momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0)

    print(args)
    train_loader, val_loader = get_dataloader(args)  # create dataloader

    if args.val_on_test:
        print('\033[0;32m===============Validation on Testset!!!===============\033[0m')
        val_tool = Test(args)

    best = {'epoch': 0, 'AUC_roc': 0.5}  # Initialize the best epoch and performance(AUC of ROC)
    trigger = 0  # Early stop Counter
    for epoch in range(args.start_epoch, args.N_epochs + 1):
        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
              (epoch, args.N_epochs, optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))

        # train stage
        train_log = train(train_loader, net, criterion, optimizer, device)
        # val stage
        if not args.val_on_test:
            val_log = val(val_loader, net, criterion, device)
        else:
            val_tool.inference(net)
            val_log = val_tool.val()

        log.update(epoch, train_log, val_log)  # Add log information
        lr_scheduler.step()

        # Save checkpoint of latest and best model.
        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['val_auc_roc'] > best['AUC_roc']:
            print('\033[0;33mSaving best model!\033[0m')
            torch.save(state, join(save_path, 'best_model.pth'))
            best['epoch'] = epoch
            best['AUC_roc'] = val_log['val_auc_roc']
            trigger = 0
        print('Best performance at Epoch: {} | AUC_roc: {}'.format(best['epoch'], best['AUC_roc']))
        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
