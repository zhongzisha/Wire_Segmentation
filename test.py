import joblib, copy
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch, sys
from tqdm import tqdm

from collections import OrderedDict
from lib.visualize import save_img, group_images, concat_result
import os
import argparse
from lib.logger import Logger, Print_Logger
from lib.extract_patches import *
from os.path import join
from lib.dataset import TestDataset
from lib.metrics import Evaluate
import models
from lib.common import setpu_seed, dict_round
from config import parse_args
from lib.pre_processing import my_PreProc

setpu_seed(2021)


class Test():
    def __init__(self, args):
        self.args = args
        assert (args.stride_height <= args.test_patch_height and args.stride_width <= args.test_patch_width)
        # save path
        self.path_experiment = join(args.outf, args.save)

        self.patches_imgs_test, self.test_imgs, self.test_masks, self.new_height, self.new_width = get_data_test_overlap(
            data_root=args.data_root,
            subset=args.test_subset,
            patch_height=args.test_patch_height,
            patch_width=args.test_patch_width,
            stride_height=args.stride_height,
            stride_width=args.stride_width
        )
        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # Inference prediction process
    def inference(self, net):
        net.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = inputs.cuda()
                outputs = net(inputs)
                outputs = outputs[:, 1].data.cpu().numpy()
                preds.append(outputs)
        predictions = np.concatenate(preds, axis=0)
        self.pred_patches = np.expand_dims(predictions, axis=1)

    # Evaluate ate and visualize the predicted images
    def evaluate(self):
        print('evaluating 1...')
        if self.args.test_subset == 'test2':
            self.pred_imgs = recompone_overlap(
                self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
            ## restore to original dimensions
            self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]
        else:
            self.pred_imgs = self.pred_patches
        # predictions only inside the FOV
        y_scores = self.pred_patches
        y_true = self.test_masks
        print('evaluating 2...')
        eval = Evaluate(save_path=self.path_experiment)
        print('evaluating 2 done...')
        eval.add_batch(y_true, y_scores)
        log = eval.save_all_result(plot_curve=True, save_name="performance.txt")
        # save labels and probs for plot ROC and PR curve when k-fold Cross-validation
        np.save('{}/result.npy'.format(self.path_experiment), np.asarray([y_true, y_scores]))
        return dict_round(log, 6)

    # save segmentation imgs
    def save_segmentation_result(self):
        with open('%s/%s.txt' % (self.args.data_root, self.args.test_subset)) as fp:
            img_name_list = [line.strip() for line in fp.readlines()]

        # kill_border(self.pred_imgs, self.test_FOVs) # only for visualization
        self.save_img_path = join(self.path_experiment, 'result_img_%s' % self.args.test_subset)
        if not os.path.exists(join(self.save_img_path)):
            os.makedirs(self.save_img_path)
        # self.test_imgs = my_PreProc(self.test_imgs) # Uncomment to save the pre processed image
        for i in range(self.test_imgs.shape[0]):
            total_img = concat_result(self.test_imgs[i], self.pred_imgs[i], self.test_masks[i])
            save_img(total_img, join(self.save_img_path, "Result_" + img_name_list[i] + '.png'))

    # Val on the test set at each epoch
    def val(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        # recover to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        # predictions only inside the FOV
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        confusion, accuracy, specificity, sensitivity, precision = eval.confusion_matrix()
        log = OrderedDict([('val_auc_roc', eval.auc_roc()),
                           ('val_f1', eval.f1_score()),
                           ('val_acc', accuracy),
                           ('SE', sensitivity),
                           ('SP', specificity)])
        return dict_round(log, 6)


if __name__ == '__main__':
    args = parse_args()
    save_path = join(args.outf, args.save)
    sys.stdout = Print_Logger(os.path.join(save_path, 'test_%s_log.txt' % args.test_subset))
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

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
    else:
        print('wrong network type. exit.')
        sys.exit(-1)

    cudnn.benchmark = True

    # Load checkpoint
    print('==> Loading checkpoint...')
    checkpoint = torch.load(join(save_path, 'best_model.pth'))
    net.load_state_dict(checkpoint['net'])

    eval = Test(args)
    eval.inference(net)
    print(eval.evaluate())
    eval.save_segmentation_result()
