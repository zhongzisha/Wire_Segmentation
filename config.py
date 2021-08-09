import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # in/out
    parser.add_argument('--outf', default='./experiments',
                        help='trained model will be saved at here')
    parser.add_argument('--save', default='UNet_vessel_seg',
                        help='save name of experiment in args.outf directory')

    # data
    parser.add_argument('--data_root', default='', type=str)
    parser.add_argument('--dataset_type', default='', type=str)
    parser.add_argument('--train_subset', default='train', type=str)
    parser.add_argument('--val_subset', default='val', type=str)
    parser.add_argument('--test_subset', default='val', type=str)
    parser.add_argument('--subset', default='', type=str)
    parser.add_argument('--network', default='', type=str)
    parser.add_argument('--train_patch_height', default=512, type=int)
    parser.add_argument('--train_patch_width', default=512, type=int)
    parser.add_argument('--N_patches', default=200000, type=int,
                        help='Number of training image patches')
    parser.add_argument('--val_ratio', default=0.2,
                        help='The ratio of the validation set in the training set')
    # model parameters
    parser.add_argument('--in_channels', default=3, type=int,
                        help='input channels of model')
    parser.add_argument('--classes', default=2, type=int,
                        help='output channels of model')

    # training
    parser.add_argument('--N_epochs', default=50, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')
    parser.add_argument('--early-stop', default=6, type=int,
                        help='early stopping')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate')
    parser.add_argument('--val_on_test', default=False, type=bool,
                        help='Validation on testset')

    # for pre_trained checkpoint
    parser.add_argument('--start_epoch', default=1,
                        help='Start epoch')
    parser.add_argument('--pre_trained', default=None,
                        help='(path of trained _model)load trained model to continue train')

    # testing
    parser.add_argument('--test_patch_height', default=512)
    parser.add_argument('--test_patch_width', default=512)
    parser.add_argument('--stride_height', default=512)
    parser.add_argument('--stride_width', default=512)

    # hardware setting
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use GPU calculating')

    # from Swin-Unet
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--num_classes', type=int,
                        default=2, help='output channel of network')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    args = parser.parse_args()

    return args
