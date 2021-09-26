import sys,os,shutil,glob,cv2
import numpy as np

subset = sys.argv[1]  # val or test
save_dir = sys.argv[2]

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

gt_dir = '/media/ubuntu/Data/gd_newAug5_Rot0_4classes_bak/refine_line_v1_512_512'

results_dir = [
    '/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/fcn_unet_s5-d16_256x256_40k_gd_line512new_lr0.01',
    '/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/deeplabv3_unet_s5-d16_256x256_40k_gd_line512new_lr0.01',
    '/media/ubuntu/Temp/gd/mmsegmentation/work_dirs/pspnet_unet_s5-d16_256x256_40k_gd_line512new_lr0.01',
    '/media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_bs=24_lr=0.1_tiny_224_4_',
    '/media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V2_bs=24_lr=0.2_tiny_224_4_',
]

with open(os.path.join(gt_dir, subset + '.txt'), 'r') as fp:
    lines = fp.readlines()
prefixes = [line.strip() for line in lines]


def existed(name):
    return os.path.exists(name)

count = 0
for prefix in prefixes:
    name1 = os.path.join(gt_dir, subset, 'images', prefix + '.jpg')
    name2 = os.path.join(gt_dir, subset, 'annotations', prefix + '.png')
    name3 = os.path.join(results_dir[0], subset, prefix + '_binary.png')
    name4 = os.path.join(results_dir[1], subset, prefix + '_binary.png')
    name5 = os.path.join(results_dir[2], subset, prefix + '_binary.png')
    name6 = os.path.join(results_dir[3], subset, prefix + '_binary.png')
    name7 = os.path.join(results_dir[4], subset, prefix + '_binary.png')

    exists = [existed(name1), existed(name2), existed(name3),
              existed(name4), existed(name5), existed(name6), existed(name7)]
    print(prefix, exists)
    if np.all(exists):
        im1 = cv2.imread(name1)
        im2 = cv2.imread(name2) * 255
        im3 = cv2.imread(name3)
        im4 = cv2.imread(name4)
        im5 = cv2.imread(name5)
        im6 = cv2.imread(name6)
        im7 = cv2.imread(name7)
        H,W = im1.shape[:2]
        blank = np.zeros((H, 5, 3), dtype=np.uint8)

        if subset == 'val':
            # img = np.concatenate([im1, im2, im3, im4, im5, im6, im7], axis=1)
            img = np.concatenate([im1, blank, im2,blank, im5,blank, im6,blank, im7], axis=1)
        else:
            img = np.concatenate([im1, blank,im3, blank,im4, blank,im5, blank,im6, blank,im7], axis=1)
        cv2.imwrite(os.path.join(save_dir, prefix+'_final.png'), img)
    else:
        print('no', prefix)
        count += 1

print(len(prefixes), count)










