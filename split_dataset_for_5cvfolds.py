import sys,os,glob,shutil
import numpy as np
from sklearn.model_selection import KFold


if __name__ == '__main__':
    data_root = '/media/ubuntu/SSD/refine_line_v1'
    img_filenames = glob.glob(os.path.join(data_root, 'images', '*.jpg'))

    for split in range(5):
        save_root = os.path.join(data_root, 'split_{}'.format(split))
        indices = np.arange(len(img_filenames))
        np.random.shuffle(indices)
        val_indices = indices[:332]
        train_indices = indices[332:]

        train_prefixes = [
            os.path.basename(img_filenames[ind]).replace('.jpg', '') + '\n'
            for ind in train_indices
        ]
        val_prefixes = [
            os.path.basename(img_filenames[ind]).replace('.jpg', '') + '\n'
            for ind in val_indices
        ]

        for subset in ['train', 'val']:
            os.makedirs(os.path.join(save_root, subset), exist_ok=True)
            os.system('ln -sf {}/images {}/{}/images'.format(data_root, save_root, subset))
            os.system('ln -sf {}/annotations {}/{}/annotations'.format(data_root, save_root, subset))
        with open(os.path.join(save_root, 'train.txt'), 'w') as fp:
            fp.writelines(train_prefixes)
        with open(os.path.join(save_root, 'val.txt'), 'w') as fp:
            fp.writelines(val_prefixes)





















