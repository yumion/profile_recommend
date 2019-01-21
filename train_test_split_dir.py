import os, shutil
from glob import glob
import numpy as np

base_dir = 'dataset'

# make train directory
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
# make test directory
test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# make class directory
clsnames = ['autumn', 'spring', 'summer', 'winter']
for clsname in clsnames:
    # train
    train_cls_dir = os.path.join(train_dir, clsname)
    os.makedirs(train_cls_dir, exist_ok=True)
    # test
    test_cls_dir = os.path.join(test_dir, clsname)
    os.makedirs(test_cls_dir, exist_ok=True)

# move datset to train/test
for clsname in clsnames:
    fnames = glob('{parent}/{clsname}/*'.format(parent=base_dir, clsname=clsname))
    np.random.shuffle(fnames)
    fnames = [fname.split(base_dir+'/')[-1] for fname in fnames]
    # number of train
    num_data = len(glob(base_dir+'/{}/*'.format(clsname)))
    train_size = int(num_data*0.9)
    # train
    for fname in fnames[:train_size]:
        src = os.path.join(base_dir, fname)
        dst = os.path.join(train_dir, fname)
        shutil.copyfile(src, dst)
    # test
    for fname in fnames[train_size:]:
        src = os.path.join(base_dir, fname)
        dst = os.path.join(test_dir, fname)
        shutil.copyfile(src, dst)
