import os, shutil, re
import numpy as np
from glob import glob

src_dir = 'raw_dataset'

# root dir of dataset
to_dir = 'rank_dataset_downsampling'
os.makedirs(to_dir, exist_ok=True)
# make train directory
train_dir = os.path.join(to_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
# make test directory
test_dir = os.path.join(to_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# get all files
all_files = glob(src_dir+'/*/*')
regex = re.compile(r'like(.*)_views(.*).jpg')

# rankごとに分ける
num_rank = 3
rank_top = []
rank_mid = []
rank_bottom = []
for fname in all_files:
    mo = regex.search(fname) # search like num
    likes = mo.group(1)
    likes = int(likes)
    if likes > 200:
        rank_top.append(fname)
    elif likes > 100 and likes <= 200:
        rank_mid.append(fname)
    elif likes <= 100:
        rank_bottom.append(fname)


# シャッフル
np.random.shuffle(rank_top)
np.random.shuffle(rank_mid)
np.random.shuffle(rank_bottom)

# train/test split
test_size_per_class = int(len(rank_top)*0.2)
test_size_per_class
len(rank_top)
len(rank_mid)
len(rank_bottom)

# top rank
# test
for fname in rank_top[:test_size_per_class]:
    dst = os.path.join(test_dir, 'top')
    os.makedirs(dst, exist_ok=True)
    fdst = os.path.join(dst, fname.split('/')[-1])
    shutil.copyfile(fname, fdst)
# train
for fname in rank_top[test_size_per_class:len(rank_top)]:
    dst = os.path.join(train_dir, 'top')
    os.makedirs(dst, exist_ok=True)
    fdst = os.path.join(dst, fname.split('/')[-1])
    shutil.copyfile(fname, fdst)

# middle rank
# test
for fname in rank_mid[:test_size_per_class]:
    dst = os.path.join(test_dir, 'middle')
    os.makedirs(dst, exist_ok=True)
    fdst = os.path.join(dst, fname.split('/')[-1])
    shutil.copyfile(fname, fdst)
# train
for fname in rank_mid[test_size_per_class:len(rank_top)]:
    dst = os.path.join(train_dir, 'middle')
    os.makedirs(dst, exist_ok=True)
    fdst = os.path.join(dst, fname.split('/')[-1])
    shutil.copyfile(fname, fdst)

# bottom rank
# test
for fname in rank_bottom[:test_size_per_class]:
    dst = os.path.join(test_dir, 'bottom')
    os.makedirs(dst, exist_ok=True)
    fdst = os.path.join(dst, fname.split('/')[-1])
    shutil.copyfile(fname, fdst)
# train
for fname in rank_bottom[test_size_per_class:len(rank_top)]:
    dst = os.path.join(train_dir, 'bottom')
    os.makedirs(dst, exist_ok=True)
    fdst = os.path.join(dst, fname.split('/')[-1])
    shutil.copyfile(fname, fdst)

print('finish')
