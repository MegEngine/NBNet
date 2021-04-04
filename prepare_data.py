import os
from glob import glob
import cv2
import numpy as np
import h5py as h5
import argparse
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
parser = argparse.ArgumentParser(prog='SIDD Train dataset Generation')
# The orignal SIDD images: /ssd1t/SIDD/
parser.add_argument('--data_dir', default='/data/sidd/SIDD_Medium_Srgb/Data', type=str, metavar='PATH',
                                      help="path to save the training set of SIDD, (default: None)")
parser.add_argument('--tar_dir', default='/data/sidd/train',type=str, help='Directory for image patches')
args = parser.parse_args()
tar = args.tar_dir

noisy_patchDir = os.path.join(tar, 'input')
clean_patchDir = os.path.join(tar, 'groundtruth')

if os.path.exists(tar):
    os.system("rm -r {}".format(tar))

os.makedirs(noisy_patchDir)
os.makedirs(clean_patchDir)
path_all_noisy = glob(os.path.join(args.data_dir, '**/*NOISY*.PNG'), recursive=True)
path_all_noisy = sorted(path_all_noisy)
path_all_gt = [x.replace('NOISY', 'GT') for x in path_all_noisy]
print('Number of big images: {:d}'.format(len(path_all_gt)))

print('Training: Split the original images to small ones!')
path_h5 = os.path.join(args.data_dir, 'small_imgs_train.hdf5')
if os.path.exists(path_h5):
    os.remove(path_h5)
pch_size = 512
stride = 512-128
num_patch = 0
C = 3

def save_files(ii):
    im_noisy_int8 = cv2.imread(path_all_noisy[ii])
    H, W, _ = im_noisy_int8.shape
    im_gt_int8 = cv2.imread(path_all_gt[ii])
    ind_H = list(range(0, H-pch_size+1, stride))
    if ind_H[-1] < H-pch_size:
        ind_H.append(H-pch_size)
    ind_W = list(range(0, W-pch_size+1, stride))
    if ind_W[-1] < W-pch_size:
        ind_W.append(W-pch_size)
    count = 1
    for start_H in ind_H:
        for start_W in ind_W:
            pch_noisy = im_noisy_int8[start_H:start_H+pch_size, start_W:start_W+pch_size, ]
            pch_gt = im_gt_int8[start_H:start_H+pch_size, start_W:start_W+pch_size, ]
            cv2.imwrite(os.path.join(noisy_patchDir, '{}_{}.png'.format(ii+1,count+1)), pch_noisy)
            cv2.imwrite(os.path.join(clean_patchDir, '{}_{}.png'.format(ii+1,count+1)), pch_gt)
            count += 1
Parallel(n_jobs=10)(delayed(save_files)(i) for i in tqdm(range(len(path_all_gt))))
print('Finish!\n')

