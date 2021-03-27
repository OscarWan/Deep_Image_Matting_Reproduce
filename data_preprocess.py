import shutil
import math
import cv2 as cv
import numpy as np
import os

def process(fg_path, a_path, bg_path, out_path, dataset, folder, bg_file_path,\
            fg_file_path, is_train):
    # copy foreground files from downloaded folder to self-designed folder
    if not os.path.exists(fg_path):
        os.makedirs(fg_path)

    if is_train:
        adobe_path = [folder + 'Adobe-licensed images/fg', folder + 'Other/fg']
    else:
        adobe_path = [folder + 'Adobe-licensed images/fg']

    for old_folder in adobe_path:
        fg_files = os.listdir(old_folder)
        for fg_file in fg_files:
            src_path = os.path.join(old_folder, fg_file)
            dest_path = os.path.join(fg_path, fg_file)
            shutil.copy(src_path, dest_path)

    # copy background files from downloaded folder to self-designed folder
    if not os.path.exists(bg_path):
        with open(os.path.join(folder, bg_file_path)) as f:
            bg_names = f.read().splitlines()
        os.makedirs(bg_path)
        for bg_name in bg_names:
            src_path = os.path.join(dataset, bg_name)
            dest_path = os.path.join(bg_path, bg_name)
            shutil.copy(src_path, dest_path)

    # copy alpha files from downloaded folder to self-designed folder
    if not os.path.exists(a_path):
        os.makedirs(a_path)

    if is_train:
        adobe_path = [folder + 'Adobe-licensed images/alpha', folder + 'Other/alpha']
    else:
        adobe_path = [folder + 'Adobe-licensed images/alpha']

    for old_folder in adobe_path:
        a_files = os.listdir(old_folder)
        for a_file in a_files:
            src_path = os.path.join(old_folder, a_file)
            dest_path = os.path.join(a_path, a_file)
            shutil.copy(src_path, dest_path)

    # Composite to make new training data in the out_path
    if is_train:
        print('Doing training composition...')
    else:
        print('Doing test composition...')

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    bg_files = bg_names
    with open(os.path.join(folder, fg_file_path)) as f:
        fg_files = f.read().splitlines()

    if is_train:
        num_bgs = 100
    else:
        num_bgs = 20
    num_samples = len(fg_files) * num_bgs
    print('num_samples: ' + str(num_samples))
    for fcount in range(len(fg_files)):
        im_name = fg_files[fcount]
        bcount = fcount * num_bgs
        for i in range(num_bgs):
            bg_name = bg_files[bcount]
            im = cv.imread(fg_path + im_name)
            a = cv.imread(a_path + im_name, 0)
            h, w = im.shape[:2]
            bg = cv.imread(bg_path + bg_name)
            bh, bw = bg.shape[:2]
            wratio = w / bw
            hratio = h / bh
            ratio = wratio if wratio > hratio else hratio
            if ratio > 1:
                bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio),\
                 math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

            fg = np.array(im, np.float32)
            bg = np.array(bg[0:h, 0:w], np.float32)
            alpha = np.zeros((h, w, 1), np.float32)
            alpha[:, :, 0] = a / 255.
            comp = alpha * fg + (1 - alpha) * bg
            out = comp.astype(np.uint8)
            filename = out_path + str(fcount) + '_' + str(bcount) + '.png'
            cv.imwrite(filename, out)
            bcount += 1

###------ Training data preprocessing ------###
fg_path = '../data/fg/'
# path to provided alpha mattes
a_path = '../data/mask/'
# Path to background images (MSCOCO)
bg_path = '../data/bg/'
# Path to folder where you want the composited images to go
out_path = '../data/merged/'
# MSCOCO data path
mscoco_path = '../data/train2017/'
# Adobe training path
train_folder = '../data/Combined_Dataset/Training_set/'
# file list
bg_file_path = 'training_bg_names.txt'
fg_file_path = 'training_fg_names.txt'

print('Moving training foreground, background, alpha images to self-designed folders...')
process(fg_path, a_path, bg_path, out_path, mscoco_path, train_folder,\
        bg_file_path, fg_file_path, True)

###------ Test Data preprocessing ------###
# path to foreground images
fg_test_path = '../data/fg_test/'
# path to provided alpha mattes
a_test_path = '../data/mask_test/'
# Path to background images (MSCOCO)
bg_test_path = '../data/bg_test/'
# Path to folder where you want the composited images to go
out_test_path = '../data/merged_test/'
# VOC data Path
voc_path = '../data/VOCdevkit/VOC2012/JPEGImages/'
# Adobe test path
test_folder = '../data/Combined_Dataset/Test_set/'
# file list
bg_file_test_path = 'test_bg_names.txt'
fg_file_test_path = 'test_fg_names.txt'

print('Moving test foreground, background, alpha images to self-designed folders...')
process(fg_test_path, a_test_path, bg_test_path, out_test_path, voc_path,\
        test_folder, bg_file_test_path, fg_file_test_path, False)
