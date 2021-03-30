import math
import time
from multiprocessing import Pool
import cv2 as cv
import numpy as np
import tqdm

# path to foreground images
fg_test_path = '../data/fg_test/'
# path to provided alpha mattes
a_test_path = '../data/mask_test/'
# Path to background images (MSCOCO)
bg_test_path = '../data/bg_test/'
# Path to folder where you want the composited images to go
out_test_path = '../data/merged_test/'
# VOC data Path
voc_path = '../data/VOCdevkit/VOC2008/JPEGImages/'
# Adobe test path
test_folder = '../data/Combined_Dataset/Test_set/'
# file list
bg_file_test_path = 'test_bg_names.txt'
fg_file_test_path = 'test_fg_names.txt'

num_bgs = 20

with open(os.path.join(test_folder, bg_file_test_path)) as f:
    bg_files = f.read().splitlines()
with open(os.path.join(test_folder, fg_file_test_path)) as f:
    fg_files = f.read().splitlines()


def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg = np.array(bg[0:h, 0:w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)
    return comp

def process(im_name, bg_name, fcount, bcount):
    im = cv.imread(fg_test_path + im_name)
    a = cv.imread(a_test_path + im_name, 0)
    h, w = im.shape[:2]
    bg = cv.imread(bg_test_path + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    out = composite4(im, bg, a, w, h)
    filename = out_path + str(fcount) + '_' + str(bcount) + '.png'
    cv.imwrite(filename, out)


def process_one_fg(fcount):
    print(fcount)
    im_name = fg_files[fcount]
    bcount = fcount * num_bgs

    for i in range(num_bgs):
        bg_name = bg_files[bcount]
        process(im_name, bg_name, fcount, bcount)
        bcount += 1

def do_composite_test():
    print('Doing composite test data...')

    num_samples = len(fg_files) * num_bgs
    print('num_samples: ' + str(num_samples))

    start = time.time()
    with Pool(processes=16) as p:
        max_ = len(fg_files)
        print('num_fg_files: ' + str(max_))
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(process_one_fg, range(0, max_)))):
                pbar.update()

    end = time.time()
    elapsed = end - start
    print('elapsed: {} seconds'.format(elapsed))