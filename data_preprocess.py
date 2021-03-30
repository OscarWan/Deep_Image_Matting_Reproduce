import shutil
import os

from multiprocess_train_composite import do_composite
from multiprocess_test_composite import do_composite_test

def move_file(fg_path, a_path, bg_path, out_path, dataset, folder, bg_file_path,\
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

    # Jumping to related files to do composition
    if is_train:
        do_composite()
    else:
        do_composite_test()

if __name__ == '__main__':
    # ###------ Training data preprocessing ------###
    # fg_path = '../data/fg/'
    # # path to provided alpha mattes
    # a_path = '../data/mask/'
    # # Path to background images (MSCOCO)
    # bg_path = '../data/bg/'
    # # Path to folder where you want the composited images to go
    # out_path = '../data/merged/'
    # # MSCOCO data path
    # mscoco_path = '../data/train2014/'
    # # Adobe training path
    # train_folder = '../data/Combined_Dataset/Training_set/'
    # # file list
    # bg_file_path = 'training_bg_names.txt'
    # fg_file_path = 'training_fg_names.txt'
    #
    # print('Moving training foreground, background, alpha images to self-designed folders...')
    # move_file(fg_path, a_path, bg_path, out_path, mscoco_path, train_folder,\
    #         bg_file_path, fg_file_path, True)

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
    voc_path = '../data/VOCdevkit/VOC2008/JPEGImages/'
    # Adobe test path
    test_folder = '../data/Combined_Dataset/Test_set/'
    # file list
    bg_file_test_path = 'test_bg_names.txt'
    fg_file_test_path = 'test_fg_names.txt'

    print('Moving test foreground, background, alpha images to self-designed folders...')
    move_file(fg_test_path, a_test_path, bg_test_path, out_test_path, voc_path,\
            test_folder, bg_file_test_path, fg_file_test_path, False)
