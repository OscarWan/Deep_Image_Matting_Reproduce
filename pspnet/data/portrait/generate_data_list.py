import os
import numpy as np

if not os.path.exists("list"):
    os.makedirs("list")

image_list, matting_list = [], []
first_level = os.listdir("clip_img")
for folder in first_level:
    second_level = os.listdir(os.path.join("clip_img", folder))
    for subfolder in second_level:
        image_list.append(os.listdir(os.path.join("clip_img", os.path.join(folder, subfolder))))

boo = True
first_level = os.listdir("matting")
for folder in first_level:
    second_level = os.listdir(os.path.join("matting", folder))
    for subfolder in second_level:
        if boo:
            print('folder=', folder)
            print('subfolder=', subfolder)
        matting_list.append(os.listdir(os.path.join("matting", os.path.join(folder, subfolder))))

if len(image_list) == len(matting_list):
    print('length same')
else:
    print('not the same')
