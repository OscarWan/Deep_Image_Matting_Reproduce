import os
import numpy as np

if not os.path.exists("list"):
    os.makedirs("list")

image_list, matting_list = [], []
first_level = os.listdir("clip_img")
for folder in first_level:
    second_level = os.listdir(os.path.join("clip_img", folder))
    for subfolder in second_level:
        temp_list = os.listdir(os.path.join("clip_img", os.path.join(folder, subfolder)))
        for image in temp_list:
            image_list.append("clip_img/"+folder+'/'+subfolder+'/'+image)

first_level = os.listdir("matting")
for folder in first_level:
    second_level = os.listdir(os.path.join("matting", folder))
    for subfolder in second_level:
        if subfolder == '._matting_00000000':
            continue
        temp_list = os.listdir(os.path.join("matting", os.path.join(folder, subfolder)))
        for matting in temp_list:
            matting_list.append("matting/"+folder+'/'+subfolder+'/'+matting)

num = len(matting_list)
ind_list = np.arange(num)
np.random.shuffle(ind_list)
train_ind = ind_list[:int(num*0.8)]
test_ind = ind_list[int(num*0.8):]

with open('list/training.txt', 'w') as file:
    for i in range(len(train_ind)):
        file.write(image_list[i] + ' ' + matting_list[i] + '\n')

with open('list/validation.txt', 'w') as file:
    for i in range(len(test_ind)):
        file.write(image_list[i] + ' ' + matting_list[i] + '\n')
