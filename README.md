# Introduction
This is my CS capstone at undergraduate level, combining Deep Image Matting (DIM) and Semantic Segmentation. The project includes reproducing classical methods in two areas, and then the attempt to combine two together.

1. Reproduce the algorithm deep image matting by this [paper](https://arxiv.org/abs/1703.03872)
2. Try to replace the trimap results with the results from semantic segmentation.
3. Compared the result with [SOTA model](https://arxiv.org/abs/2011.11961v2)

# Dataset
### MSCOCO Dataset
Download MSCOCO [2017 Train image](https://cocodataset.org/#download)
### PASCAL VOC Dataset
Download VOC2012 challenge [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)

Download [test data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#testdata) for VOC2012 challenge
### Adobe DIM Dataset
Go to the [website](https://sites.google.com/view/deepimagematting#h.p_LwcLE-VLY7-S) to contact the author for the use of the dataset.

# Reproduce DIM
1. Prepare for the training by unziping folders (run [data_unzip.sbatch](https://github.com/OscarWan/Deep_Image_Matting_Reproduce/blob/main/data_unzip.sbatch)), getting file lists, and generalizing figures (run [data_preprocess.sbatch](https://github.com/OscarWan/Deep_Image_Matting_Reproduce/blob/main/data_preprocess.sbatch)).
