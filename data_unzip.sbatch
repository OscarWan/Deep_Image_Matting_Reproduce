#!/bin/sh -e
#SBATCH --time=24:00:00
#SBATCH --job-name='data_unzip'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=25GB

module purge
module load anaconda3/2020.07
module load cuda/11.1.74
module load gcc/10.2.0


## Replace with your NetID
NETID=mw3706
source activate python36
cd /scratch/${NETID}/dim/data


# wget http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar
# tar -xvf VOCtrainval_14-Jul-2008.tar
# touch VOCtrain
# wget http://images.cocodataset.org/zips/train2014.zip
# unzip -a train2014.zip
# touch COCOtrain
# unzip -a Adobe_Deep_Matting_Dataset.zip
# touch Adobe

mkdir alphamatting
cd alphamatting
# wget http://www.alphamatting.com/datasets/zip/input_lowres.zip
# unzip -a input_lowres.zip
# touch evalinput
wget http://www.alphamatting.com/datasets/zip/trimap_lowres.zip
unzip -a trimap_lowres.zip
touch evaltrimap
