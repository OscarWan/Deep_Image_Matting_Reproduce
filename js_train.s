#!/bin/sh -e
#SBATCH --time=24:00:00
#SBATCH --job-name='training'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2020.07
module load cuda/11.1.74
module load gcc/10.2.0


# Replace with your NetID
NETID=mw3706
source activate python36
cd /scratch/${NETID}/dim/Deep_Image_Matting_Reproduce

# python data_gen.py
python train.py
