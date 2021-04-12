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
cd /scratch/${NETID}/dim
singularity exec --overlay overlay-5GB-200K.ext3 /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash
source /ext3/env.sh
cd Deep_Image_Matting_Reproduce

# python data_gen.py
python train.py
