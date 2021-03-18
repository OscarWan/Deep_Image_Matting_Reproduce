#!/bin/sh -e
#SBATCH --time=24:00:00
#SBATCH --job-name='data_preprocess'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB

module purge
module load anaconda3/2020.07
module load cuda/11.1.74
module load gcc/10.2.0


# Replace with your NetID
NETID=mw3706
source activate /scratch/${NETID}/dim/env
cd /scratch/${NETID}/dim/data

python ../Deep_Image_Matting_Reproduce/data_preprocess.py
