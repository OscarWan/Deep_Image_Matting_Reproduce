#!/bin/sh -e
#SBATCH --time=24:00:00
#SBATCH --job-name='training'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1

module purge

# Replace with your NetID
NETID=mw3706
cd /scratch/${NETID}/dim

singularity exec --nv --overlay /home/mw3706/pytorch-1.1.0.ext3:ro \
		/scratch/work/public/singularity/cuda9.0-cudnn7-devel-ubuntu16.04-20201127.sif \
		/bin/bash -c "
source /ext3/env.sh
cd Deep_Image_Matting_Reproduce
python train.py
exit
"
