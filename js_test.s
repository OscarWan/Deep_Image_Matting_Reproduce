#!/bin/sh -e
#SBATCH --time=24:00:00
#SBATCH --job-name='testing'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50GB
#SBATCH --gres=gpu:2

module purge

# Replace with your NetID
NETID=mw3706
cd /scratch/${NETID}/dim

singularity exec --nv \
	    --overlay /scratch/mw3706/dim/overlay-5GB-200K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "
source /ext3/env.sh
cd Deep_Image_Matting_Reproduce
python test.py
exit
"
