#!/bin/sh

# uncomment for slurm
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name='pspnet'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB

module purge

# Replace with your NetID
NETID=mw3706
cd /scratch/${NETID}/dim/Deep_Image_Matting_Reproduce/pspnet/


singularity exec --nv --overlay /home/mw3706/pytorch-1.1.0.ext3:ro \
        /scratch/work/public/singularity/cuda9.0-cudnn7-devel-ubuntu16.04-20201127.sif \
        /bin/bash -c "
source /ext3/env.sh  # pytorch 1.4.0 env

export PYTHONPATH=./
python -u tool/demo.py \
  --config=config/portrait/portrait_pspnet101.yaml \
  --image=data/portrait/clip_img/1803151818/clip_00000001/1803151818-00001000.jpg \
  TEST.scales '[1.0]'

# export PYTHONPATH=./
# python -u tool/demo.py \
#  --config=config/portrait/portrait_pspnet101.yaml \
#  --image=data/portrait/list/training.txt \
#  TEST.scales '[1.0]'

# export PYTHONPATH=./
# python -u tool/demo.py \
#    --config=config/portrait/portrait_pspnet101.yaml \
#    --image=data/portrait/list/validation.txt \
#    TEST.scales '[1.0]'
"
