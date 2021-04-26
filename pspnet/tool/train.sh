#!/bin/sh

# uncomment for slurm
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --job-name='pspnet'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB

module purge

# Replace with your NetID
NETID=mw3706
cd /scratch/${NETID}/dim/Deep_Image_Matting_Reproduce/pspnet

cp tool/train.sh tool/train.py tool/test.sh tool/test.py \
        config/voc2012/voc2012_psp101.yaml exp/voc2012/pspnet101

singularity exec --nv --overlay /home/mw3706/pytorch-1.1.0.ext3:ro \
        /scratch/work/public/singularity/cuda9.0-cudnn7-devel-ubuntu16.04-20201127.sif \
        /bin/bash -c "
source /ext3/env.sh  # pytorch 1.4.0 env

export PYTHONPATH=./
python -u exp/voc2012/pspnet101/train.py \
  --config=config/voc2012/portrait_psp101.yaml \
  2>&1 | tee exp/voc2012/pspnet101/model/train_log.txt


python -u exp/voc2012/pspnet101/test.py \
  --config=config/voc2012/portrait_psp101.yaml \
  2>&1 | tee exp/voc2012/pspnet101/result/test_log.txt
"
