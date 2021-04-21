#!/bin/sh

# uncomment for slurm
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name='pspnet'
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=16
#SBATCH --mem=100GB

module purge

# Replace with your NetID
NETID=mw3706
cd /scratch/${NETID}/dim/Deep_Image_Matting_Reproduce/pspnet

singularity exec --nv --overlay /home/mw3706/pytorch-1.1.0.ext3:ro \
        /scratch/work/public/singularity/cuda9.0-cudnn7-devel-ubuntu16.04-20201127.sif \
        /bin/bash -c "
source /ext3/env.sh  # pytorch 1.4.0 env

exp_dir=exp/portrait/psp101
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/portrait/portrait_psp101.yaml
now=$(date +"%Y%m%d_%H%M%S")

cp tool/train.sh ${exp_dir}
cp tool/train.py ${exp_dir}
cp tool/test.sh ${exp_dir}
cp tool/test.py ${exp_dir}
cp ${config} ${exp_dir}

export PYTHONPATH=./
python -u ${exp_dir}/train.py \
  --config=${config} \
  2>&1 | tee ${model_dir}/train-$now.txt

python -u ${exp_dir}/test.py \
  --config=${config} \
  2>&1 | tee ${result_dir}/test-$now.txt
"
