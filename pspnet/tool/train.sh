#!/bin/sh

# uncomment for slurm
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name='data_preprocess'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50GB

module purge

# Replace with your NetID
NETID=mw3706
cd /scratch/${NETID}/dim

singularity exec --nv \
	    --overlay /scratch/mw3706/dim/overlay-5GB-200K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "
cd Deep_Image_Matting_Reproduce/pspnet
export PYTHONPATH=./
eval "$(conda shell.bash hook)"
source /ext3/env.sh  # pytorch 1.4.0 env
PYTHON=python

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool/train.sh tool/train.py tool/test.sh tool/test.py ${config} ${exp_dir}

export PYTHONPATH=./
$PYTHON -u ${exp_dir}/train.py \
  --config=${config} \
  2>&1 | tee ${model_dir}/train-$now.log

$PYTHON -u ${exp_dir}/test.py \
  --config=${config} \
  2>&1 | tee ${result_dir}/test-$now.log
"
