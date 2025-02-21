#!/bin/bash

#SBATCH -J torchMS2
#SBATCH -o logs/sbatch_out/%x.%A_%a.%N.out
#SBATCH -e logs/sbatch_out/%x.%A_%a.%N.gerr
#SBATCH -D ./
#SBATCH --get-user-env


#SBATCH --partition=exbio-gpu            # compms-cpu-small | shared-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --tasks-per-node=1

#SBATCH --export=NONE
#SBATCH --time=96:00:00
##SBATCH --array=1-100%3

export CUDA_VERSION=11.8        # 11.8 | 12.2
export CUDNN_VERSION=8.9.7.29

export TF_GPU_ALLOCATOR=cuda_malloc_async

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_ms2
# export HF_DATASETS_CACHE="/cmnfs/proj/prosit_astral"
# export HF_DATASETS_CACHE="/cmnfs/proj/prosit_astral/datasets"

# export HF_HOME='/cmnfs/proj/prosit/ptms/huggingface'
# export HF_DATASETS_CACHE='/cmnfs/proj/prosit/ptms/huggingface/datasets'

python -u train_model.py &> ./logs/outputs/$1.log
