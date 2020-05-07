#!/bin/bash

#SBATCH -A p_da_mlforcode
#SBATCH --partition=ml
#SBATCH --nodes=1
#SBATCH --mincpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4000

module load modenv/ml
module load TensorFlow

source ~/env/bin/activate && $@
