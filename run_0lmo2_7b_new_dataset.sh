#!/bin/bash -l
# $ -P tin-lab      
#$ -l h_rt=8:00:00  
#$ -l gpus=1        
#$ -pe omp 4        # Request 4 CPU cores
#$ -l gpu_memory=60G     
#$ -N olmo2_7b_new_dataset     # job name
#$ -m bea           # Send you an email when the job starts and ends
#$ -j y             # including both print statements and error messages

module load miniconda
conda activate /projectnb/tin-lab/audrey/champ/hf_env

export HF_HOME=/projectnb/tin-lab/cache/huggingface
export TRANSFORMERS_CACHE=/projectnb/tin-lab/cache/huggingface

python cs505am/students/amao/src/olmo2_7b_new_dataset.py