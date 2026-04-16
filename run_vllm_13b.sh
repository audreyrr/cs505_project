#!/bin/bash -l
# $ -P tin-lab      
#$ -l h_rt=11:00:00  
#$ -l gpus=1        
#$ -pe omp 4        # Request 4 CPU cores
#$ -l gpu_memory=60G     
#$ -N olmo2_13b     # job name
#$ -m bea           # Send you an email when the job starts and ends
#$ -j y             # including both print statements and error messages

module load miniconda
conda activate /projectnb/tin-lab/audrey/champ/hf_env
export HF_HOME=/projectnb/tin-lab/cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /projectnb/cs505am/students/amao  

python src/olmo2_vllm_3sizes.py --model_size 13b --gpu_memory_utilization 0.7 --tensor_parallel_size 1 --batch_size 4 --resume

