#!/bin/bash
#SBATCH --job-name=llama-tab
#SBATCH --mail-user=jianyuanzhong@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:rtx3090:1
#SBATCH -a 1-4
#SBATCH --output=array_job_%A_%a.out
#SBATCH --error=array_job_%A_%a.err

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# change cuda version to 11
export PATH=$PATH:/usr/local/cuda-11/bin
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64

echo "Driver Cuda Version"
nvidia-smi

echo "Local Cuda Version"
nvcc -V

# activate conda env
echo "activate conda env"
conda init bash
source activate tabcure

if [ $SLURM_ARRAY_TASK_ID == 1 ]; then
    python pipline_LLM.py --config configs/adult.yaml
    python pipline_LLM.py --config configs/adult.yaml --test
fi

if [ $SLURM_ARRAY_TASK_ID == 2 ]; then
    python pipline_LLM.py --config configs/heloc.yaml
    python pipline_LLM.py --config configs/heloc.yaml --test
fi

if [ $SLURM_ARRAY_TASK_ID == 3 ]; then
    python pipline_LLM.py --config configs/travel.yaml
    python pipline_LLM.py --config configs/travel.yaml --test
fi

if [ $SLURM_ARRAY_TASK_ID == 4 ]; then
    python pipline_LLM.py --config configs/adult-gpt-neo.yaml
    python pipline_LLM.py --config configs/adult-gpt-neo.yaml --test
fi
