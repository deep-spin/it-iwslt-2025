#!/bin/bash
#SBATCH --output=./logs/%A.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4

source ~/mydata/venvs/speech_lm/bin/activate
echo "Python is at:"
echo `whereis python`

source /etc/profile.d/02-lmod.sh
module load cuda

if [ -z "$1" ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

CONFIG_FILE=$1
ACCELERATE_CONFIG=$2
GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps' $CONFIG_FILE | awk '{print $2}')
echo "Using config file: $CONFIG_FILE"
echo "Gradient Accumulation Steps: $GRAD_ACC_STEPS"

# setup SCRATCH only if you are not on an HPC cluster
export SCRATCH="/mnt/home/giuseppe/myscratch"

# set the base directory for the project
export BASEDIR="$SCRATCH/speech_lm"

# setup run-specific envs
# export LOCAL_DATASETS_DIR="/mnt/home/giuseppe/myscratch/speech_lm/datasets"
export TMPDIR="/tmp"
export LOCAL_DATASETS_DIR="/mnt/scratch-artemis/shared/datasets"
export HF_HOME="$SCRATCH/hf_home_speech"
export HF_DATASETS_CACHE="$BASEDIR/hf_datasets_cache"
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

export WANDB_PROJECT=speech_lm
# export WANDB_MODE=offline
export TORCHDYNAMO_VERBOSE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file $ACCELERATE_CONFIG  \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --max_restarts 1 \
    --role \$(hostname -s): \
    --tee 3 \
    "

export CMD="src/train.py --config-file $CONFIG_FILE"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1

