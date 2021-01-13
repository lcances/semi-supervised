#!/bin/bash
# ___________________________________________________________________________________ #
die() {
    printf '%s\n' "$1" >& 2
    exit 1
}

parse_long() {
    if [ "$1" ]; then
        echo $1
    else
        die "missing argument value"
    fi
}

function show_help {
    echo "usage:  $BASH_SOURCE dataset model [-r | --ratio] [-n | --node] [-N | --nb_task] \
                  [-g | --nb_gpu] [-p | --partition]"
    echo ""
    echo "Mandatory argument"
    echo "    --dataset DATASET               Available are {ubs8k, esc10, esc50, speechcommand}"
    echo "    --model MODEL                   Available are {cnn03, resnet[18|34|50], wideresnet28_[2|4|8]}"
    echo ""
    echo "Options"
    echo "    -n | --node NODE              On which node the job will be executed"
    echo "    -N | --nb_task NB TASK        On many parallel task"
    echo "    -g | --nb_gpu  NB GPU         On how many gpu this training should be done"
    echo "    -p | --partition PARTITION    On which partition the script will be executed"
    echo ""
    echo "Miscalleous arguments"
    echo "    -R | --resume     RESUME (default FALSE)"
    echo "    -h help"
    echo ""
    echo "Training parameters"
    echo "    --supervised_ratio           SUPERVISED RATIO (default 1.0)"
    echo "    --nb_epoch           NB_EPOCH (default 200)"
    echo "    --learning_rate      LR (default 0.001)"
    echo "    --batch_size         BATCH_SIZE (default 64)"
    echo "    --seed               SEED (default 1234)"
    echo "    --mixup          "
    echo "    --mixup_alpha        ALPHA"
    echo "    --mixup_max      "
    echo "    --mixup_label    "
    echo ""
    echo "    --specAugment    "
    echo "    --sa_time_drop_width     SA_TDW (default 32)"
    echo "    --sa_time_stripes_num    SA_TSN (default 2)"
    echo "    --sa_freq_drop_width     SA_FDW (default 4)"
    echo "    --sa_freq_stripes_num    SA_FSN (default 2)"
    echo ""
    echo "Available partition"
    echo "    GPUNodes"
    echo "    RTX6000Node"
}

# default parameters
# osirim parameters
NODE=" "
NB_TASK=1
NB_GPU=1
PARTITION="GPUNodes"

# training parameters
MODEL=cnn03
DATASET="ubs8k"
RATIO=1.0
ALPHA=0.4
NB_EPOCH=200
BATCH_SIZE=64
LR=0.003
SEED=1234

SA_TDW=32
SA_TSN=2
SA_FDW=4
SA_FSN=2
FLAG=""
F_MIXUP=""
NAME_EXTRA=""

# Parse the optional parameters
while :; do
    # If no more option (o no option at all)
    if ! [ "$1" ]; then break; fi

    case $1 in
        -R | --resume)      FLAG="${FLAG} --resume"; shift;;

        --dataset) DATASET=$(parse_long $2); shift; shift;;
        --model) MODEL=$(parse_long $2); shift; shift;;
        --supervised_ratio) RATIO=$(parse_long $2); shift; shift;;
        --nb_epoch)            EPOCH=$(parse_long $2); shift; shift;;
        --learning_rate)    LR=$(parse_long $2); shift; shift;;
        --batch_size)       BATCH_SIZE=$(parse_long $2); shift; shift;;
        --seed)             SEED=$(parse_long $2); shift; shift;;
        
        -n | --node) NODE=$(parse_long $2); shift; shift;;
        -N | --nb_task) NB_TASK=$(parse_long $2); shift; shift;;
        -g | --nb_gpu) NB_GPU=$(parse_long $2); shift; shift;;
        -p | --partition) PARTITION=$(parse_long $2); shift; shift;;
        
        --mixup) FLAG="${FLAG} --mixup"; F_MIXUP="mixup"; shift;;
        --mixup_alpha) ALPHA=$(parse_long $2); shift; shift;;
        --mixup_max) FLAG="${FLAG} --mixup_max"; shift;;
        --mixup_label) FLAG="${FLAG} --mixup_label"; shift;;
        
        --specAugment) FLAG="${FLAG} --specAugment"; shift;;
        --sa_time_drop_width) SA_TDW=$(parse_long $2); shift; shift;;
        --sa_time_stripes_num) SA_TSN=$(parse_long $2); shift; shift;;
        --sa_freq_drop_width) SA_FDW=$(parse_long $2); shift; shift;;
        --sa_freq_stripes_num) SA_FSN=$(parse_long $2); shift; shift;;

        -?*) echo "WARN: unknown option" $1 >&2
    esac
done


# prepare osirim node argument
if [ "${NODE}" = " " ]; then
   NODELINE=""
else
    NODELINE="#SBATCH --nodelist=${NODE}"
fi

# ___________________________________________________________________________________ #
LOG_DIR="logs"
SBATCH_JOB_NAME=sup_${DATASET}_${MODEL}_${RATIO}S_${F_MIXUP}

cat << EOT > .sbatch_tmp.sh
#!/bin/bash
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LOG_DIR}/${SBATCH_JOB_NAME}.out
#SBATCH --error=${LOG_DIR}/${SBATCH_JOB_NAME}.err
#SBATCH --ntasks=$NB_TASK
#SBATCH --cpus-per-task=5
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$NB_GPU
#SBATCH --gres-flags=enforce-binding
$NODELINE


# sbatch configuration
# container=/logiciels/containerCollections/CUDA10/pytorch.sif
container=/users/samova/lcances/container/pytorch-dev.sif
python=/users/samova/lcances/.miniconda3/envs/pytorch-dev/bin/python
script=../supervised/supervised_multi-label.py

# -------- hardware parameters --------
common_args="\${common_args} --nb_gpu ${NB_GPU}"

# -------- dataset & model ------
common_args="\${common_args} --dataset ${DATASET}"
common_args="\${common_args} --model ${MODEL}"

# -------- training common_args --------
common_args="\${common_args} --supervised_ratio ${RATIO}"
common_args="\${common_args} --nb_epoch ${EPOCH}"
common_args="\${common_args} --learning_rate ${LR}"
common_args="\${common_args} --batch_size ${BATCH_SIZE}"
common_args="\${common_args} --seed ${SEED}"

# ------- mixup parameters --------
common_args="\${common_args} --mixup_alpha ${ALPHA}"

# -------- flags --------
common_args="\${common_args} $FLAG"

# -------- log sufix --------
if [ -n "${F_MIXUP}" ]; then
    l_suffix="--log_suffix ${F_MIXUP}"
fi
    
echo srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${common_args} \${l_suffix}
srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${common_args} \${l_suffix}

EOT

echo "sbatch store in .sbatch_tmp.sh"
sbatch .sbatch_tmp.sh
