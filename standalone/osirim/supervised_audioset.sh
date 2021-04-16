#!/bin/bash
# ___________________________________________________________________________________ #
source bash_scripts/parse_option.sh
source bash_scripts/cross_validation.sh

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
    echo "    -c | --nb_cpu  NB CPU         On how many CPU this training should be done"
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
NB_GPU=2
NB_CPU=10
PARTITION="GPUNodes"

# training parameters
MODEL=MobileNetV2
DATASET="audioset-unbalanced"
RATIO=1.0
NB_EPOCH=125000
BATCH_SIZE=256
LR=0.003
SEED=1234

# MixUp default parameters
ALPHA=0.4

# Spec augment default parameters
SA_TDW=""
SA_TSN=""
SA_FDW=""
SA_FSN=""

FLAG=""
F_MIXUP=""
F_SA=""
EXTRA_NAME=""

# Parse the optional parameters
while :; do
    # If no more option (o no option at all)
    if ! [ "$1" ]; then break; fi

    case $1 in
        # Osirim computation parameters
        -n | --node)      NODE=$(parse_long $2); shift; shift;;
        -N | --nb_task)   NB_TASK=$(parse_long $2); shift; shift;;
        -g | --nb_gpu)    NB_GPU=$(parse_long $2); shift; shift;;
        -c | --nb_cpu)    NB_CPU=$(parse_long $2); shift; shift;;
        -p | --partition) PARTITION=$(parse_long $2); shift; shift;;
        
        # Common training parameters
        -R | --resume)      FLAG="${FLAG} --resume"; shift;;
        --dataset)          DATASET=$(parse_long $2); shift; shift;;
        --model)            MODEL=$(parse_long $2); shift; shift;;
        --supervised_ratio) RATIO=$(parse_long $2); shift; shift;;
        --nb_epoch)         NB_EPOCH=$(parse_long $2); shift; shift;;
        --learning_rate)    LR=$(parse_long $2); shift; shift;;
        --batch_size)       BATCH_SIZE=$(parse_long $2); shift; shift;;
        --seed)             SEED=$(parse_long $2); shift; shift;;
        
        # MixUp parameters
        --mixup)         FLAG="${FLAG} --mixup";       EXTRA_NAME="${EXTRA_NAME}_mixup"; shift;;
        --mixup_alpha)   M_ALPHA=$(parse_long $2);     EXTRA_NAME="${EXTRA_NAME}-${M_ALPHA}a"; shift; shift;;
        --mixup_max)     FLAG="${FLAG} --mixup_max";   EXTRA_NAME="${EXTRA_NAME}-max"; shift;;
        --mixup_label)   FLAG="${FLAG} --mixup_label"; EXTRA_NAME="${EXTRA_NAME}-label"; shift;;
        
        # SpecAugment paramters
        --specAugment)         FLAG="${FLAG} --specAugment";  EXTRA_NAME="${EXTRA_NAME}_SpecAugment"; shift;;
        --sa_time_drop_width)  SA_TDW=$(parse_long $2);       EXTRA_NAME="${EXTRA_NAME}-${SA_TDW}tdw"; shift; shift;;
        --sa_time_stripes_num) SA_TSN=$(parse_long $2);       EXTRA_NAME="${EXTRA_NAME}-${SA_TSN}tsn"; shift; shift;;
        --sa_freq_drop_width)  SA_FDW=$(parse_long $2);       EXTRA_NAME="${EXTRA_NAME}-${SA_FDW}fdw"; shift; shift;;
        --sa_freq_stripes_num) SA_FSN=$(parse_long $2);       EXTRA_NAME="${EXTRA_NAME}-${SA_FSN}fsn"; shift; shift;;

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
SBATCH_JOB_NAME=sup_${DATASET}_${MODEL}_${RATIO}S${EXTRA_NAME}

echo logs are at:
echo $LOG_DIR/$SBATCH_JOB_NAME


cat << EOT > .sbatch_tmp.sh
#!/bin/bash
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LOG_DIR}/${SBATCH_JOB_NAME}.out
#SBATCH --error=${LOG_DIR}/${SBATCH_JOB_NAME}.err
#SBATCH --ntasks=$NB_TASK
#SBATCH --cpus-per-task=$NB_CPU
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$NB_GPU
#SBATCH --gres-flags=enforce-binding
$NODELINE


# sbatch configuration
# container=/logiciels/containerCollections/CUDA10/pytorch.sif
container=/users/samova/lcances/container/pytorch-dev.sif
python=/users/samova/lcances/.miniconda3/envs/pytorch-dev/bin/python
script=../supervised/supervised_audioset.py

source bash_scripts/add_option.sh

# -------- hardware parameters --------
common_args="\${common_args} --nb_gpu ${NB_GPU}"
common_args="\${common_args} --nb_cpu ${NB_CPU}"

# -------- dataset & model ------
common_args=\$(append "\$common_args" $DATASET '--dataset')
common_args=\$(append "\$common_args" $MODEL '--model')

# -------- training common_args --------
common_args=\$(append "\$common_args" $RATIO '--supervised_ratio')
common_args=\$(append "\$common_args" $NB_EPOCH '--nb_epoch')
common_args=\$(append "\$common_args" $LR '--learning_rate')
common_args=\$(append "\$common_args" $BATCH_SIZE '--batch_size')
common_args=\$(append "\$common_args" $SEED '--seed')

# -------- mixup parameters --------
common_args=\$(append "\$common_args" $M_ALPHA '--mixup_alpha')

# -------- flags --------
# should contain mixup_max, mixup_label, ccost_softmax
common_args="\${common_args} ${FLAG}"

# -------- SpecAugment parameters --------
common_args=\$(append "\$common_args" $SA_TDW '--sa_time_drop_width')
common_args=\$(append "\$common_args" $SA_TSN '--sa_time_stripes_num')
common_args=\$(append "\$common_args" $SA_FDW '--sa_freq_drop_width')
common_args=\$(append "\$common_args" $SA_FSN '--sa_freq_stripes_num')

# -------- log sufix --------
if [ -n "${EXTRA_NAME}" ]; then
    l_suffix="--log_suffix ${EXTRA_NAME}"
fi
    
echo srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${common_args} \${l_suffix}
srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${common_args} \${l_suffix}

EOT

echo "sbatch store in .sbatch_tmp.sh"
sbatch .sbatch_tmp.sh
# bash .sbatch_tmp.sh
