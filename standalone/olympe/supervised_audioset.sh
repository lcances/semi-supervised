#!/bin/bash
# ___________________________________________________________________________________ #
source bash_scripts/parse_option.sh
source bash_scripts/cross_validation.sh

function show_help {
    echo "usage:  $BASH_SOURCE [--training parameters] [-n | --node] [-N | --nb_task] \
                  [-g | --nb_gpu] [-p | --partition]"
    echo ""
    echo "Calmip specific options"
    echo "    --ntasks NB TASK        NB_TASK        How many task to lunch"
    echo "    --nnodes NB NODE        NB_NODE        How many node to reserve"
    echo "    --ntasks_per_node       NTASKS_PER_NODE How many task in each nodes"
    echo "    --ntasks_per_core       NTASSK_PER_CORE How many core will be reserved for each task"
    echo "    -g | --nb_gpu           NB GPU         On how many gpu this training should be done"
    echo "    -c | --nb_cpu           NB_CPU         How many CPU per task"
    echo "    --mem                   MEM            How muh memory to reserve"
    echo "    --time                  TIME           How long the job will be running"
    echo ""
    echo "Miscalleous arguments"
    echo "    --script          SCRIPT the default supervised script that will executed,"
    echo "                             When training using audioset, please use supervised_multi-label.py"
    echo "    -C | --crossval   CROSSVAL (default FALSE)"
    echo "    -R | --resume     RESUME (default FALSE)"
    echo "    -h help"
    echo ""
    echo "Training parameters"
    echo "    --supervised_ratio           SUPERVISED RATIO (default 1.0)"
    echo "    --nb_epoch           NB_EPOCH (default 200)"
    echo "    --learning_rate      LR (default 0.001)"
    echo "    --batch_size         BATCH_SIZE (default 64)"
    echo "    --seed               SEED (default 1234)"
    echo "    --mixup              if set, will add mixup augmentation"
    echo "    --mixup_alpha        ALPHA mixup beta distribution parameters"
    echo "    --mixup_max          If set, will apply max between l and (1 - l)"
    echo "    --mixup_label        If set, will apply mixup also on the labels"
    echo ""
}

# default parameters
# osirim parameters
NB_TASK=1
NB_NODE=1
NB_GPU=1
NB_CPU=1
NTASKS_PER_NODE=1
NTASKS_PER_CORE=1
MEM=20000
TIME=01:00:00

# training parameters
MODEL=wideresnet28_2
DATASET="audioset"
RATIO=1.0
EMA_ALPHA=0.4
NB_EPOCH=200
BATCH_SIZE=64
LR=0.003
SEED=1234

FLAG=""
F_MIXUP=""
NAME_EXTRA=""
CROSSVAL=0
SCRIPT=../supervised/supervised_audioset.py

# Parse the optional parameters
while :; do
    # If no more option (o no option at all)
    if ! [ "$1" ]; then break; fi

    case $1 in
        --script)           SCRIPT=$(parse_long $2); shift; shift;;
        -R | --resume)      FLAG="${FLAG} --resume"; shift;;
        -C | --crossval)      CROSSVAL=1; shift;;

        --dataset)          DATASET=$(parse_long $2); shift; shift;;
        --model)            MODEL=$(parse_long $2); shift; shift;;
        --supervised_ratio) RATIO=$(parse_long $2); shift; shift;;
        --nb_epoch)         NB_EPOCH=$(parse_long $2); shift; shift;;
        --learning_rate)    LR=$(parse_long $2); shift; shift;;
        --batch_size)       BATCH_SIZE=$(parse_long $2); shift; shift;;
        --seed)             SEED=$(parse_long $2); shift; shift;;

        --nb_task)          NB_TASK=$(parse_long $2); shift; shift;;
        --nb_node)          NB_NODE=$(parse_long $2); shift; shift;;
        --nb_gpu)           NB_GPU=$(parse_long $2); shift; shift;;
        --nb_cpu)           NB_CPU=$(parse_long $2); shift; shift;;
        --ntasks_per_node)  NTASKS_PER_NODE=$(parse_long $2); shift; shift;;
        --ntasks_per_core)  NTASKS_PER_CORE=$(parse_long $2); shift; shift;;
        --mem)              MEM=$(parse_long $2); shift; shift;;
        --time)             TIME=$(parse_long $2); shift; shift;;

        --mixup)              FLAG="${FLAG} --mixup";       EXTRA_NAME="${EXTRA_NAME}_mixup"; shift;;
        --mixup_alpha)        M_ALPHA=$(parse_long $2);     EXTRA_NAME="${EXTRA_NAME}-${M_ALPHA}a": shift; shift;;
        --mixup_max)          FLAG="${FLAG} --mixup_max";   EXTRA_NAME="${EXTRA_NAME}-max"; shift;;
        --mixup_label)        FLAG="${FLAG} --mixup_label"; EXTRA_NAME="${EXTRA_NAME}-label"; shift;;

        -?*) echo "WARN: unknown option" $1 >&2
    esac
done


# ___________________________________________________________________________________ #
LOG_DIR="logs"
SBATCH_JOB_NAME=sup_${DATASET}_${MODEL}_${RATIO}S_${EXTRA_NAME}

cat << EOT > .sbatch_tmp.sh
#!/bin/bash
#SBATCH -J=${SBATCH_JOB_NAME}
#SBATCH --output=${LOG_DIR}/${SBATCH_JOB_NAME}.out
#SBATCH --error=${LOG_DIR}/${SBATCH_JOB_NAME}.err
#SBATCH -N $NB_NODE
#SBATCH -n $NB_TASK   # for gpu user it correspond to the number of cpu
#SBATCH --ntasks-per-node=$NB_TASK
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=$NB_CPU
#SBATCH --gres=gpu:$NB_GPU
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --mail-user=leo.cances@irit.fr

# Log some module
module load cuda/10.1.105
module load singularity/3.0.3


# sbatch configuration
container=/usr/local/containers/cances/pytorch_cuda_10.sif
python=/tmpdir/cances/miniconda3/envs/ssl/bin/python
script=$SCRIPT

source bash_scripts/add_option.sh
source bash_scripts/parse_option.sh

# prepare cross validation parameters
folds_str="$(cross_validation $DATASET $CROSSVAL)"
IFS=";" read -a folds <<< \$folds_str

# -------- hardware parameters --------
common_args=\$(append "\$common_args" $NB_GPU '--nb_gpu')
common_args=\$(append "\$common_args" $NB_CPU '--nb_cpu')

# -------- dataset & model ------
common_args=\$(append "\$common_args" $DTASET '--dataset')
common_args=\$(append "\$common_args" $MODEL '--model')

# -------- training common_args --------
common_args=\$(append "\$common_args" $RATIO '--supervised_ratio')
common_args=\$(append "\$common_args" $NB_EPOCH '--nb_epoch')
common_args=\$(append "\$common_args" $LR '--learning_rate')
common_args=\$(append "\$common_args" $BATCH_SIZE '--batch_size')
common_args=\$(append "\$common_args" $SEED '--seed')

# ------- mixup parameters --------
common_args=\$(append "\$common_args" $ALPHA '--mixup_alpha')

# -------- flags --------
common_args="\${common_args} $FLAG"

# -------- log sufix --------
if [ -n "${EXTRA_NAME}" ]; then
    l_suffix="--log_suffix ${EXTRA_NAME}"
fi

# -------- dataset specific parameters --------
case $DATASET in
    ubs8k | esc10)        dataset_args="--num_classes 10";;
    esc50)                dataset_args="--num_classes 50";;
    speechcommand)        dataset_args="--num_classes 35";;
    audioset)             dataset_args="--num_classes 527";;
    audioset-unbalanced)  dataset_args="--num_classes 527";;
    audioset-balanced)    dataset_args="--num_classes 527";;
    ?*) die "dataset ${DATASET} is not available"; exit 1;;
esac

run_number=0
for i in \${!folds[*]}
do
    run_number=\$(( \$run_number + 1 ))

    if [ $CROSSVAL -eq 1 ]; then
        tensorboard_sufix="--tensorboard_sufix run\${run_number}"
    else
        tensorboard_sufix=""
    fi

    extra_params="\${tensorboard_sufix} \${folds[\$i]}"

    echo srun \${python} \${script} \${common_args} \${dataset_args} \${extra_params}
    srun \${python} \${script} \${common_args} \${dataset_args} \${extra_params}
    # echo srun singularity exec \${container} \${python} \${script} \${common_args} \${dataset_args} \${extra_params}
    # srun singularity exec \${container} \${python} \${script} \${common_args} \${dataset_args} \${extra_params}
done

EOT

echo "sbatch store in .sbatch_tmp.sh"
sbatch .sbatch_tmp.sh
# bash .sbatch_tmp.sh
