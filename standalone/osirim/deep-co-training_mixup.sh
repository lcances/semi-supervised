source bash_scripts/parse_option.sh
source bash_scripts/cross_validation.sh

function show_help {
    echo "usage:  $BASH_SOURCE [osirim option] [training option]"
    echo ""
    echo "Miscalleous arguments"
    echo "    -C | --crossval   CROSSVAL (default FALSE)"
    echo "    -R | --resume     RESUME (default FALSE)"
    echo "    -h help"
    echo ""
    echo "Training parameters"
    echo "    --dataset         DATASET (default ubs8k)"
    echo "    --model           MODEL (default wideresnet28_4)"
    echo "    --supervised_ratio           SUPERVISED RATIO (default 1.0)"
    echo "    --batch_size      BATCH_SIZE (default 64)"
    echo "    --nb_epoch           EPOCH (default 200)"
    echo "    --learning_rate   LR (default 0.001)"
    echo "    --seed            SEED (default 1234)"
    echo ""
    echo " Deep Co training parameters"
    echo "    --lambda_cot_max LCM          Lambda cot max"
    echo "    --lambda_diff_max LDM         Lambda diff max"
    echo "    --epsilon               EPSILON"
    echo "    --warmup_lenght WL            Warmup lenght"
    echo ""
    echo " Mixup augmentation parameters"
    echo "    --mixup              F_MIXUP"
    echo "    --mixup_alpha        M_ALPHA"
    echo "    --mixup_max      "
    echo "    --mixup_label    "
    echo ""
    echo "Osirim related parameters"
    echo "    -n | --node NODE              On which node the job will be executed"
    echo "    -N | --nb_task NB TASK        On many parallel task"
    echo "    -g | --nb_gpu  NB GPU         On how many gpu this training should be done"
    echo "    -p | --partition PARTITION    On which partition the script will be executed"
    echo ""
    echo "Available partition"
    echo "    GPUNodes"
    echo "    RTX6000Node"
}

# default parameters
# osirim parametesr
NODE=" "
NB_TASK=1
NB_GPU=1
PARTITION="GPUNodes"

# training parameters
DATASET="ubs8k"
MODEL="wideresnet28_2"
SUPERVISED_RATIO=0.1
BATCH_SIZE=300
NB_EPOCH=300
LR=0.0005
SEED=1234

# Deep Co training parameters
LCM=1
LDM=0.5
EPSILON=0.02
WL=160

# mixup parameters
M_ALPHA=""

# Flag and miscallenous
RESUME=0
CROSSVAL=0
FLAG=""
F_MIXUP=""
F_MAX=""
F_LABEL=""
EXTRA_NAME=""

# Parse the optional parameters
while :; do
    # If no more option (o no option at all)
    if ! [ "$1" ]; then break; fi

    case $1 in
        -R | --resume)      RESUME=1; shift;;
        -C | --crossval)    CROSSVAL=1; shift;;

        --dataset)          DATASET=$(parse_long $2); shift; shift;;
        --model)            MODEL=$(parse_long $2); shift; shift;;
        --supervised_ratio) SUPERVISED_RATIO=$(parse_long $2); shift; shift;;
        --nb_epoch)         NB_EPOCH=$(parse_long $2); shift; shift;;
        --learning_rate)    LR=$(parse_long $2); shift; shift;;
        --batch_size)       BATCH_SIZE=$(parse_long $2); shift; shift;;
        --seed)             SEED=$(parse_long $2); shift; shift;;

        --lambda_cot_max)   LCM=$(parse_long $2); shift; shift;; #
        --lambda_diff_max)  LDM=$(parse_long $2); shift; shift;; #
        --warmup_length)    WL=$(parse_long $2); shift; shift;; #
        --epsilon)          EPSILON=$(parse_long $2); shift; shift;; #
        
        --mixup)              FLAG="${FLAG} --mixup";       EXTRA_NAME="${EXTRA_NAME}_mixup"; shift;;
        --mixup_alpha)        M_ALPHA=$(parse_long $2);     EXTRA_NAME="${EXTRA_NAME}-${M_ALPHA}a": shift; shift;;
        --mixup_max)          FLAG="${FLAG} --mixup_max";   EXTRA_NAME="${EXTRA_NAME}-max"; shift;;
        --mixup_label)        FLAG="${FLAG} --mixup_label"; EXTRA_NAME="${EXTRA_NAME}-label"; shift;;

        -n | --node)      NODE=$(parse_long $2); shift; shift;;
        -N | --nb_task)   NB_TASK=$(parse_long $2); shift; shift;;
        -g | --nb_gpu)    NB_GPU=$(parse_long $2); shift; shift;;
        -p | --partition) PARTITION=$(parse_long $2); shift; shift;;
        -s | --script) SCRIPT=$(parse_long $2); shift; shift;;

        -?*) echo "WARN: unknown option" $1 >&2
    esac
done


if [ "${NODE}" = " " ]; then
   NODELINE=""
else
    NODELINE="#SBATCH --nodelist=${NODE}"
fi

# ___________________________________________________________________________________ #
LOG_DIR="logs"
SBATCH_JOB_NAME=dct_${DATASET}_${MODEL}_${SUPERVISED_RATIO}S${EXTRA_NAME}
echo logs save at $SBATCH_JOB_NAME

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
script=../co-training/co-training_mixup.py

source bash_scripts/add_option.sh


# prepare cross validation parameters
# ---- default, no crossvalidation
if [ "$DATASET" = "ubs8k" ]; then
    folds=("-t 1 2 3 4 5 6 7 8 9 -v 10")
elif [ "$DATASET" = "esc10" ] || [ "$DATASET" = "esc50" ]; then
    folds=("-t 1 2 3 4 -v 5")
elif [ "$DATASET" = "SpeechCommand" ]; then
    folds=("-t 1 -v 2") # fake array to ensure exactly one run. Nut used by SpeechCommand anyway"
fi

# prepare cross validation parameters
folds_str="$(cross_validation $DATASET $CROSSVAL)"
IFS=";" read -a folds <<< \$folds_str

# -------- dataset & model ------
common_args=\$(append "\$common_args" $DATASET '--dataset')
common_args=\$(append "\$common_args" $MODEL '--model')

# -------- training common_args --------
common_args=\$(append "\$common_args" $SUPERVISED_RATIO '--supervised_ratio')
common_args=\$(append "\$common_args" $NB_EPOCH '--nb_epoch')
common_args=\$(append "\$common_args" $LR '--learning_rate')
common_args=\$(append "\$common_args" $BATCH_SIZE '--batch_size')
common_args=\$(append "\$common_args" $SEED '--seed')

# -------- deep co training specific parameters --------
common_args=\$(append "\$common_args" $LCM '--lambda_cot_max')
common_args=\$(append "\$common_args" $LDM '--lambda_diff_max')
common_args=\$(append "\$common_args" $WL '--warmup_length')
common_args=\$(append "\$common_args" $EPSILON '--epsilon')

# -------- mixup parameters --------
common_args=\$(append "\$common_args" $M_ALPHA '--mixup_alpha')

# -------- flags --------
# should contain mixup_max, mixup_label, ccost_softmax
common_args="\${common_args} $FLAG"

# -------- resume training --------
if [ $RESUME -eq 1 ]; then
    common_args="\${common_args} --resume"
fi

# dataset specific parameters
case $DATASET in
    ubs8k | esc10) dataset_args="--num_classes 10";;
    esc50) dataset_args="--num_classes 50";;
    speechcommand) dataset_args="--num_classes 35";;
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
    
    echo srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${common_args} \${dataset_args} \${extra_params}
    srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${common_args} \${dataset_args} \${extra_params}
done


EOT

echo "sbatch store in .sbatch_tmp.sh"
sbatch .sbatch_tmp.sh
#bash .sbatch_tmp.sh
