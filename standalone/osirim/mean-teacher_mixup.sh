source bash_scripts/parse_option.sh
source bash_scripts/cross_validation.sh

function show_help {
    echo "usage:  $BASH_SOURCE dataset model [-r | --ratio] [training options]"
    echo ""
    echo "Mandatory argument"
    echo "    dataset DATASET               Available are {ubs8k, esc{10|05}, speechcommand}"
    echo "    model MODEL                   Available are {cnn03, resnet{18|34|50}, wideresnet28_[2|4|8]}"
    echo ""
    echo "Miscalleous arguments"
    echo "    -C | --crossval   CROSSVAL (default FALSE)"
    echo "    -R | --resume     RESUME (default FALSE)"
    echo "    -h help"
    echo "    -T | --tensorboard_path T_PATH (default=mean-teacher_mixup)"
    echo "    --checkpoint_save       C_PATH (default=mean-teacher_mixup)"
    echo ""
    echo "Training parameters"
    echo "    --dataset            DATASET (default ubs8k)"
    echo "    --model              MODEL (default wideresnet28_4)"
    echo "    --ratio              SUPERVISED RATIO (default 1.0)"
    echo "    --epoch              EPOCH (default 200)"
    echo "    --learning_rate      LR (default 0.001)"
    echo "    --batch_size         BATCH_SIZE (default 64)"
    echo "    --seed               SEED (default 1234)"
    echo ""
    echo "    --lambda_ccost_max   LCM The consistency cost maximum value"
    echo "    --alpha              ALPHA value for the exponential moving average"
    echo "    --warmup_length      WL The length of the warmup"
    echo "    --noise              NOISE Add noise to the teacher input"
    echo "    --ccost_softmax      SOFTMAX Uses a softmax on the teacher logits (store false)"
    echo "    --ccost_method       CC_METHOD Uses JS or MSE for consistency cost"
    echo "    --tensorboard_sufix  SUFIX for the tensorboard name, more precision"
    echo "    --mixup_alpha        M_ALPHA alpha coef for beta distribution"
    echo "    --mixup_max          M_MAX apply max on lambda_ if 1, else not"
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
# osirim parameters
NODE=" "
NB_TASK=1
NB_GPU=1
PARTITION="GPUNodes"

# training parameters
MODEL=wideresnet28_2
DATASET="ubs8k"
RATIO=0.1
EPOCH=200
NB_CLS=10
BATCH_SIZE=64
RESUME=0
CROSSVAL=0
LR=0.001
SEED=1234
LCM=1
ALPHA=0.999
M_ALPHA=0.2
M_MAX=0
WL=50
SOFTMAX=0
CC_METHOD=mse
SUFIX="_"
T_PATH="mean-teacher_mixup"
C_PATH="mean-teacher_mixup"


# Parse the first two parameters
MODEL=$1; shift;
DATASET=$1; shift;
[[ $MODEL = -?* || $MODEL = "" ]] && die "please provide a model and a dataset"
[[ $DATASET = -?* || $DATASET = "" ]] && die "please provide a dataset"

# Parse the optional parameters
while :; do
    # If no more option (o no option at all)
    if ! [ "$1" ]; then break; fi

    case $1 in
        -R | --resume)           RESUME=1; shift;;
        -C | --crossval)         CROSSVAL=1; shift;;
        -T | --tensorboard_path) T_PATH=$(parse_long $2); shift; shift;;
        --checkpoint_path)       C_PATH=$(parse_long $2); shift; shift;;
        --ccost_softmax)         SOFTMAX=0; shift;;
        --mixup_max)             M_MAX=1; shift;;

        --dataset)            DATASET=$(parse_long $2); shift; shift;;
        --model)              MODEL=$(parse_long $2); shift; shift;;
        --ratio)              RATIO=$(parse_long $2); shift; shift;;
        --epoch)              EPOCH=$(parse_long $2); shift; shift;;
        --learning_rate)      LR=$(parse_long $2); shift; shift;;
        --batch_size)         BATCH_SIZE=$(parse_long $2); shift; shift;;
        --seed)               SEED=$(parse_long $2); shift; shift;;
        --tensorboard_sufix)  SUFIX=$(parse_long $2); shift; shift;;

        --lambda_ccost_max)   LCM=$(parse_long $2); shift; shift;;
        --alpha)              ALPHA=$(parse_long $2); shift; shift;;
        --ccost_method)       CC_METHOD=$(parse_long $2); shift; shift;;
        --warmup_length)      WL=$(parse_long $2); shift; shift;;
        --mixup_alpha)        M_ALPHA=$(parse_long $2); shift; shift;;

        -n | --node)      NODE=$(parse_long $2); shift; shift;;
        -N | --nb_task)   NB_TASK=$(parse_long $2); shift; shift;;
        -g | --nb_gpu)    NB_GPU=$(parse_long $2); shift; shift;;
        -p | --partition) PARTITION=$(parse_long $2); shift; shift;;

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
SBATCH_JOB_NAME=st_${DATASET}_${MODEL}_${RATIO}S

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
script=../mean-teacher/mean-teacher_mixup.py

# prepare cross validation parameters
folds_str="$(cross_validation $DATASET $CROSSVAL)"
IFS=";" read -a folds <<< \$folds_str

# -------- dataset & model ------
common_args="\${common_args} --dataset ${DATASET}"
common_args="\${common_args} --model ${MODEL}"

# -------- training common_args --------
common_args="\${common_args} --tensorboard_path ${T_PATH}"
common_args="\${common_args} --checkpoint_path ${C_PATH}"
common_args="\${common_args} --supervised_ratio ${RATIO}"
common_args="\${common_args} --nb_epoch ${EPOCH}"
common_args="\${common_args} --learning_rate ${LR}"
common_args="\${common_args} --batch_size ${BATCH_SIZE}"
common_args="\${common_args} --seed ${SEED}"

common_args="\${common_args} --lambda_cost_max ${LCM}"
common_args="\${common_args} --warmup_length ${WL}"
common_args="\${common_args} --ema_alpha ${ALPHA}"
common_args="\${common_args} --ccost_method ${CC_METHOD}"
common_args="\${common_args} --mixup_alpha ${M_ALPHA}"
if [ $SOFTMAX -eq 1 ]; then
    common_args="\${common_args} --ccost_softmax"
fi

if [ $M_MAX -eq 1 ]; then
    common_args="\${common_args} --mixup_max"
fi

# -------- resume training --------
if [ $RESUME -eq 1 ]; then
    common_args="\${common_args} --resume"
fi

# -------- dataset specific parameters --------
case $DATASET in
    ubs8k | esc10) dataset_args="--num_classes 10";;
    esc50) dataset_args="--num_classes 50";;
    SpeechCommand) dataset_args="--num_classes 35";;
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
