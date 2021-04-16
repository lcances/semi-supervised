#!/bin/bash
# ___________________________________________________________________________________ #
source bash_scripts/parse_option.sh
source bash_scripts/cross_validation.sh

function show_help {
    echo "usage:  $BASH_SOURCE [-r | --ratio] [-n | --node] [-N | --nb_task] \
                  [-g | --nb_gpu] [-p | --partition] [--name]"
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
    echo "    --name           NAME use to be display by slurm and for the configuration files."
    echo '' 
    echo 'Training parameters'
    echo '    --dataset     DATASET   Which dataset to use'
    echo '    --method      METHOD    Which method to use [supervised|mean-teacher|deep-co-training|fixmatch]'
    echo '    --cmd         CMD       Which command to execute [train|cross-validation]'
    echo '    --script_param'
    echo '        Script arguments are define in run_ssl.py and the corresponding configuration hydra file'
    echo '        They must be fine by the argument --script_param'
    echo '        Then, define like usual'
    echo "        exemple: $BASH_SOURCE [Options] --script_param hydra.param1=1 hydra.param2=2 ..."
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
NB_CPU=4
PARTITION="GPUNodes"
FINISH="0"
NAME="noname"
DATASET='ubs8k'
METHOD='supervised'
CMD='train'

# Parse the optional parameters
while :; do
    # If no more option (o no option at all)
    if ! [ "$1" ] || [ "$FINISH" = "1" ]; then break; fi

    case $1 in
        # Osirim computation parameters
        -n | --node)      NODE=$(parse_long $2); shift; shift;;
        -N | --nb_task)   NB_TASK=$(parse_long $2); shift; shift;;
        -g | --nb_gpu)    NB_GPU=$(parse_long $2); shift; shift;;
        -c | --nb_cpu)    NB_CPU=$(parse_long $2); shift; shift;;
        -p | --partition) PARTITION=$(parse_long $2); shift; shift;;
        --name)           NAME=$(parse_long $2); shift; shift;;
        
        --dataset)       DATASET=$(parse_long $2); shift; shift;;
        --method)        METHOD=$(parse_long $2); shift; shift;;
        --cmd)           CMD=$(parse_long $2); shift; shift;;
        --script_param)  FINISH="1"; shift;;

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
LOG_DIR="osirim/logs"
SBATCH_JOB_NAME=${CMD}_${METHOD}_${DATASET}
echo logs are at:
echo SBATCH_JOB_NAME

cat << EOT > ../.sbatch_tmp.sh
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
container=/logiciels/containerCollections/CUDA10/pytorch-NGC-20-03-py3.sif
# container=/users/samova/lcances/container/pytorch-dev.sif
python=/users/samova/lcances/.miniconda3/envs/ssl/bin/python
script=run_ssl.py

args="--dataset ${DATASET} --method ${METHOD} --python_bin \${python}"

echo srun -n 1 -N 1 singularity exec \${container} \${python} \${script} ${CMD} \${args} $@ 
srun -n 1 -N 1 singularity exec \${container} \${python} \${script} ${CMD} \${args} $@

EOT

CWD=`pwd`
cd ..
echo "sbatch store in ${CWD}/../.sbatch_tmp.sh"
sbatch .sbatch_tmp.sh
# bash .sbatch_tmp.sh
cd $CWD
