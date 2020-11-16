# ------------------- CONSTANT PARAMETERS ----------------------
# training_parameters
T_ARGS="--supervised_ratio 0.1 --batch_size 100 --nb_epoch 300"
T_ARGS="${T_ARGS} --learning_rate 0.0005 --seed 1234"

# DCT parameters
DCT_ARGS="--lambda_cot_max 1 --lambda_diff_max 0.5 --warmup_length 160 --epsilon 0.02"
DCT_ARGS="${DCT_ARGS} --loss_cot_method mse"

# MT parameters
MT_ARGS="--ema_alpha 0.999 --teacher_noise 0 --lambda_ccost_max 1"
MT_ARGS="${MT_ARGS} --ccost_method js"

TOTAL_ARGS="${T_ARGS} ${DCT_ARGS} ${MT_ARGS}"
# -----------------------------------------------------------------------------

# For Mean Teacher, previous experiments show that using a JS divergence as consistency cost
# bash deep-co-training_teacher.sh wideresnet28_2 esc10 ${TOTAL_ARGS} --fusion_method m1 -C -p RTX6000Node
# bash deep-co-training_teacher.sh wideresnet28_2 esc10 ${TOTAL_ARGS} --fusion_method m2 -C
# bash deep-co-training_teacher.sh wideresnet28_2 esc10 ${TOTAL_ARGS} --fusion_method arithmetic_mean -C -p RTX6000Node
# bash deep-co-training_teacher.sh wideresnet28_2 esc10 ${TOTAL_ARGS} --fusion_method geometric_mean -C -p RTX6000Node
# bash deep-co-training_teacher.sh wideresnet28_2 esc10 ${TOTAL_ARGS} --fusion_method harmonic_mean  -C -p RTX6000Node

bash deep-co-training_teacher.sh wideresnet28_2 ubs8k ${TOTAL_ARGS} --fusion_method m1 -C -p RTX6000Node -N 2
