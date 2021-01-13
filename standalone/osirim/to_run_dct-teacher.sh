# ------------------- CONSTANT PARAMETERS ----------------------
# training_parameters
T_ARGS="--supervised_ratio 0.1 --batch_size 100 --nb_epoch 300"
T_ARGS="${T_ARGS} --learning_rate 0.0005 --seed 1234"

# DCT parameters
DCT_ARGS="--lambda_cot_max 1 --lambda_diff_max 0.5 --warmup_length 160 --epsilon 0.02"
DCT_ARGS="${DCT_ARGS} --loss_cot_method js"

# MT parameters
MT_ARGS="--ema_alpha 0.999 --teacher_noise 0 --lambda_ccost_max 1"
MT_ARGS="${MT_ARGS} --ccost_method mse"

TOTAL_ARGS="${T_ARGS} ${DCT_ARGS} ${MT_ARGS}"
# -----------------------------------------------------------------------------

# For Mean Teacher, previous experiments show that using a JS divergence as consistency cost
# bash deep-co-training_teacher.sh wideresnet28_2 esc10 ${TOTAL_ARGS} --fusion_method m1 -C
# bash deep-co-training_teacher.sh wideresnet28_2 esc10 ${TOTAL_ARGS} --fusion_method m2 -C
# bash deep-co-training_teacher.sh wideresnet28_2 esc10 ${TOTAL_ARGS} --fusion_method arithmetic_mean -C
# bash deep-co-training_teacher.sh wideresnet28_2 esc10 ${TOTAL_ARGS} --fusion_method geometric_mean -C 
# bash deep-co-training_teacher.sh wideresnet28_2 esc10 ${TOTAL_ARGS} --fusion_method harmonic_mean  -C

# bash deep-co-training_teacher.sh wideresnet28_2 ubs8k ${TOTAL_ARGS} --fusion_method m1 -C -p RTX6000Node
# bash deep-co-training_teacher.sh wideresnet28_2 ubs8k ${TOTAL_ARGS} --fusion_method arithmetic_mean -C -p RTX6000Node
# bash deep-co-training_teacher.sh wideresnet28_2 ubs8k ${TOTAL_ARGS} --fusion_method geometric_mean -C -p RTX6000Node

# TOTAL_ARGS="--supervised_ratio 0.1 --batch_size 100 --nb_epoch 150 --learning_rate 0.0005 --seed 1234"
# TOTAL_ARGS="$TOTAL_ARGS --lambda_cot_max 1 --lambda_diff_max 0.5 --warmup_length 160 --epsilon 0.02"
# TOTAL_ARGS="$TOTAL_ARGS --ema_alpha 0.999 --teacher_noise 0 --lambda_ccost_max 1"
# 
# bash deep-co-training_teacher.sh wideresnet28_2 esc10 ${TOTAL_ARGS} --fusion_method m1 --loss_cot_method mse --ccost_method js -p RTX6000Node
# bash deep-co-training_teacher.sh wideresnet28_2 esc10 ${TOTAL_ARGS} --fusion_method m1 --loss_cot_method mse --ccost_method mse -p RTX6000Node
# bash deep-co-training_teacher.sh wideresnet28_2 esc10 ${TOTAL_ARGS} --fusion_method m1 --loss_cot_method js --ccost_method js -p RTX6000Node
# bash deep-co-training_teacher.sh wideresnet28_2 esc10 ${TOTAL_ARGS} --fusion_method m1 --loss_cot_method js --ccost_method mse -p RTX6000Node



# === === === === === === === === === === === === === === === === === === === === === === === === === === ===
#  GRID SEARCH
# === === === === === === === === === === === === === === === === === === === === === === === === === === ===
fix_params="--supervised_ratio 0.1 --batch_size 256 --nb_epoch 300 --seed 1234"

loss_cot_method=("js" "mse")
ccost_method=("js" "mse")

LCM=(1 5 10)    # Lambda cot max
LDM=(1 5 10)    # Lambda diff max
WL=(80 125 160)    # Warmup Length
EPS=(0.02 0.01) # Epsilon
ema_alpha=(0.999 0.99)
teacher_noise=(0)
LCCM=(1 5 10)  # Lambda consistency cost max
LR=(0.003 0.001)
fusion=("m1")

dataset="SpeechCommand"
model="wideresnet28_2"

for lcme in ${loss_cot_method[@]}; do
for cme in ${ccost_method[@]}; do
for lcm in ${LCM[@]}; do
for ldm in ${LDM[@]}; do
for wl in ${WL[@]}; do
for eps in ${EPS[@]}; do
for ea in ${ema_alpha[@]}; do
for tn in ${teacher_noise[@]}; do
for lccm in ${LCCM[@]}; do
for lr in ${LR[@]}; do
for f in ${fusion[@]}; do

R=$(( $RANDOM % 100 + 1 ))
if [ $R -gt 99 ]; then
    bash deep-co-training_teacher.sh $model $dataset ${fix_params} \
            --loss_cot_method $lcme \
            --ccost_method $cme \
            --lambda_cot_max $lcm \
            --lambda_diff_max $ldm \
            --warmup_length $wl \
            --epsilon $eps \
            --ema_alpha $ea \
            --teacher_noise $tn \
            --lambda_ccost_max $lccm \
            --learning_rate $lr \
            --fusion_method $f
fi

done
done
done
done
done
done
done
done
done
done
done
