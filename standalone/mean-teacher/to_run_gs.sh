#!/bin/bash

c_args="--model wideresnet28_2 --dataset esc10"
c_args="${c_args} --nb_epoch 200"

LR=(0.001 0.003)
WL=(50 100)
ALPHA=(0.999)
LCM=(1 2)
BS=(64 100)

for l in ${!LR[*]}; do
for w in ${!WL[*]}; do
for a in ${!ALPHA[*]}; do
for c in ${!LCM[*]}; do
for b in ${!BS[*]}; do


args="--learning_rate ${LR[$l]}"
args="${args} --warmup_length ${WL[$w]}"
args="${args} --ema_alpha ${ALPHA[$a]}"
args="${args} --lambda_cost_max ${LCM[$c]}"
args="${args} --tensorboard_sufix ${LR[$l]}lr_${WL[$w]}wl_${ALPHA[$a]}a_${LCM[$c]}lccm"
args="${args} --batch_size ${BS[$b]}"

python student-teacher.py ${c_args} ${args}

done
done
done
done
done