c_args="--alpha 0.999 --warmup_length 50 --lambda_ccost_max 1 --learning_rate 0.003"
c_args="${c_args} --teacher_noise 0 --batch_size 64 --nb_epoch 200"

bash student-teacher.sh wideresnet28_2 esc10 ${c_args} --ccost_method mse --ratio 0.10 -C
#bash student-teacher.sh wideresnet28_2 esc10 ${c_args} --ccost_method js --ratio 0.10 -C
#
#bash student-teacher.sh wideresnet28_2 ubs8k ${c_args} --ccost_method mse --ratio 0.10 -C
#bash student-teacher.sh wideresnet28_2 ubs8k ${c_args} --ccost_method js --ratio 0.10 -C
