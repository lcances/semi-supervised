t_args="--learning_rate 0.001 --batch_size 64 --nb_epoch 200 --supervised_ratio 0.1"
mt_args="--ema_alpha 0.999 --warmup_length 50 --lambda_cost_max 1 --ccost_method js --ccost_softmax"
m_args="--mixup --mixup_alpha 0.4"

# small grid search with mixup
# bash mean-teacher_mixup.sh --dataset ubs8k --model wideresnet28_2 ${t_args} ${mt_args}
# bash mean-teacher_mixup.sh --dataset ubs8k --model wideresnet28_2 ${t_args} ${mt_args} ${m_args}
# bash mean-teacher_mixup.sh --dataset ubs8k --model wideresnet28_2 ${t_args} ${mt_args} ${m_args} --mixup_max
# bash mean-teacher_mixup.sh --dataset ubs8k --model wideresnet28_2 ${t_args} ${mt_args} ${m_args} --mixup_label
# bash mean-teacher_mixup.sh --dataset ubs8k --model wideresnet28_2 ${t_args} ${mt_args} ${m_args} --mixup_label --mixup_max

# Cross validation run on the best
bash mean-teacher_mixup.sh --dataset ubs8k --model wideresnet28_2 ${t_args} ${mt_args} -C
#bash mean-teacher_mixup.sh --dataset ubs8k --model wideresnet28_2 ${t_args} ${mt_args} ${m_args} -C
# 
bash mean-teacher_mixup.sh --dataset esc10 --model wideresnet28_2 ${t_args} ${mt_args} -C
#bash mean-teacher_mixup.sh --dataset esc10 --model wideresnet28_2 ${t_args} ${mt_args} ${m_args} -C
# 
bash mean-teacher_mixup.sh --dataset speechcommand --model wideresnet28_2 ${t_args} ${mt_args}
#bash mean-teacher_mixup.sh --dataset speechcommand --model wideresnet28_2 ${t_args} ${mt_args} ${m_args}
