t_args="--learning_rate 0.0005 --batch_size 64 --nb_epoch 300 --supervised_ratio 0.1"
dct_args="--lambda_cot_max 1 --lambda_diff_max 0.5 --warmup_length 160 --epsilon 0.02"
m_args="--mixup --mixup_alpha 0.4"

#bash deep-co-training_mixup.sh --dataset ubs8k --model wideresnet28_2 ${t_args} ${dct_args}
# bash deep-co-training_mixup.sh --dataset ubs8k --model wideresnet28_2 ${t_args} ${dct_args} ${m_args}
# bash deep-co-training_mixup.sh --dataset ubs8k --model wideresnet28_2 ${t_args} ${dct_args} ${m_args} --mixup_max
# bash deep-co-training_mixup.sh --dataset ubs8k --model wideresnet28_2 ${t_args} ${dct_args} ${m_args} --mixup_label
# bash deep-co-training_mixup.sh --dataset ubs8k --model wideresnet28_2 ${t_args} ${dct_args} ${m_args} --mixup_max --mixup_label


# bash deep-co-training_mixup.sh --dataset ubs8k --model wideresnet28_2 ${t_args} ${dct_args} -C 
# bash deep-co-training_mixup.sh --dataset ubs8k --model wideresnet28_2 ${t_args} ${dct_args} ${m_args} -C
# 
# bash deep-co-training_mixup.sh --dataset esc10 --model wideresnet28_2 ${t_args} ${dct_args} -C 
# bash deep-co-training_mixup.sh --dataset esc10 --model wideresnet28_2 ${t_args} ${dct_args} ${m_args} -C

bash deep-co-training_mixup.sh --dataset speechcommand --model wideresnet28_2 ${t_args} ${dct_args} -p RTX6000Node 
bash deep-co-training_mixup.sh --dataset speechcommand --model wideresnet28_2 ${t_args} ${dct_args} ${m_args}

