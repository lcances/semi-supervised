c_args="--alpha 0.999 --warmup_length 50 --lambda_ccost_max 3 --learning_rate 0.001"
c_args="${c_args} --batch_size 64 --epoch 200"


bash mean-teacher_mixup.sh wideresnet28_2 ubs8k ${c_args} --ratio 0.10 --mixup_alpha 0.4 -C

# c_args="--alpha 0.999 --warmup_length 50 --learning_rate 0.001"
# c_args="${c_args} --batch_size 64 --epoch 200"
# c_args="${c_args} --tensorboard_path mean-teacher_mixup_grid-search"
# c_args="${c_args} --checkpoint_path mean-teacher_mixup_grid-search"
# 
# aa=(1 3 5 8 10 15)
# 
# 
# for i in ${!aa[*]}
# do
#         cm=${aa[$i]}
# 
#         bash mean-teacher_mixup.sh wideresnet28_2 ubs8k ${c_args} --ratio 0.10 --lambda_ccost_max $cm --mixup_alpha 0
# done
