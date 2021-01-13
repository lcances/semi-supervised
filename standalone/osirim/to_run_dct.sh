c_args="--lambda_cot_max 1 --lambda_diff_max 0.5 --warmup_length 160 --learning_rate 0.0005 --epoch 300 --batch_size 100"

bash deep-co-training.sh wideresnet28_2 ubs8k ${c_args} --ratio 0.10 -C
bash deep-co-training.sh wideresnet28_2 ubs8k ${c_args} --ratio 0.10 --script ../co-training/co-training_mse.py -C

# bash deep-co-training.sh wideresnet28_2 esc10 ${c_args} --ratio 0.10 -C
# bash deep-co-training.sh wideresnet28_2 esc10 ${c_args} --ratio 0.10 --script ../co-training/co-training_mse.py -C

# bash deep-co-training.sh wideresnet28_2 SpeechCommand ${c_args} --ratio 0.10 
# bash deep-co-training.sh wideresnet28_2 SpeechCommand ${c_args} --ratio 0.10 --script ../co-training/co-training_mse.py

