# /!\ BATCH SIZE IS DOUBLED BECAUSE RUNNING ON TWO GPU !!, if one GPU, use 32

common_args="--dataset audioset-unbalanced --supervised_ratio 1.0 --nb_epoch 15 --batch_size 128 --learning_rate 0.003 --seed 1234"
mixup_args="--mixup --mixup_alpha 0.4 --mixup_label"
specaugment_args="--specAugment --sa_time_drop_width 32 --sa_time_stripes_num 1 --sa_freq_drop_width 4 --sa_freq_stripes_num 1"
args="$common_args $mixup_args $specaugment_args"

# all together
bash supervised-multi-label.sh --model wideresnet28_2 -c 10 -g 2 -p GPUNodes $args

# # mixup only
# bash supervised-multi-label.sh --model wideresnet28_2 -c 10 -g 2 -p GPUNodes $common_args $mixup_arsg
 
# # specaugment only
# bash supervised-multi-label.sh --model wideresnet28_2 -c 10 -g 2 -p RTX6000Node $common_args $specaugment_args
