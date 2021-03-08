# /!\ BATCH SIZE IS DOUBLED BECAUSE RUNNING ON TWO GPU !!, if one GPU, use 64

common_args="--dataset audioset-unbalanced --supervised_ratio 1.0 --nb_epoch 125000 --batch_size 256 --learning_rate 0.003 --seed 1234"
mixup_args="--mixup --mixup_alpha 1.0 --mixup_max --mixup_label"
specaugment_args="--specAugment --sa_time_drop_width 32 --sa_time_stripes_num 1 --sa_freq_drop_width 4 --sa_freq_stripes_num 1"
args="$common_args $mixup_args $specaugment_args"

# Wideresnet28_2
# bash supervised_audioset.sh --model wideresnet28_2 -c 10 -g 2 -p GPUNodes $common_args
# bash supervised_audioset.sh --model wideresnet28_2 -c 10 -g 2 -p GPUNodes $common_args $mixup_arsg
# bash supervised_audioset.sh --model wideresnet28_2 -c 10 -g 2 -p RTX6000Node $common_args $specaugment_args
# bash supervised_audioset.sh --model wideresnet28_2 -c 10 -g 2 -p GPUNodes $args


# MobileNet
# bash supervised_audioset.sh --model MobileNetV2 -c 10 -g 2 -p GPUNodes $common_args
# bash supervised_audioset.sh --model MobileNetV2 -c 10 -g 2 -p GPUNodes $common_args $mixup_arsg
# bash supervised_audioset.sh --model MobileNetV2 -c 10 -g 2 -p RTX6000Node $common_args $specaugment_args
# bash supervised_audioset.sh --model MobileNetV2 -c 10 -g 2 -p RTX6000Node $args

bash supervised_audioset.sh --model cnn14 -c 10 -g 2 -p RTX6000Node $args
 
