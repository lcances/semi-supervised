params="--dataset audioset-unbalanced --supervised_ratio 1.0 --nb_epoch 15 --batch_size 128 --learning_rate 0.001 --seed 1234 --mixup_max --mixup_label --mixup_alpha 0.4"

# No spec augment neither mixup 
bash supervised-multi-label.sh --model wideresnet28_2 -g 2 -p GPUNodes $params


# bash supervised-multi-label.sh --model wideresnet28_2 -g 2 -p GPUNodes $params --specAugment --mixup
# bash supervised-multi-label.sh --model wideresnet28_2 -g 2 -p GPUNodes $params --specAugment
# 
# bash supervised-multi-label.sh -g 1 -p RTX6000Node --model MobileNetV1 $params --specAugment --mixup
# bash supervised-multi-label.sh -g 1 -p RTX6000Node --model MobileNetV1 $params --specAugment
