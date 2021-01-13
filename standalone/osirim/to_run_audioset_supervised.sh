params="--dataset audioset-unbalanced --supervised_ratio 1.0 --nb_epoch 15 --batch_size 128 --learning_rate 0.001 --seed 1234 --mixup_max --mixup_label --mixup_alpha 0.4 --specAugment"

bash supervised-multi-label.sh --model wideresnet28_2 -g 2 -p GPUNodes $params --mixup
bash supervised-multi-label.sh --model wideresnet28_2 -g 2 -p GPUNodes $params

bash supervised-multi-label.sh -g 1 -p RTX6000Node --model MobileNetV1 $params --mixup
bash supervised-multi-label.sh -g 1 -p RTX6000Node --model MobileNetV1 $params
