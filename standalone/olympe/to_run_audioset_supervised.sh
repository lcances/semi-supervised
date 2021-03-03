# /!\ BATCH SIZE IS DOUBLED BECAUSE RUNNING ON TWO GPU !!, if one GPU, use 64

common_args="--dataset audioset-unbalanced --supervised_ratio 1.0 --nb_epoch 250000 --batch_size 128 --learning_rate 0.003 --seed 1234"
mixup_args="--mixup --mixup_alpha 1.0 --mixup_max --mixup_label"
specaugment_args="--specAugment --sa_time_drop_width 32 --sa_time_stripes_num 1 --sa_freq_drop_width 4 --sa_freq_stripes_num 1"
args="$common_args $mixup_args $specaugment_args"

# Wideresnet28_2
olympe_args="--time 7-01" # Seven day and 1 hour (good for bs=128, 1 gpu, 10 cpu, chunked mel hdfs)
bash supervised_audioset.sh --model wideresnet28_2 --nb_cpu 10 --nb_gpu 1 $olympe_args $common_args
