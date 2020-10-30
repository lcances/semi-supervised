# python co-training.py --dataset ubs8k --model wideresnet28_2 -t 1 2 3 4 5 6 7 8 9 -v 10 --num_classes 10
# python co-training_mse.py --dataset ubs8k --model wideresnet28_2 -t 1 2 3 4 5 6 7 8 9 -v 10 --num_classes 10

# python co-training.py --dataset esc10 --model wideresnet28_2 -t 1 2 3 4 -v 5 --num_classes 10
# python co-training_mse.py --dataset esc10 --model wideresnet28_2 -t 1 2 3 4 -v 5 --num_classes 10

python co-training.py --dataset SpeechCommand --model wideresnet28_2 --num_classes 35
python co-training_mse.py --dataset SpeechCommand --model wideresnet28_2 --num_classes 35
