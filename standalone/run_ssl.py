import argparse
import os
from stat import ST_CTIME
import subprocess

from torch import sigmoid, softmax, argmax
from torchaudio import load as ta_load

from SSL.util.checkpoint import CheckPoint
from SSL.util.model_loader import load_model
from SSL.util.loaders import load_optimizer, load_preprocesser

available_methods = [
    'supervised',
    'mean-teacher',
    'deep-co-training',
    'fixmatch',
]


# TODO find a way to differentiate models from layers
available_models = [
    'ScalableCnn', 'cnn', 'cnn0', 'cnn_advBN', 'scallable1',
    'resnet18', 'resnet34', 'resnet50', 'wideresnet28_2', 'wideresnet28_4', 'wideresnet28_8',
    'cnn0', 'cnn01', 'cnn02', 'cnn03', 'cnn04', 'cnn05', 'cnn06', 'cnn07', 'cnn1',
    'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn61', 'cnn7']

nb_class = {
    'ubs8k': 10,
    'esc10': 10,
    'esc50': 50,
    'speechcommand': 35,
    'compare2021-prs': 5,
    'audioset-unbalanced': 527,
    'audioset-balanced': 527,
}


def get_script_path(method: str, dataset: str) -> str:
    path = 'supervised'

    if dataset.lower() == 'ComParE2021-PRS'.lower():
        return os.path.join(path, 'compare2021-prs.py')

    elif dataset.lower() == 'audioset'.lower():
        return os.path.join(path, 'audioset.py')

    else:
        return os.path.join(path, method + '.py')


def list_file_per_date(path: str) -> list:
    paths = [os.path.join(path, f) for f in os.listdir(path)]
    status = [(os.stat(p)[ST_CTIME], p) for p in paths]

    return sorted(status, key=lambda x: x[0])


def train(args):
    '''Perform training for the specified model on the specified dataset.'''
    # Automatically load the configuration file
    config_path = os.path.join('./..', 'config', args.method, args.dataset + '.yaml')
    script_path = get_script_path(args.method, args.dataset)

    print('Default training parameters at: ', config_path)
    print('Executing: ', script_path)

    # Prepare the parameters to override
    override_hydra_params = args.kwargs
    override_hydra_params += [f'model.model={args.model}']
    override_hydra_params += ['path.dataset_root=../datasets']
    override_hydra_params += ['path.checkpoint_root=../model_save']
    override_hydra_params += ['path.tensorboard_root=../tensorboard']

    subprocess.call(['python', '-c', 'import os; print(os.getcwd())'])
    subprocess_params = ['python', script_path] + override_hydra_params
    print(subprocess_params)

    subprocess.call(subprocess_params)


def inference(args):
    ''' Will load the specified model trained with the specified methods on
    the specified dataset.
    '''

    if args.file == '' and args.dir == '':
        raise RuntimeError(
            'You must specified an audiofile or a directory containing audiofiles'
        )

    weight_path = os.path.join('../', 'model_save', args.dataset, args.method, args.model)

    # If no weight file specified, use the most recent one
    if args.weights == '':
        date_sorted_weights = list_file_per_date(weight_path)
        weight_path = date_sorted_weights[-1][1]

        print('No weights file specified.')
        print('Using the most recent compatible weight_file')
        print('file: ', os.path.basename(date_sorted_weights[-1][1]))

    else:
        pass

    # Load the transforms, the model and the checkpoint
    print('Loading pre-processing ...')
    _, transform = load_preprocesser(args.dataset, args.method)

    print('Loading model ...')
    model_func = load_model(args.dataset, args.model)
    model = model_func(num_classes=nb_class[args.dataset.lower()])

    print('Loading weights ...')
    optimizer = load_optimizer(args.dataset, args.method, model=model, learning_rate=0.003)
    checkpoint = CheckPoint(model, optimizer, mode="max", name=weight_path)
    checkpoint.load_best()

    print('Loadind file and applying preprocessing ...')
    waveform, sr = ta_load(args.file)
    data = transform(waveform)
    data = data.unsqueeze(0)

    if args.cuda:
        print('Moving model to GPU')
        model = model.cuda()
        data = data.cuda()

    logits = model(data)

    if args.output == 'logits':
        results = logits

    elif args.output == 'sigmoid':
        results = sigmoid(logits)

    elif args.output == 'softmax':
        results = softmax(logits, dim=1)

    elif args.output == 'pred':
        results = argmax(logits, dim=1)
    
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List of possible action')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--method', type=str, choices=available_methods, default='supervised')
    parser_train.add_argument('--model', type=str, default='MobileNetV2')
    parser_train.add_argument('--dataset', type=str, default='ubs8k')
    parser_train.add_argument('kwargs', nargs='*')

    parser_infer = subparsers.add_parser('inference')
    parser_infer.add_argument('-f', '--file', type=str, default='')
    parser_infer.add_argument('-d', '--dir', type=str, default='')
    parser_infer.add_argument('-w', '--weights', type=str, default='')
    parser_infer.add_argument('-o', '--output', choices=['logits', 'sigmoid', 'softmax', 'pred'], default='logits')
    parser_infer.add_argument('--cuda', action='store_true', default=False)
    parser_infer.add_argument('--method', type=str, choices=available_methods, default='supervised')
    parser_infer.add_argument('--model', type=str, default='MobileNetV2')
    parser_infer.add_argument('--dataset', type=str, default='ubs8k')
    parser_infer.add_argument('kwargs', nargs='*')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference':
        inference(args)

    else:
        raise Exception('Error argument!')