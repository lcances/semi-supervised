import argparse
import os
from stat import ST_CTIME
import subprocess

from torch import sigmoid, softmax, argmax
from torchaudio import load as ta_load

from SSL.util.checkpoint import CheckPoint
from SSL.util.model_loader import load_model
from SSL.util.loaders import load_optimizer, load_preprocesser
from SSL.util.utils import DotDict


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
    'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn61', 'cnn7',
    'MobileNetV1', 'MobileNetV2']


dataset_info = DotDict({
    'script': DotDict({
        'ComParE2021-PRS'.lower(): 'compare2021-prs.py',
        'audioset-unbalanced': 'audioset.py',
        'audioset-balanced': 'audioset.py',
    }),
    
    'config': DotDict({
        'ComParE2021-PRS'.lower(): 'compare2021-prs.yaml',
        'esc10': 'esc10.yaml',
        'speechcommand': 'speechcommand.yaml',
        'ubs8k': 'ubs8k.yaml',
        'audioset-unbalanced': 'audioset.yaml',
        'audioset-balanced': 'audioset.yaml',
    }),
        
    'nb_class': DotDict({
        'ubs8k': 10,
        'esc10': 10,
        'esc50': 50,
        'speechcommand': 35,
        'compare2021-prs': 5,
        'audioset-unbalanced': 527,
        'audioset-balanced': 527,
    })
})


ubs8k_folds = [
        [[1,2,3,4,5,6,7,8,9],[10,]],
        [[2,3,4,5,6,7,8,9,10], [1,]],
        [[3,4,5,6,7,8,9,10,1], [2,]],
        [[4,5,6,7,8,9,10,1,2], [3,]],
        [[5,6,7,8,9,10,1,2,3], [4,]],
        [[6,7,8,9,10,1,2,3,4], [5,]],
        [[7,8,9,10,1,2,3,4,5], [6,]],
        [[8,9,10,1,2,3,4,5,6], [7,]],
        [[9,10,1,2,3,4,5,6,7], [8,]],
        [[10,1,2,3,4,5,6,7,8], [9,]],
]
esc_folds = [
        [[1,2,3,4], [5,]],
        [[2,3,4,5], [1,]],
        [[3,4,5,1], [2,]],
        [[4,5,1,2], [3,]],
        [[5,1,2,3], [4,]],
]



def get_script_path(method: str, dataset: str) -> str:
    path = method
    
    if dataset.lower() in dataset_info.script.keys():
        return os.path.join(path, dataset_info.script[dataset.lower()])
    
    else:
        return os.path.join(path, method + '.py')
    
    
def get_config_path(method: str, dataset: str) -> str:
    return os.path.join('./../..', 'config', method, dataset_info.config[dataset.lower()])


def list_file_per_date(path: str) -> list:
    paths = [os.path.join(path, f) for f in os.listdir(path)]
    status = [(os.stat(p)[ST_CTIME], p) for p in paths]

    return sorted(status, key=lambda x: x[0])


def train(args):
    '''Perform training for the specified model on the specified dataset.'''
    # Automatically load the configuration file
    config_path = get_config_path(args.method, args.dataset)
    script_path = get_script_path(args.method, args.dataset)
    python_path = args.python_bin if args.python_bin != '' else 'python'

    print('Default training parameters at: ', config_path)
    print('Executing: ', script_path)

    # Prepare the parameters to override
    override_hydra_params = args.kwargs
    override_hydra_params += ['path.dataset_root=../datasets']
    override_hydra_params += ['path.logs_root=../']
#     override_hydra_params += ['path.tensorboard_root=../tensorboard']

    subprocess_params = [python_path, script_path] + ['-cn', config_path] + override_hydra_params
    subprocess.call(subprocess_params)


def cross_validation(args):
    '''Perform cross-validation only for the dataset that requires it'''
    # Automatically load the configuration file
    config_path = os.path.join('./..', 'config', args.method, args.dataset + '.yaml')
    script_path = get_script_path(args.method, args.dataset)
    python_path = args.python_bin if args.python_bin != '' else 'python'

    print('Default training parameters at: ', config_path)
    print('Executing: ', script_path)

    # Prepare the parameters to override
    override_hydra_params = args.kwargs
    override_hydra_params += ['path.dataset_root=../datasets']
    override_hydra_params += ['path.logs_root=../']
#     override_hydra_params += ['path.checkpoint_root=../model_save']
#     override_hydra_params += ['path.tensorboard_root=../tensorboard']

    # prepare the set of run
    if args.dataset == 'ubs8k': folds = ubs8k_folds
    elif args.dataset == 'esc10': folds = esc_folds
    else: raise ValueError(f'there is no cross validation available for {args.dataset}')

    for i, (tf, vf) in enumerate(folds):
        override_hydra_params += [f'train_param.train_folds={tf}']
        override_hydra_params += [f'train_param.val_folds={vf}']
        override_hydra_params += [f'path.sufix=run-{i}']

        subprocess_params = [python_path, script_path] + override_hydra_params
        subprocess.call(subprocess_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List of possible action')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--python_bin', type=str, default='')
    parser_train.add_argument('--method', type=str, choices=available_methods, default='supervised')
    parser_train.add_argument('--dataset', type=str, default='ubs8k')
    parser_train.add_argument('kwargs', nargs='*')

    parser_cv = subparsers.add_parser('cross-validation')
    parser_cv.add_argument('--python_bin', type=str, default='')
    parser_cv.add_argument('--method', type=str, choices=available_methods, default='supervised')
    parser_cv.add_argument('--dataset', choices=['ubs8k', 'esc10'], default='ubs8k')
    parser_cv.add_argument('kwargs', nargs='*')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)

    elif args.mode == 'cross-validation':
        cross_validation(args)

    else:
        raise Exception('Error argument!')
