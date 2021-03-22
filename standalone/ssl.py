import argparse
import os
import subprocess

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


def get_script_path(method: str, dataset: str) -> str:
    path = 'supervised'

    if dataset.lower() == 'ComParE2021-PRS'.lower():
        return os.path.join(path, 'compare2021-prs.py')

    elif dataset.lower() == 'audioset'.lower():
        return os.path.join(path, 'audioset.py')

    else:
        return os.path.join(path, method + '.py')


def train(args):
    '''Perform training for the specified model on the specified dataset.'''
    # Automatically load the configuration file
    config_path = os.path.join('./..', 'config', args.method, args.dataset + '.yaml')
    script_path = get_script_path(args.method, args.dataset)

    print('current working directory: ', os.getcwd())
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List of possible action')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--method', type=str, choices=available_methods, default='supervised')
    parser_train.add_argument('--model', type=str, default='MobileNetV2')
    parser_train.add_argument('--dataset', type=str, default='ubs8k')
    parser_train.add_argument('kwargs', nargs='*')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')