"""
The parameters for the best system for every dataset and every framework is describe into configuration files.
This file are written in YAML and can be found:
    - cifar10/config
    - ubs8k/config

This add every element of the configuration file into a NameSpace the same way argparse do.
It allow the usage of both argparse and configuration files.
"""
import yaml
from argparse import Namespace


def load_config(path: str):
    with open(path) as yml_file:
        config = yaml.safe_load(yml_file)

    return Namespace(**config)


def overide_config(args_form_argparse, args_from_config):
    _args_from_config = vars(args_from_config)
    _args_form_argparse = vars(args_form_argparse)

    # Override configuration arguments by the cmdline parameters
    for key, value in _args_form_argparse.items():
        if value is not None:
            _args_from_config[key] = value

    return Namespace(**_args_from_config)