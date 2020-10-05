import torch.optim

import DCT.callbacks.esc as esc
import DCT.callbacks.cifar10 as c10
import DCT.callbacks.ubs8k as u8
import DCT.callbacks.speechcommand as sc

dataset_mapper = {
    "ubs8k": {
        "supervised": u8.supervised,
        "dct": u8.dct,
        "uniloss": None,
        "aug4adv": None,
    },

    "cifar10": {
        "supervised": c10.supervised,
        "dct": c10.dct,
        "uniloss": None,
        "aug4adv": None,
    },

    "esc10": {
        "supervised": esc.supervised,
        "dct": esc.dct,
        "uniloss": None,
        "aug4adv": None,
    },

    "esc50": {
        "supervised": esc.supervised,
        "dct": esc.dct,
        "uniloss": None,
        "aug4adv": None,
    },

    "SpeechCommand": {
        "supervised": sc.supervised,
        "dct": sc.dct,
        "uniloss": None,
        "aug4adv": None,
    },

    "gtzan": {
        "supervised": None,
        "dct": None,
        "uniloss": None,
        "aug4adv": None,
    }
}


def load_callbacks_helper(framework: str, mapper: dict, **kwargs):
    if framework not in mapper:
        raise ValueError(f"Framework {framework} doesn't exist. Available framework are {list(mapper.keys())}")

    else:
        return mapper[framework](**kwargs)


def load_callbacks(dataset_name: str, framework: str,
                   **kwargs) -> torch.optim.Optimizer:
    """ The list of parameters depend on the dataset and framework used.
        See DCT.optimizer.<dataset> for more detail
    """

    if dataset_name not in dataset_mapper:
        available_dataset = ", ".join(list(dataset_mapper.keys()))
        raise ValueError(
            f"dataset {dataset_name} is not available.\n Available datasets: {available_dataset}")

    return load_callbacks_helper(framework, dataset_mapper[dataset_name],
                                 **kwargs)