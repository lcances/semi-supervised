def build_mapper(modules: list) -> dict:
    dataset_mapper = dict()

    for dataset_name, dataset_module in modules.items():
        print(dataset_name)
        dataset_mapper[str(dataset_name)] = {
            "supervised": dataset_module.supervised,
            "dct": dataset_module.dct,
            "dct_uniloss": dataset_module.dct_uniloss,
            "mean-teacher": dataset_module.mean_teacher,
        }

    return dataset_mapper


def load_callbacks(dataset: str, framework: str, **kwargs):
    import SSL.callbacks.esc as esc
    import SSL.callbacks.ubs8k as ubs8k
    import SSL.callbacks.speechcommand as speechcommand

    # get the corresping function mapper
    dataset_mapper = build_mapper([esc, ubs8k, speechcommand])

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_optimizer(dataset: str, framework: str, **kwargs):
    import SSL.optimizer.esc as esc
    import SSL.optimizer.ubs8k as ubs8k
    import SSL.optimizer.speechcommand as speechcommand

    dataset_mapper = build_mapper([esc, ubs8k, speechcommand])

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_preprocesser(dataset: str, framework: str, **kwargs):
    import SSL.preprocessing.esc as esc
    import SSL.preprocessing.ubs8k as ubs8k
    import SSL.preprocessing.speechcommand as speechcommand

    dataset_mapper = build_mapper([esc, ubs8k, speechcommand])

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_dataset(dataset: str, framework: str, **kwargs):
    import SSL.dataset_loader.esc as esc
    import SSL.dataset_loader.ubs8k as ubs8k
    import SSL.dataset_loader.speechcommand as speechcommand

    dataset_mapper = build_mapper([esc, ubs8k, speechcommand])

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_helper(dataset: str, framework: str, mapper: dict, **kwargs):
    if dataset not in mapper.keys():
        available_dataset = "{" + " | ".join(list(mapper.keys())) + "}"
        raise ValueError(f"dataset {dataset} is not available. Available dataset are: {available_dataset}")

    if framework not in mapper[dataset].keys():
        available_framework = "{" + " | ".join(list(mapper[dataset].keys()))
        raise ValueError(f"framework {framework} is not available. Available framework are: {available_framework}")

    return mapper[dataset][framework](**kwargs)