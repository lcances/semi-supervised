def build_mapper(modules: dict) -> dict:
    dataset_mapper = dict()

    for dataset_name, dataset_module in modules.items():
        dataset_mapper[dataset_name] = {
            "supervised": dataset_module.supervised,
            "dct": dataset_module.dct,
            "dct_uniloss": dataset_module.dct_uniloss,
            "mean-teacher": dataset_module.mean_teacher,
        }

    return dataset_mapper


def load_callbacks(dataset: str, framework: str, **kwargs):
    import SSL.callbacks.esc as e
    import SSL.callbacks.ubs8k as u
    import SSL.callbacks.speechcommand as s
    import SSL.callbacks.audioset as a

    # get the corresping function mapper
    dataset_mapper = build_mapper({"esc10": e, "esc50": e, "ubs8k": u, "speechcommand": s,
                                   "audioset-balanced": a, "audioset-unbalanced": a})

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_optimizer(dataset: str, framework: str, **kwargs):
    import SSL.optimizer.esc as e
    import SSL.optimizer.ubs8k as u
    import SSL.optimizer.speechcommand as s
    import SSL.optimizer.audioset as a

    dataset_mapper = build_mapper({"esc10": e, "esc50":e, "ubs8k": u, "speechcommand": s,
                                   "audioset-balanced": a, "audioset-unbalanced": a})

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_preprocesser(dataset: str, framework: str, **kwargs):
    import SSL.preprocessing.esc as e
    import SSL.preprocessing.ubs8k as u
    import SSL.preprocessing.speechcommand as s
    import SSL.preprocessing.audioset as a

    dataset_mapper = build_mapper({"esc10": e, "esc50":e, "ubs8k": u, "speechcommand": s,
                                   "audioset-balanced": a, "audioset-unbalanced": a})

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_dataset(dataset: str, framework: str, **kwargs):
    import SSL.dataset_loader.esc as e
    import SSL.dataset_loader.ubs8k as u
    import SSL.dataset_loader.speechcommand as s
    import SSL.dataset_loader.audioset_balanced as a_bal
    import SSL.dataset_loader.audioset_unbalanced as a_unbal
    
    # Default dataset for audioset is the unsupervised version
    if dataset == "audioset":
        dataset = "audioset-unbalanced"

    dataset_mapper = build_mapper(
        {"esc10": e, "esc50":e, "ubs8k": u, "speechcommand": s, 
         "audioset-balanced": a_bal, "audioset-unbalanced": a_unbal})

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_helper(dataset: str, framework: str, mapper: dict, **kwargs):
    _dataset = dataset.lower()
    _framework = framework.lower()

    if _dataset not in mapper.keys():
        available_dataset = "{" + " | ".join(list(mapper.keys())) + "}"
        raise ValueError(f"dataset {_dataset} is not available. Available dataset are: {available_dataset}")

    if _framework not in mapper[_dataset].keys():
        available_framework = "{" + " | ".join(list(mapper[_dataset].keys()))
        raise ValueError(f"framework {_framework} is not available. Available framework are: {available_framework}")

    return mapper[_dataset][_framework](**kwargs)
