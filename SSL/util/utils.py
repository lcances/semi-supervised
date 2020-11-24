import logging
import datetime
import random
import numpy as np
import torch
import time
from collections import Iterable, Sized
from zipfile import ZipFile, ZIP_DEFLATED 
import os

# TODO write q timer decorator that deppend on the logging level


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def timeit_logging(func):
    def decorator(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        logging.info("%s executed in: %.3fs" %
                     (func.__name__, time.time()-start_time))

    return decorator


def conditional_cache_v2(func):
    def decorator(*args, **kwargs):
        key_list = ",".join(map(str, args))
        key = kwargs.get("key", None)
        cached = kwargs.get("cached", None)

        if cached is not None and key is not None:
            if key not in decorator.cache.keys():
                decorator.cache[key] = func(*args, **kwargs)

            return decorator.cache[key]

        return func(*args, **kwargs)

    decorator.cache = dict()

    return decorator


def track_maximum():
    def func(key, value):
        if key not in func.max:
            func.max[key] = value
        else:
            if func.max[key] < value:
                func.max[key] = value
        return func.max[key]

    func.max = dict()
    return func


def get_datetime():
    now = datetime.datetime.now()
    return str(now)[:10] + "_" + str(now)[11:-7]


def get_model_from_name(model_name):
    import DCT.ubs8k.models as ubs8k_models
    import DCT.ubs8k.models_test as ubs8k_models_test
    import DCT.cifar10.models as cifar10_models
    import inspect

    all_members = []
    for module in [ubs8k_models, ubs8k_models_test, cifar10_models]:
        all_members += inspect.getmembers(module)

    for name, obj in all_members:
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if obj.__name__ == model_name:
                logging.info("Model loaded: %s" % model_name)
                return obj

    msg = "This model does not exist: %s\n" % model_name
    msg += "Available models are: %s" % [name for name,
                                         obj in all_members if inspect.isclass(obj) or inspect.isfunction(obj)]
    raise AttributeError("This model does not exist: %s " % msg)


def reset_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ZipCycle(Iterable, Sized):
    """
        Zip through a list of iterables and sized objects of different lengths.
        When a iterable smaller than the longest is over, this iterator is reset to the beginning.

        Example :
        r1 = range(1, 4)
        r2 = range(1, 6)
        iters = ZipCycle([r1, r2])
        for v1, v2 in iters:
            print(v1, v2)

        will print :
        1 1
        2 2
        3 3
        1 4
        2 5
    """

    def __init__(self, iterables: list):
        for iterable in iterables:
            if len(iterable) == 0:
                raise RuntimeError("An iterable is empty.")

        self._iterables = iterables
        self._len = max([len(iterable) for iterable in self._iterables])

    def __iter__(self) -> list:
        cur_iters = [iter(iterable) for iterable in self._iterables]
        cur_count = [0 for _ in self._iterables]

        for _ in range(len(self)):
            items = []

            for i, _ in enumerate(cur_iters):
                if cur_count[i] < len(self._iterables[i]):
                    item = next(cur_iters[i])
                    cur_count[i] += 1
                else:
                    cur_iters[i] = iter(self._iterables[i])
                    item = next(cur_iters[i])
                    cur_count[i] = 1
                items.append(item)

            yield items

    def __len__(self) -> int:
        return self._len


def create_bash_crossvalidation(nb_fold: int = 10):
    cross_validation = []
    end = nb_fold

    for i in range(nb_fold):
        train_folds = []
        start = i

        for i in range(nb_fold - 1):
            start = (start + 1) % nb_fold
            start = start if start != 0 else nb_fold
            train_folds.append(start)

        cross_validation.append("-t " + " ".join(map(str, train_folds)) + " -v %d" % end)
        end = (end % nb_fold) + 1
        end = end if end != 0 else nb_fold

    print(";".join(cross_validation))


def save_source_as_img(sourcepath: str):
    # Create a zip file of the current source code

    with ZipFile(sourcepath + ".zip", "w", compression=ZIP_DEFLATED, compresslevel=9) as myzip:
        myzip.write(sourcepath)

    # Read the just created zip file and store it into
    # a uint8 numpy array
    with open(sourcepath + ".zip", "rb") as myzip:
        zip_bin = myzip.read()

    zip_bin_n = np.array(list(map(int, zip_bin)), dtype=np.uint8)

    # Convert it into a 2d matrix
    desired_dimension = 500
    missing = desired_dimension - (zip_bin_n.size % desired_dimension)
    zip_bin_p = np.concatenate((zip_bin_n, np.array([0]*missing, dtype=np.uint8)))
    zip_bin_i = np.asarray(zip_bin_p).reshape((desired_dimension, zip_bin_p.size // desired_dimension))

    # Cleaning (remove zip file)
    os.remove(sourcepath + ".zip")

    return zip_bin_i, missing



# from PIL import Image
# 
# im = Image.open("tmp-232.png")
# source_bin = numpy.asarray(im, dtype=numpy.uint8)
# source_bin = source_bin[:, :, 0]
# 
# source_bin = source_bin.flatten()[:-232]
# 
# with open("student-teacher.ipynb.zip.bak", "wb") as mynewzip:
#     mynewzip.write(source_bin)