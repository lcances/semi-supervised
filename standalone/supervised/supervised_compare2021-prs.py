"""
Not MIXUP READY !
"""

import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import time
from typing import Union
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary
from SSL.util.model_loader import load_model
from SSL.util.loaders import load_dataset, load_optimizer, load_callbacks, load_preprocesser
from SSL.util.checkpoint import CheckPoint, mSummaryWriter
from SSL.util.utils import reset_seed, get_datetime, track_maximum, DotDict, get_train_format, get_lr
from SSL.util.mixup import MixUpBatchShuffle
from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage, MAP
from augmentation_utils.spec_augmentations import SpecAugment
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name='../../config/supervised/compare2021_prs.yaml')
def run(cfg: DictConfig) -> DictConfig:
    # keep the file directory as the current working directory
    os.chdir(hydra.utils.get_original_cwd())

    print(OmegaConf.to_yaml(cfg))
    print('current dir: ', os.getcwd())

    reset_seed(cfg.train_param.seed)

    # -------- Get the pre-processer --------
    train_transform, val_transform = load_preprocesser(cfg.dataset.dataset, "supervised")

    # -------- Get the dataset --------
    _, train_loader, val_loader = load_dataset(
        cfg.dataset.dataset,
        "supervised",

        dataset_root=cfg.path.dataset_root,
        supervised_ratio=cfg.train_param.supervised_ratio,
        batch_size=cfg.train_param.batch_size,
        train_folds=cfg.train_param.train_folds,
        val_folds=cfg.train_param.val_folds,

        train_transform=train_transform,
        val_transform=val_transform,

        num_workers=cfg.hardware.nb_cpu,  # With the cache enable, it is faster to have only one worker
        pin_memory=True,

        verbose=1
    )

    # The input shape of the data is used to generate the model
    input_shape = tuple(train_loader.dataset[0][0].shape)

    # -------- Prepare the model --------
    torch.cuda.empty_cache()
    model_func = load_model(cfg.dataset.dataset, cfg.model.model)
    model = model_func(input_shape=input_shape, num_classes=cfg.dataset.num_classes)
    model = model.cuda()
    summary(model, input_shape)

    if cfg.hardware.nb_gpu > 1:
        model = nn.DataParallel(model)

    # -------- Tensorboard and checkpoint --------
    # Prepare suffix
    # normale training parameters
    sufix_title = ''
    sufix_title += f'_{cfg.train_param.learning_rate}-lr'
    sufix_title += f'_{cfg.train_param.supervised_ratio}-sr'
    sufix_title += f'_{cfg.train_param.nb_iteration}-e'
    sufix_title += f'_{cfg.train_param.batch_size}-bs'
    sufix_title += f'_{cfg.train_param.seed}-seed'

    # mixup parameters
    if cfg.mixup.use:
        sufix_title += '_mixup'
        if cfg.mixup.max: sufix_title += "-max"
        if cfg.mixup.label: sufix_title += "-label"
        sufix_title += f"-{cfg.mixup.alpha}-a"

    # SpecAugment parameters
    if cfg.specaugment.use:
        sufix_title += '_specAugment'
        sufix_title += f'-{cfg.specaugment.time_drop_width}-tdw'
        sufix_title += f'-{cfg.specaugment.time_stripe_num}-tsn'
        sufix_title += f'-{cfg.specaugment.freq_drop_width}-fdw'
        sufix_title += f'-{cfg.specaugment.freq_stripe_num}-fsn'

    # -------- Tensorboard logging --------
    tensorboard_title = f'{get_datetime()}_{cfg.model.model}_{sufix_title}'
    log_dir = f'{cfg.path.tensorboard_path}/{tensorboard_title}'
    print('Tensorboard log at: ', log_dir)

    tensorboard = mSummaryWriter(log_dir=log_dir, comment=model_func.__name__)

    # -------- Optimizer, callbacks, loss and checkpoint --------
    optimizer = load_optimizer(cfg.dataset.dataset, "supervised", learning_rate=cfg.train_param.learning_rate, model=model)
    callbacks = load_callbacks(cfg.dataset.dataset, "supervised", optimizer=optimizer, nb_epoch=cfg.train_param.nb_iteration)
    loss_ce = nn.CrossEntropyLoss(reduction="mean")

    checkpoint_title = f'{cfg.model.model}_{sufix_title}'
    checkpoint_path = f'{cfg.path.checkpoint_path}/{checkpoint_title}'
    checkpoint = CheckPoint(model, optimizer, mode="max", name=checkpoint_path)

    # -------- Metrics and print formater --------
    metrics = DotDict(
        fscore_fn=FScore(),
        acc_fn=CategoricalAccuracy(),
        avg_fn=ContinueAverage(),
        mAP_fn=MAP()
    )

    val_metrics = DotDict(
        fscore_fn=FScore(),
        acc_fn=CategoricalAccuracy(),
        avg_fn=ContinueAverage(),
        mAP_fn=MAP()
    )

    maximum_tracker = track_maximum()

    reset_metrics = lambda m_d: [fn.reset() for fn in m_d.values()]
    m_ = lambda m: m.mean(size=100)  # running mean over the last 100 iteration

    header, train_formater, val_formater = get_train_format('compare2021-prs-sup')

    # -------- Augmentations ---------
    spec_augmenter = SpecAugment(time_drop_width=cfg.specaugment.time_drop_width,
                                 time_stripes_num=cfg.specaugment.time_stripe_num,
                                 freq_drop_width=cfg.specaugment.freq_drop_width,
                                 freq_stripes_num=cfg.specaugment.freq_stripe_num)

    mixup_fn = MixUpBatchShuffle(alpha=cfg.mixup.alpha, apply_max=cfg.mixup.max, mix_labels=cfg.mixup.label)

    # -------- Training and Validation function --------
    def train_fn(epoch, X, y, start_time) -> Union[float, float]:
        # aliases
        M = metrics
        T = tensorboard.add_scalar

        model.train()

        X = X.cuda().float()
        y = y.cuda().long()

        # apply augmentation if needed
        if cfg.mixup.use:
            X, y = mixup_fn(X, y)

        if cfg.specaugment.use:
            X = spec_augmenter(X)

        logits = model(X)
        loss = loss_ce(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.set_grad_enabled(False):

            pred = torch.softmax(logits, dim=1)
            pred_arg = torch.argmax(logits, dim=1)
            y_one_hot = F.one_hot(y, num_classes=cfg.dataset.num_classes)

            acc = M.acc_fn(pred_arg, y)
            fscore = M.fscore_fn(pred, y_one_hot)
            avg_ce = M.avg_fn(loss.item())

            # logs
            print(train_formater.format(
                "Training: ",
                epoch + 1,
                e, cfg.train_param.nb_iteration,
                "", m_(avg_ce),
                "", m_(acc), m_(fscore), 0.0,
                time.time() - start_time,
            ), end="\r")

        T("train/Lce", m_(avg_ce), epoch)
        T("train/f1", m_(fscore), epoch)
        T("train/acc", m_(acc), epoch)

        return avg_ce.value, fscore.value

    def val_fn(epoch: int) -> Union[float, float]:
        # aliases
        M = val_metrics
        T = tensorboard.add_scalar
        nb_batch = len(val_loader)

        start_time = time.time()
        print("")

        reset_metrics(val_metrics)
        model.eval()

        with torch.set_grad_enabled(False):
            for i, (X, y) in enumerate(val_loader):
                X = X.cuda().float()
                y = y.cuda().long()

                logits = model(X)
                loss = loss_ce(logits, y)

                pred = torch.softmax(logits, dim=1)
                pred_arg = torch.argmax(logits, dim=1)
                pred_one_hot = F.one_hot(pred_arg, num_classes=cfg.dataset.num_classes)
                y_one_hot = F.one_hot(y, num_classes=cfg.dataset.num_classes)

                acc = M.acc_fn(pred_arg, y).mean()
                fscore = M.fscore_fn(pred, y_one_hot).mean()
                avg_ce = M.avg_fn(loss.item()).mean()

                mAP = M.mAP_fn(pred_one_hot.cpu().reshape(-1), y_one_hot.cpu().reshape(-1)).mean()

                # logs
                print(val_formater.format(
                    "Validation: ",
                    epoch + 1,
                    i, nb_batch,
                    "", avg_ce,
                    "", acc, fscore, mAP,
                    time.time() - start_time
                ), end="\r")

        ("val/Lce", avg_ce, epoch)
        T("val/f1", fscore, epoch)
        T("val/acc", acc, epoch)
        T("val/mAP", mAP, epoch)

        T("hyperparameters/learning_rate", get_lr(optimizer), epoch)

        T("max/acc", maximum_tracker("acc", acc), epoch)
        T("max/f1", maximum_tracker("f1", fscore), epoch)
        T('max/mAP', maximum_tracker('mAP', mAP), epoch)

        return avg_ce, fscore, mAP

    # -------- Training loop --------
    if cfg.train_param.resume:
        checkpoint.load_last()

    start_iteration = checkpoint.epoch_counter
    end_iteration = cfg.train_param.nb_iteration

    train_iterator = iter(train_loader)
    start_time = time.time()

    print(header)
    for e in range(start_iteration, end_iteration):
        # Validation every 500 iteration
        if e % 500 == 0:
            val_avg_ce, val_fscore, val_mAP = val_fn(e)
            print('')
            checkpoint.step(val_mAP, iter=e)

            # apply all callbacks
            for c in callbacks:
                c.step()

            tensorboard.flush()

        # Perform train
        train_fn(e, *train_iterator.next(), start_time)

    # -------- Save the hyper parameters and the metrics --------
    hparams = {
        'dataset': cfg.dataset.dataset,
        'model': cfg.model.model,
        'supervised_ratio': cfg.train_param.supervised_ratio,
        'batch_size': cfg.train_param.batch_size,
        'nb_iteration': cfg.train_param.nb_iteration,
        'learning_rate': cfg.train_param.learning_rate,
        'seed': cfg.train_param.seed,
        'mixup': cfg.mixup.use,
        'mixup-alpha': cfg.mixup.alpha,
        'mixup-max': cfg.mixup.max,
        'mixup-label': cfg.mixup.label,
        'specaugment(sa)': cfg.specaugment.use,
        'sa_time_drop_width': cfg.specaugment.time_drop_width,
        'sa_time_stripe_num': cfg.specaugment.time_stripe_num,
        'sa_freq_drop_width': cfg.specaugment.freq_drop_width,
        'sa_freq_stripe_num': cfg.specaugment.freq_stripe_num,
    }

    # convert all value to str
    hparams = dict(zip(hparams.keys(), map(str, hparams.values())))

    final_metrics = {
        "max_acc": maximum_tracker.max["acc"],
        "max_f1": maximum_tracker.max["f1"],
    }

    tensorboard.add_hparams(hparams, final_metrics)

    tensorboard.flush()
    tensorboard.close()


if __name__ == '__main__':
    run()
