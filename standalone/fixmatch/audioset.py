import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import time
from typing import Union
import numpy
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from SSL.util.model_loader import load_model
from SSL.util.loaders import load_dataset, load_optimizer, load_callbacks, load_preprocesser
from SSL.util.checkpoint import CheckPoint, mSummaryWriter
from SSL.util.utils import reset_seed, get_datetime, track_maximum, DotDict, cache_to_disk, get_train_format, get_lr
from SSL.util.mixup import MixUpBatchShuffle
from SSL.loss import FixMatchLoss
from SSL.ramps import Warmup, sigmoid_rampup
from metric_utils.metrics import BinaryAccuracy, FScore, ContinueAverage, MAP
from augmentation_utils.spec_augmentations import SpecAugment
from torch.cuda import empty_cache
from torchsummary import summary
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name='../../config/fixmatch/audioset-unbalanced.yaml')
def run(cfg: DictConfig) -> DictConfig:
    # keep the file directory as the current working directory
    os.chdir(hydra.utils.get_original_cwd())

    print(OmegaConf.to_yaml(cfg))
    print('current dir: ', os.getcwd())
    
    # reset the seed
    reset_seed(cfg.train_param.seed)

     # -------- Get the pre-processer --------
    train_transform, val_transform = load_preprocesser(cfg.dataset.dataset, "fixmatch")

    # -------- Get the dataset --------
    _, train_loader, val_loader = load_dataset(
        cfg.dataset.dataset,
        "fixmatch",

        dataset_root=cfg.path.dataset_root,
        supervised_ratio=cfg.train_param.supervised_ratio,
        batch_size=cfg.train_param.batch_size,
        train_folds=cfg.train_param.train_folds,
        val_folds=cfg.train_param.val_folds,

        train_transform=train_transform,
        val_transform=val_transform,

        num_workers=cfg.hardware.nb_cpu,  # With the cache enable, it is faster to have only one worker
        pin_memory=True,
        seed=cfg.train_param.seed,

        verbose=1
    )

    # The input shape of the data is used to generate the model
    input_shape = tuple(train_loader._iterables[0].dataset[0][0].shape)
    
    # -------- Prepare the model --------
    torch.cuda.empty_cache()
    model_func = load_model(cfg.dataset.dataset, cfg.model.model)
    model = model_func(input_shape=input_shape, num_classes=cfg.dataset.num_classes)
    model = model.cuda()
    summary(model, input_shape)

    if cfg.hardware.nb_gpu > 1:
        model = nn.DataParallel(model)
        
    # Prepare suffix
    # normale training parameters
    sufix_title = ''
    sufix_title += f'_{cfg.train_param.learning_rate}-lr'
    sufix_title += f'_{cfg.train_param.supervised_ratio}-sr'
    sufix_title += f'_{cfg.train_param.batch_size}-bs'
    sufix_title += f'_{cfg.train_param.seed}-seed'

    # fixmatch parameters
    sufix_fm = f'_{cfg.fixmatch.mask_threshold}-mTh'
    sufix_fm += f'_{cfg.fixmatch.guess_threshold}-gTh'
    sufix_fm += f'_{cfg.fixmatch.lambda_s}-ls'
    sufix_fm += f'_{cfg.fixmatch.lambda_u}-lu'

    # mixup parameters
    sufix_mixup = ''
    if cfg.mixup.use:
        sufix_mixup = '_mixup'
        if cfg.mixup.max: sufix_mixup += "-max"
        if cfg.mixup.label: sufix_mixup += "-label"
        sufix_mixup += f"-{cfg.mixup.alpha}-a"

    # SpecAugment parameters
    sufix_sa = ''
    if cfg.specaugment.use:
        sufix_sa = '_specAugment'
        sufix_sa += f'-{cfg.specaugment.time_drop_width}-tdw'
        sufix_sa += f'-{cfg.specaugment.time_stripe_num}-tsn'
        sufix_sa += f'-{cfg.specaugment.freq_drop_width}-fdw'
        sufix_sa += f'-{cfg.specaugment.freq_stripe_num}-fsn'
        
    # -------- Tensorboard logging --------
    tensorboard_sufix = sufix_title + f'_{cfg.train_param.nb_iteration}-e' + sufix_fm + sufix_sa + f'__{cfg.path.sufix}'
    tensorboard_title = f'{get_datetime()}_{cfg.model.model}_{tensorboard_sufix}'
    log_dir = f'{cfg.path.tensorboard_path}/{cfg.model.model}/{tensorboard_title}'
    print('Tensorboard log at: ', log_dir)

    tensorboard = mSummaryWriter(log_dir=log_dir, comment=model_func.__name__)
    
    # -------- Optimizer, callbacks, loss and checkpoint --------
    optimizer = load_optimizer(cfg.dataset.dataset, 'fixmatch', learning_rate=cfg.train_param.learning_rate, model=model)
    callbacks = load_callbacks(cfg.dataset.dataset, 'fixmatch', optimizer=optimizer, nb_epoch=cfg.train_param.nb_iteration)
    loss_sup = nn.BCEWithLogitsLoss(reduction="sum")
    loss_unsup = nn.BCEWithLogitsLoss(reduction="none")

    lambda_u = Warmup(cfg.fixmatch.lambda_u, cfg.fixmatch.warmup_length, sigmoid_rampup)

    checkpoint_sufix = sufix_title + sufix_mixup + sufix_sa + f'__{cfg.path.sufix}'
    checkpoint_title = f'{cfg.model.model}_{checkpoint_sufix}'
    checkpoint_path = f'{cfg.path.checkpoint_path}/{cfg.model.model}/{checkpoint_title}'
    checkpoint = CheckPoint(model, optimizer, mode="max", name=checkpoint_path)
    
    # -------- Metrics and print formater --------
    metrics = DotDict(
        fscore_s=FScore(),
        fscore_u=FScore(),
        acc_s=BinaryAccuracy(),
        acc_u=BinaryAccuracy(),
        avg_fn=ContinueAverage(),
    )

    val_metrics = DotDict(
        fscore_s=FScore(),
        acc_s=BinaryAccuracy(),
        mAP_fn=MAP(),
        avg_fn=ContinueAverage(),
    )

    maximum_tracker = track_maximum()
    
    header, train_formater, val_formater = get_train_format('audioset-fixmatch')
    
    # -------- Augmentations ---------
    spec_augmenter = SpecAugment(time_drop_width=cfg.specaugment.time_drop_width,
                                  time_stripes_num=cfg.specaugment.time_stripe_num,
                                  freq_drop_width=cfg.specaugment.freq_drop_width,
                                  freq_stripes_num=cfg.specaugment.freq_stripe_num)

    mixup_fn = MixUpBatchShuffle(alpha=cfg.mixup.alpha, apply_max=cfg.mixup.max, mix_labels=cfg.mixup.label)
    
    # -------- Training and Validation function --------
    def guess_label(x_uw: Tensor) -> (Tensor, Tensor):    
        logits_uw = model(x_uw)
        pred_uw = torch.sigmoid(logits_uw)

        nb_classes = cfg.dataset.num_classes
        labels_u = (pred_uw > cfg.fixmatch.guess_threshold).float()
    #     labels_u = F.one_hot(pred_uw, cfg.dataset.num_classes)
        return labels_u, pred_uw


    def confidence_mask(pred: Tensor, threshold: float, dim: int) -> Tensor:
        max_values, _ = pred.max(dim=dim)
        return (max_values > threshold).float()

    m_ = lambda x: x.mean(size=100)
    
    def train(epoch, Sw, Uw, Us, start_time):
        # aliases
        M = metrics
        T = tensorboard.add_scalar

        model.train()

        x_sw, y_sw = Sw  # Supervised weak augmented
        x_uw, y_uw = Uw  # Unsupervised weak augmented
        x_us, y_us = Us  # Unsupervised strong augmented

        x_sw, y_sw = x_sw.cuda().float(), y_sw.cuda().float()
        x_uw, y_uw = x_uw.cuda().float(), y_uw.cuda().float()
        x_us, y_us = x_us.cuda().float(), y_us.cuda().float()

        # Apply mixup is needed
        if cfg.mixup.use:
            pass

        # Apply specaugment if needed
        if cfg.specaugment.use:
            pass

        # Use guess u label with prediction of weak augmentation of u
        with torch.no_grad():
            pseudo_y_uw, pred_uw = guess_label(x_uw)
            mask = confidence_mask(pseudo_y_uw, cfg.fixmatch.mask_threshold, dim=1)

        optimizer.zero_grad()

        # Compute predictions
        logits_sw = model(x_sw)
        logits_uw = model(x_uw)
        logits_us = model(x_us)

        # Update model
        loss_s = loss_sup(logits_sw, y_sw)
        loss_u = loss_unsup(logits_us, pseudo_y_uw)
        loss_u = torch.sum(loss_u, dim=-1)
        loss_u = torch.sum(loss_u * mask)
        loss = loss_s + lambda_u() * loss_u
        loss.backward()

        optimizer.step()

        # Compute metrics
        with torch.no_grad():
            fscore_s = M.fscore_s(torch.sigmoid(logits_sw), y_sw)
            acc_s = M.acc_s(logits_sw, y_sw)
            acc_u = M.acc_u(logits_uw, y_uw)
            fscore_uw = M.fscore_u(torch.sigmoid(logits_uw), pseudo_y_uw)  # Use true label for monitoring purpose
            avg_ce = M.avg_fn(loss.item())

            # logs
            print(train_formater.format(
                "Training: ",
                epoch + 1,
                e, cfg.train_param.nb_iteration,
                "", m_(avg_ce),
                "", m_(acc_s), m_(fscore_s), m_(acc_u), m_(fscore_uw), 0.0,
                time.time() - start_time,
            ), end="\r")

            T("train/loss",   loss.item(), epoch)
            T("train/loss_s", loss_s.item(), epoch)
            T("train/loss_u", loss_u.item(), epoch)
            T("train/labels_used", mask.mean().item(), epoch)

            T('train/fscore_s', m_(fscore_s), epoch)
            T('train/fscore_uw', m_(fscore_uw), epoch)
            T('train/acc_s', m_(acc_s), epoch)
            T('train/acc_u', m_(acc_u), epoch)

    def val(epoch):
        # aliases
        M = val_metrics
        T = tensorboard.add_scalar
        nb_batch = len(val_loader)

        start_time = time.time()
        print("")

        model.eval()

        with torch.set_grad_enabled(False):
            for i, (X, y) in enumerate(val_loader):
                X = X.cuda().float()
                y = y.cuda().float()

                logits = model(X)
                loss = loss_sup(logits, y)

                pred = torch.sigmoid(logits)
                fscore = M.fscore_s(pred, y)
                acc = M.acc_s(pred, y)
                mAP = M.mAP_fn(pred.cpu().reshape(-1), y.cpu().reshape(-1))
                avg_ce = M.avg_fn(loss.item())

                # logs
                print(val_formater.format(
                    "Validation: ",
                    epoch + 1,
                    i, nb_batch,
                    "", m_(avg_ce),
                    "", m_(acc), m_(fscore), 0.0, 0.0, m_(mAP),
                    time.time() - start_time
                ), end="\r")

        T("val/Lce", m_(avg_ce), epoch)
        T("val/f1", m_(fscore), epoch)
        T("val/acc", m_(acc), epoch)
        T("val/mAP", m_(mAP), epoch)

        T("hyperparameters/learning_rate", get_lr(optimizer), epoch)

        T("max/acc", maximum_tracker("acc", m_(acc)), epoch)
        T("max/f1", maximum_tracker("f1", m_(fscore)), epoch)
        T('max/mAP', maximum_tracker('mAP', m_(mAP)), epoch)

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
            val_avg_ce, val_fscore, val_mAP = val(e)
            print('')
            checkpoint.step(m_(val_mAP))
            tensorboard.flush()

        # Perform train
        train(e, *next(train_iterator), start_time)
        
    # -------- Save the hyper parameters and the metrics --------
    hparams = {
        'dataset': cfg.dataset.dataset,
        'model': cfg.model.model,
        'supervised_ratio': cfg.train_param.supervised_ratio,
        'batch_size': cfg.train_param.batch_size,
        'nb_iteration': cfg.train_param.nb_iteration,
        'learning_rate': cfg.train_param.learning_rate,
        'seed': cfg.train_param.seed,

        'threshold': cfg.fixmatch.threshold,
        'lambda_s': cfg.fixmatch.lambda_s,
        'lambda_u': cfg.fixmatch.lambda_u,

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