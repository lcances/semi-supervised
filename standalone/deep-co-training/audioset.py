import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
# from torch.utils.tensorboard import SummaryWriter
from advertorch.attacks import GradientSignAttack
from metric_utils.metrics import BinaryAccuracy, FScore, ContinueAverage, MAP
from SSL.util.checkpoint import CheckPoint, mSummaryWriter
from SSL.util.utils import reset_seed, get_datetime, track_maximum, get_lr, get_training_printers, DotDict
from SSL.util.model_loader import load_model
from SSL.util.loaders import load_dataset, load_optimizer, load_callbacks, load_preprocesser
from SSL.util.mixup import MixUpBatchShuffle
from SSL.ramps import Warmup, sigmoid_rampup
from SSL.loss.losses import loss_cot, loss_diff, loss_sup, Activation
from SSL.loss.losses import DCTSupWithLogitsLoss, DCTDiffWithLogitsLoss, DCTCotWithLogitsLoss, Activation, ValidLoss
from torchsummary import summary
from SSL.util.mixup import MixUpBatchShuffle
from augmentation_utils.spec_augmentations import SpecAugment
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name='../../config/deep-co-training/audioset.yaml')
def run(cfg: DictConfig) -> DictConfig:
    # keep the file directory as the current working directory
    os.chdir(hydra.utils.get_original_cwd())

    print(OmegaConf.to_yaml(cfg))
    print('current dir: ', os.getcwd())

    reset_seed(cfg.train_param.seed)

    # -------- Get the pre-processer --------
    train_transform, val_transform = load_preprocesser(cfg.dataset.dataset, "dct")

    # -------- Get the dataset --------
    manager, train_loader, val_loader = load_dataset(
        cfg.dataset.dataset,
        "dct",

        dataset_root=cfg.path.dataset_root,
        supervised_ratio=cfg.train_param.supervised_ratio,
        batch_size=cfg.train_param.batch_size,
        train_folds=cfg.train_param.train_folds,
        val_folds=cfg.train_param.val_folds,

        train_transform=train_transform,
        val_transform=val_transform,

        num_workers=cfg.hardware.nb_cpu,
        pin_memory=True,

        verbose=1
    )

    # The input shape of the data is used to generate the model
    input_shape = train_loader._iterables[0].dataset[0][0].shape

    # -------- Prepare the model --------
    torch.cuda.empty_cache()
    model_func = load_model(cfg.dataset.dataset, cfg.model.model)

    commun_args = dict(
        manager=manager,
        num_classes=cfg.dataset.num_classes,
        input_shape=list(input_shape),
    )

    m1 = model_func(**commun_args)
    m2 = model_func(**commun_args)

    m1 = m1.cuda()
    m2 = m2.cuda()

    if cfg.hardware.nb_gpu > 1:
        m1 = DataParallel(m1)
        m2 = DataParallel(m2)

    summary(m1, input_shape)

    # -------- Tensorboard and checkpoint --------
        # -- Prepare suffix
    sufix_title = ''
    sufix_title += f'_{cfg.train_param.learning_rate}-lr'
    sufix_title += f'_{cfg.train_param.supervised_ratio}-sr'
    sufix_title += f'_{cfg.train_param.batch_size}-bs'
    sufix_title += f'_{cfg.train_param.seed}-seed'

    # deep co training parameters
    sufix_title += f'_{cfg.dct.epsilon}eps'
    sufix_title += f'-{cfg.dct.warmup_length}wl'
    sufix_title += f'-{cfg.dct.lambda_cot_max}lcm'
    sufix_title += f'-{cfg.dct.lambda_diff_max}ldm'
    
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
    tensorboard_sufix = sufix_title + f'_{cfg.train_param.nb_iteration}-e' + sufix_mixup + sufix_sa + f'__{cfg.path.sufix}'
    tensorboard_title = f'{get_datetime()}_{cfg.model.model}_{tensorboard_sufix}'
    log_dir = f'{cfg.path.tensorboard_path}/{cfg.model.model}/{tensorboard_title}'
    print('Tensorboard log at: ', log_dir)

    tensorboard = mSummaryWriter(log_dir=log_dir, comment=model_func.__name__)

    # -------- Optimizer, callbacks, loss, adversarial generator and checkpoint --------
    optimizer = load_optimizer(cfg.dataset.dataset, "dct", model1=m1, model2=m2, learning_rate=cfg.train_param.learning_rate)
    callbacks = load_callbacks(cfg.dataset.dataset, "dct", optimizer=optimizer, nb_epoch=cfg.train_param.nb_iteration)
    
    losses = DotDict({
        'sup': DCTSupWithLogitsLoss(reduction='mean', sub_loss=ValidLoss.BINARY_CROSS_ENTROPY),
        'cot': DCTCotWithLogitsLoss(activation=Activation.SIGMOID),
        'diff': DCTDiffWithLogitsLoss(activation=Activation.SIGMOID),
    })

    # Checkpoint
    checkpoint_sufix = sufix_title + sufix_mixup + sufix_sa + f'__{cfg.path.sufix}'
    checkpoint_title = f'{cfg.model.model}_{checkpoint_sufix}'
    checkpoint_path = f'{cfg.path.checkpoint_path}/{cfg.model.model}/{checkpoint_title}'
    checkpoint = CheckPoint([m1, m2], optimizer, mode="max", name=checkpoint_path)

    # define the warmups & add them to the callbacks (for update)
    lambda_cot = Warmup(cfg.dct.lambda_cot_max, cfg.dct.warmup_length, sigmoid_rampup)
    lambda_diff = Warmup(cfg.dct.lambda_diff_max, cfg.dct.warmup_length, sigmoid_rampup)
    callbacks += [lambda_cot, lambda_diff]

    # adversarial generation
    adv_generator_1 = GradientSignAttack(
        m1, loss_fn=nn.BCEWithLogitsLoss(reduction="sum"),
        eps=cfg.dct.epsilon, clip_min=-np.inf, clip_max=np.inf, targeted=False
    )

    adv_generator_2 = GradientSignAttack(
        m2, loss_fn=nn.BCEWithLogitsLoss(reduction="sum"),
        eps=cfg.dct.epsilon, clip_min=-np.inf, clip_max=np.inf, targeted=False
    )

    # -------- Metrics and print formater --------
    metrics = DotDict({
        'sup': DotDict({
            'fscore_1': FScore(),
            'fscore_2': FScore(),
            'map_1': MAP(),
            'map_2': MAP(),
        }),

        'unsup': DotDict({
            'fscore_1': FScore(),
            'fscore_2': FScore(),
            'map_1': MAP(),
            'map_2': MAP(),
        }),

        'avg': DotDict({
            'total': ContinueAverage(),
            'sup': ContinueAverage(),
            'cot': ContinueAverage(),
            'diff': ContinueAverage(),
        })
    })

    def reset_metrics(metrics: dict):
        for k, v in metrics.items():
            if isinstance(v, dict):
                reset_metrics(v)

            else:
                v.reset()

    maximum_tracker = track_maximum()

    header, train_formater, val_formater = get_training_printers(metrics.avg, metrics.sup)

    # -------- Augmentations ---------
    spec_augmenter = SpecAugment(time_drop_width=cfg.specaugment.time_drop_width,
                                 time_stripes_num=cfg.specaugment.time_stripe_num,
                                 freq_drop_width=cfg.specaugment.freq_drop_width,
                                 freq_stripes_num=cfg.specaugment.freq_stripe_num)

    mixup_fn = MixUpBatchShuffle(alpha=cfg.mixup.alpha, apply_max=cfg.mixup.max, mix_labels=cfg.mixup.label)

    # -------- Training and Validation function --------
    m_ = lambda x: x.mean(size=100)
    S = nn.Sigmoid()

    def train(iteration, S1, S2, U, start_time):
        m1.train()
        m2.train()

        x_s1, y_s1 = S1
        x_s2, y_s2 = S2
        x_u, y_u = U

        # Apply mixup if needed, otherwise no mixup.
        if cfg.mixup.use:
            x_u, y_u = mixup_fn(x_u, y_u)

        # Apply spec augmentation if needed
        if cfg.specaugment.use:
            x_s1 = spec_augmenter(x_s1)
            x_s2 = spec_augmenter(x_s2)
            x_u = spec_augmenter(x_u)

        x_s1, x_s2, x_u = x_s1.cuda().float(), x_s2.cuda().float(), x_u.cuda().float()
        y_s1, y_s2, y_u = y_s1.cuda().float(), y_s2.cuda().float(), y_u.cuda().float()

        # Prediction
        logits_s1 = m1(x_s1)
        logits_s2 = m2(x_s2)
        logits_u1 = m1(x_u)
        logits_u2 = m2(x_u)

        # pseudo labels of U
        # TODO most probaby need to apply a threshold
        pseudo_yu1 = torch.sigmoid(logits_u1)
        pseudo_yu2 = torch.sigmoid(logits_u2)

        # ======== Generate adversarial examples ========
        # fix batchnorm ----
        m1.eval()
        m2.eval()

        # generate adversarial examples ----
        adv_data_s1 = adv_generator_1.perturb(x_s1, y_s1)
        adv_data_u1 = adv_generator_1.perturb(x_u, pseudo_yu1)

        adv_data_s2 = adv_generator_2.perturb(x_s2, y_s2)
        adv_data_u2 = adv_generator_2.perturb(x_u, pseudo_yu2)

        m1.train()
        m2.train()

        # predict adversarial examples ----
        adv_logits_s1 = m1(adv_data_s2)
        adv_logits_s2 = m2(adv_data_s1)

        adv_logits_u1 = m1(adv_data_u2)
        adv_logits_u2 = m2(adv_data_u1)

        # ======== calculate the differents loss ========
        # zero the parameter gradients ----
        for p in m1.parameters(): p.grad = None  # zero grad
        for p in m2.parameters(): p.grad = None

        # losses ----
        l_sup = losses.sup((logits_s1, y_s1), (logits_s2, y_s2))
        l_cot = losses.cot(logits_u1, logits_u2)
        l_diff = losses.diff(
            (logits_s1, adv_logits_s1), (logits_s2, adv_logits_s2),
            (logits_u1, adv_logits_u1), (logits_u2, adv_logits_u2)
        )

        total_loss = l_sup + lambda_cot() * l_cot + lambda_diff() * l_diff
        total_loss.backward()
        optimizer.step()

        # ======== Calc the metrics ========
        with torch.set_grad_enabled(False):
            # Fscore
            fscore_s1 = m_(metrics.sup.fscore_1(S(logits_s1), y_s1))
            fscore_s2 = m_(metrics.sup.fscore_2(S(logits_s2), y_s2))
            fscore_u1 = m_(metrics.unsup.fscore_1(S(logits_u1), y_u))
            fscore_u2 = m_(metrics.unsup.fscore_2(S(logits_u2), y_u))


            # Running average of the losses
            avg_total = m_(metrics.avg.total(total_loss.item()))
            avg_sup = m_(metrics.avg.sup(l_sup.item()))
            avg_cot = m_(metrics.avg.cot(l_cot.item()))
            avg_diff = m_(metrics.avg.diff(l_diff.item()))

            # logs
            print(train_formater.format(
                1, iteration, cfg.train_param.nb_iteration,
                avg_total, avg_sup, avg_cot, avg_diff,
                fscore_s1, fscore_s2, 0.0, 0.0,
                time.time() - start_time),
                end='\r'
            )

        # Using tensorboard to monitor loss and acc\n",
        tensorboard.add_scalar('train/total_loss', avg_total, iteration)
        tensorboard.add_scalar('train/Lsup', avg_sup, iteration)
        tensorboard.add_scalar('train/Lcot', avg_cot, iteration)
        tensorboard.add_scalar('train/Ldiff', avg_diff, iteration)

        tensorboard.add_scalar("train/fscore_s1", fscore_s1, iteration)
        tensorboard.add_scalar("train/fscore_s2", fscore_s2, iteration)
        tensorboard.add_scalar("train/fscore_u1", fscore_u1, iteration)
        tensorboard.add_scalar("train/fscore_u2", fscore_u2, iteration)

    def val(iteration):
        start_time = time.time()
        nb_batch = len(val_loader)
        print("")

        reset_metrics(metrics)
        m1.eval()
        m2.eval()

        with torch.set_grad_enabled(False):
            for i, (X, y) in enumerate(val_loader):
                x = X.cuda().float()
                y = y.cuda().float()

                logits_1 = m1(x)
                logits_2 = m2(x)

                # losses ----
                l_sup = losses.sup((logits_1, y), (logits_2, y))

                # ======== Calc the metrics ========
                fscore_1 = m_(metrics.sup.fscore_1(S(logits_1), y))
                fscore_2 = m_(metrics.sup.fscore_2(S(logits_2), y))
                mAP_1 = m_(metrics.sup.map_1(S(logits_1).cpu().reshape(-1), y.cpu().reshape(-1)))
                mAP_2 = m_(metrics.sup.map_2(S(logits_2).cpu().reshape(-1), y.cpu().reshape(-1)))
                avg_sup = m_(metrics.avg.sup(l_sup.item()))

                # logs
                print(val_formater.format(
                    iteration+1, i, nb_batch,
                    0.0, avg_sup, 0.0, 0.0,
                    fscore_1, fscore_2, mAP_1, mAP_2,
                    time.time() - start_time,
                ), end="\r")

        tensorboard.add_scalar('val/Lsup', avg_sup, iteration)

        tensorboard.add_scalar("val/fscore_1", fscore_1, iteration)
        tensorboard.add_scalar("val/fscore_2", fscore_2, iteration)
        tensorboard.add_scalar("val/mAP_1", mAP_1, iteration)
        tensorboard.add_scalar("val/mAP_2", mAP_2, iteration)

        tensorboard.add_scalar("max/fscore_1", maximum_tracker("fscore_1", fscore_1), iteration)
        tensorboard.add_scalar("max/fscore_2", maximum_tracker("fscore_2", fscore_2), iteration)
        tensorboard.add_scalar("max/mAP_1", maximum_tracker("mAP_1", mAP_1), iteration)
        tensorboard.add_scalar("max/mAP_2", maximum_tracker("mAP_2", mAP_2), iteration)

        tensorboard.add_scalar("detail_hyperparameters/lambda_cot", lambda_cot(), iteration)
        tensorboard.add_scalar("detail_hyperparameters/lambda_diff", lambda_diff(), iteration)
        tensorboard.add_scalar("detail_hyperparameters/learning_rate", get_lr(optimizer), iteration)

        return mAP_1, mAP_2

    # -------- Training loop ------
    print(header)

    if cfg.train_param.resume:
        checkpoint.load_last()

    start_iteration = checkpoint.epoch_counter
    end_iteration = cfg.train_param.nb_iteration

    train_iterator = iter(train_loader)
    start_time = time.time()

    for i in range(start_iteration, end_iteration):
        # Validation every 500 iteration
        if i % 500 == 0:
            mAP_1, mAP_2 = val(i)
            print('')
            checkpoint.step((mAP_1 + mAP_2)/2.0, iter=i)
            tensorboard.flush()

        # Perform train
        train(i, *next(train_iterator), start_time)
        
        # Apply iteration callbakcs
        for c in callbacks:
            c.step()

    # -------- Save the hyper parameters and the metrics --------
    hparams = {
        'dataset': cfg.dataset.dataset,
        'model': cfg.model.model,
        'supervised_ratio': cfg.train_param.supervised_ratio,
        'batch_size': cfg.train_param.batch_size,
        'nb_iteration': cfg.train_param.nb_iteration,
        'learning_rate': cfg.train_param.learning_rate,
        'seed': cfg.train_param.seed,

        'epsilon': cfg.dct.epsilon,
        'warmup_length': cfg.dct.warmup_length,
        'lamda_cot_max': cfg.dct.lambda_cot_max,
        'lamda_diff_max': cfg.dct.lambda_diff_max,

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
        "fscore_1": maximum_tracker.max["fscore_1"],
        "fscore_2": maximum_tracker.max["fscore_2"],
        "mAP_1": maximum_tracker.max["mAP_1"],
        "mAP_2": maximum_tracker.max["mAP_2"],
    }

    tensorboard.add_hparams(hparams, final_metrics)

    tensorboard.flush()
    tensorboard.close()


if __name__ == '__main__':
    run()
