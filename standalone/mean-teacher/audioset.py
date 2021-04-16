import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import hydra
from omegaconf import DictConfig, OmegaConf
from SSL.util.loaders import load_dataset, load_optimizer, load_callbacks, load_preprocesser
from SSL.util.model_loader import load_model
from SSL.util.checkpoint import CheckPoint, mSummaryWriter
from SSL.util.mixup import MixUpBatchShuffle
from SSL.util.utils import reset_seed, get_datetime, DotDict, track_maximum, get_lr
from SSL.util.utils import get_training_printers, DotDict
from SSL.ramps import Warmup, sigmoid_rampup
from SSL.loss.losses import JensenShanon
from metric_utils.metrics import BinaryAccuracy, FScore, ContinueAverage, MAP
from augmentation_utils.spec_augmentations import SpecAugment



@hydra.main(config_name='../../config/mean-teacher/audioset.yaml')
def run(cfg: DictConfig) -> DictConfig:
    # keep the file directory as the current working directory
    os.chdir(hydra.utils.get_original_cwd())

    print(OmegaConf.to_yaml(cfg))
    print('current dir: ', os.getcwd())

    reset_seed(cfg.train_param.seed)

    # -------- Get the pre-processer --------
    train_transform, val_transform = load_preprocesser(cfg.dataset.dataset, "mean-teacher")

    # -------- Get the dataset --------
    manager, train_loader, val_loader = load_dataset(
        cfg.dataset.dataset,
        "mean-teacher",

        dataset_root=cfg.path.dataset_root,
        supervised_ratio=cfg.train_param.supervised_ratio,
        batch_size=cfg.train_param.batch_size,
        train_folds=cfg.train_param.train_folds,
        val_folds=cfg.train_param.val_folds,

        train_transform=train_transform,
        val_transform=val_transform,

        num_workers=cfg.hardware.nb_cpu,
        pin_memory=True,

        verbose=1)

    # The input shape of the data is used to generate the model
    input_shape = tuple(train_loader._iterables[0].dataset[0][0].shape)

    # -------- Prepare the model --------
    torch.cuda.empty_cache()

    model_func = load_model(cfg.dataset.dataset, cfg.model.model)

    student = model_func(input_shape=input_shape, num_classes=cfg.dataset.num_classes)
    teacher = model_func(input_shape=input_shape, num_classes=cfg.dataset.num_classes)

    # We do not need gradient for the teacher model
    for p in teacher.parameters():
        p.detach()

    student = student.cuda()
    teacher = teacher.cuda()

    if cfg.hardware.nb_gpu > 1:
        student = nn.DataParallel(student)
        teacher = nn.DataParallel(teacher)

    summary(student, input_shape)

    # -------- Tensorboard and checkpoint --------
    # -- Prepare suffix
    sufix_title = ''
    sufix_title += f'_{cfg.train_param.learning_rate}-lr'
    sufix_title += f'_{cfg.train_param.supervised_ratio}-sr'
    sufix_title += f'_{cfg.train_param.batch_size}-bs'
    sufix_title += f'_{cfg.train_param.seed}-seed'

    # mean teacher parameters
    sufix_title += f'_{cfg.mt.alpha}a'
    sufix_title += f'-{cfg.mt.warmup_length}wl'
    sufix_title += f'-{cfg.mt.lambda_ccost_max}lcm'
    sufix_title += f'-{cfg.mt.activation}act'
    sufix_title += f'-{cfg.mt.ccost_method}'

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

    # -------- Optimizer, callbacks, loss and checkpoint --------
    optimizer = load_optimizer(cfg.dataset.dataset, "mean-teacher", student=student, learning_rate=cfg.train_param.learning_rate)
    callbacks = load_callbacks(cfg.dataset.dataset, "mean-teacher", optimizer=optimizer, nb_epoch=cfg.train_param.nb_iteration)
    loss_ce = nn.BCEWithLogitsLoss(reduction='mean')

    # Unsupervised loss
    if cfg.mt.ccost_method == "mse":
        consistency_cost = nn.MSELoss(reduction="mean")
    elif cfg.mt.ccost_method == "js":
        consistency_cost = JensenShanon
    else:
        raise ValueError(f'ccost methods can either be "mse" (Mean Square Error) or "js" (Jensen Shanon). ccost_method={cfg.mt.ccost_method}')

    # Warmups
    lambda_cost = Warmup(cfg.mt.lambda_ccost_max, cfg.mt.warmup_length, sigmoid_rampup)
    callbacks += [lambda_cost]

    checkpoint_sufix = sufix_title + sufix_mixup + sufix_sa + f'__{cfg.path.sufix}'
    checkpoint_title = f'{cfg.model.model}_{checkpoint_sufix}'
    checkpoint_path = f'{cfg.path.checkpoint_path}/{cfg.model.model}/{checkpoint_title}'
    checkpoint = CheckPoint([student, teacher], optimizer, mode="max", name=checkpoint_path)

    # -------- Metrics and print formater --------
    metrics = DotDict({
        'sup': DotDict({
            'acc_s': BinaryAccuracy(),
            'acc_t': BinaryAccuracy(),
            'fscore_s': FScore(),
            'fscore_t': FScore(),
            'map_s': MAP(),
            'map_t': MAP(),
        }),

        'unsup': DotDict({
            'acc_s': BinaryAccuracy(),
            'acc_t': BinaryAccuracy(),
            'fscore_s': FScore(),
            'fscore_t': FScore(),
            'map_s': MAP(),
            'map_t': MAP(),
        }),

        'avg': DotDict({
            'sce': ContinueAverage(),
            'tce': ContinueAverage(),
            'ccost': ContinueAverage(),
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
    # use softmax or not
    activation_fn = lambda x: x
    if cfg.mt.activation == 'sigmoid':
        activation_fn = nn.Sigmoid()
    elif cfg.mt.activation == 'softmax':
        activation_fn = nn.Softmax(dim=1)
    else:
        raise ValueError(f'activation {cfg.mt.activation} is not available. available: [sigmoid | sofmax]')

    # update the teacher using exponentiel moving average
    def update_teacher_model(student_model, teacher_model, alpha, epoch):

        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (epoch + 1), alpha)

        for param, ema_param in zip(student_model.parameters(), teacher_model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def train(iteration, S, U, start_time):
        student.train()

        x_s, y_s = S
        x_u, y_u = U

        x_s, x_u = x_s.cuda().float(), x_u.cuda().float()
        y_s, y_u = y_s.cuda().float(), y_u.cuda().float()

        # Apply mixup if needed, otherwise no mixup.
        if cfg.mixup.use:
            n_x_s, _ = mixup_fn(x_s, y_s)
            n_x_u, _ = mixup_fn(x_u, y_u)
        else:
            n_x_s, n_x_u = x_s, x_u

        # Apply specaugmentation if needed
        if cfg.specaugment.use:
            x_s = spec_augmenter(x_s)
            x_u = spec_augmenter(x_u)

        # Predictions
        student_s_logits = student(x_s)
        student_u_logits = student(x_u)
        teacher_s_logits = teacher(n_x_s)
        teacher_u_logits = teacher(n_x_u)

        # Calculate supervised loss (only student on S)
        loss = loss_ce(student_s_logits, y_s)

        # Calculate consistency cost (mse(student(x), teacher(x))) x is S + U
        student_logits = torch.cat((student_s_logits, student_u_logits), dim=0)
        teacher_logits = torch.cat((teacher_s_logits, teacher_u_logits), dim=0)
        ccost = consistency_cost(activation_fn(student_logits), activation_fn(teacher_logits))

        total_loss = loss + lambda_cost() * ccost

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.set_grad_enabled(False):
            # Teacher prediction (for metrics purpose)
            _teacher_loss = loss_ce(teacher_s_logits, y_s)

            # Update teacher
            update_teacher_model(student, teacher, cfg.mt.alpha, iteration)

            # Compute the metrics for the student
            acc_ss = metrics.sup.acc_s(student_s_logits, y_s).mean(size=100)
            acc_su = metrics.unsup.acc_s(student_u_logits, y_u).mean(size=100)
            fscore_ss = metrics.sup.fscore_s(activation_fn(student_s_logits), y_s).mean(size=100)
            fscore_su = metrics.unsup.fscore_s(activation_fn(student_u_logits), y_u).mean(size=100)

            # Compute the metrics for the teacher
            acc_ts = metrics.sup.acc_t(teacher_s_logits, y_s).mean(size=100)
            acc_tu = metrics.unsup.acc_t(teacher_u_logits, y_u).mean(size=100)
            fscore_ts = metrics.sup.fscore_t(activation_fn(teacher_s_logits), y_s).mean(size=100)
            fscore_tu = metrics.unsup.fscore_t(activation_fn(teacher_u_logits), y_u).mean(size=100)

            # Running average of the two losses
            sce_avg = metrics.avg.sce(loss.item()).mean(size=100)
            tce_avg = metrics.avg.tce(_teacher_loss.item()).mean(size=100)
            ccost_avg = metrics.avg.ccost(ccost.item()).mean(size=100)

            # logs
            print(train_formater.format(
                1, iteration, cfg.train_param.nb_iteration,
                sce_avg, tce_avg, ccost_avg,
                acc_ss, acc_ts, fscore_ss, fscore_ts, 0.0, 0.0,
                time.time() - start_time),
                end="\r")

        tensorboard.add_scalar("train/student_acc_s", acc_ss, iteration)
        tensorboard.add_scalar("train/student_acc_u", acc_su, iteration)
        tensorboard.add_scalar("train/student_f1_s", fscore_ss, iteration)
        tensorboard.add_scalar("train/student_f1_u", fscore_su, iteration)

        tensorboard.add_scalar("train/teacher_acc_s", acc_ts, iteration)
        tensorboard.add_scalar("train/teacher_acc_u", acc_tu, iteration)
        tensorboard.add_scalar("train/teacher_f1_s", fscore_ts, iteration)
        tensorboard.add_scalar("train/teacher_f1_u", fscore_tu, iteration)

        tensorboard.add_scalar("train/student_loss", sce_avg, iteration)
        tensorboard.add_scalar("train/teacher_loss", tce_avg, iteration)
        tensorboard.add_scalar("train/consistency_cost", ccost_avg, iteration)

    def val(iteration):
        start_time = time.time()
        print("")
        nb_batch = len(val_loader)
        reset_metrics(metrics)
        student.eval()

        with torch.set_grad_enabled(False):
            for i, (X, y) in enumerate(val_loader):
                X = X.cuda().float()
                y = y.cuda().float()

                # Predictions
                student_logits = student(X)
                teacher_logits = teacher(X)

                # Calculate supervised loss (only student on S)
                loss = loss_ce(student_logits, y)
                _teacher_loss = loss_ce(teacher_logits, y)  # for metrics only
                ccost = consistency_cost(activation_fn(student_logits), activation_fn(teacher_logits))

                # Compute the metrics
                acc_s = metrics.sup.acc_s(student_logits, y).mean(size=100)
                acc_t = metrics.sup.acc_t(teacher_logits, y).mean(size=100)
                fscore_s = metrics.sup.fscore_s(activation_fn(student_logits), y).mean(size=100)
                fscore_t = metrics.sup.fscore_t(activation_fn(teacher_logits), y).mean(size=100)
                mAP_s = metrics.sup.map_s(activation_fn(student_logits).cpu().reshape(-1), y.cpu().reshape(-1)).mean(size=100)
                mAP_t = metrics.sup.map_s(activation_fn(teacher_logits).cpu().reshape(-1), y.cpu().reshape(-1)).mean(size=100)

                # Running average of the two losses
                sce_avg = metrics.avg.sce(loss.item()).mean(size=100)
                tce_avg = metrics.avg.tce(_teacher_loss.item()).mean(size=100)
                ccost_avg = metrics.avg.ccost(ccost.item()).mean(size=100)

                # logs
                print(val_formater.format(
                    iteration + 1, i, nb_batch,
                    sce_avg, tce_avg, ccost_avg, 0.0, 0.0,
                    acc_s, acc_t, fscore_s, fscore_t,
                    time.time() - start_time,
                ), end="\r")

        tensorboard.add_scalar("val/student_acc", acc_s, iteration)
        tensorboard.add_scalar("val/student_f1", fscore_s, iteration)
        tensorboard.add_scalar("val/teacher_acc", acc_t, iteration)
        tensorboard.add_scalar("val/teacher_f1", fscore_t, iteration)
        tensorboard.add_scalar("val/student_map", mAP_s, iteration)
        tensorboard.add_scalar("val/teacher_map", mAP_t, iteration)
        tensorboard.add_scalar("val/student_loss", sce_avg, iteration)
        tensorboard.add_scalar("val/teacher_loss", tce_avg, iteration)
        tensorboard.add_scalar("val/consistency_cost", ccost_avg, iteration)

        tensorboard.add_scalar("hyperparameters/learning_rate", get_lr(optimizer), iteration)
        tensorboard.add_scalar("hyperparameters/lambda_cost_max", lambda_cost(), iteration)

        tensorboard.add_scalar("max/student_acc", maximum_tracker("student_acc", acc_s), iteration)
        tensorboard.add_scalar("max/teacher_acc", maximum_tracker("teacher_acc", acc_t), iteration)
        tensorboard.add_scalar("max/student_f1", maximum_tracker("student_f1", fscore_s), iteration)
        tensorboard.add_scalar("max/teacher_f1", maximum_tracker("teacher_f1", fscore_t), iteration)
        tensorboard.add_scalar("max/student_map", maximum_tracker("student_map",mAP_s), iteration)
        tensorboard.add_scalar("max/teacher_map", maximum_tracker("teacher_map", mAP_t), iteration)

        return mAP_t

    # -------- Training loop --------
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
            mAP_t = val(i)
            print('')
            checkpoint.step(mAP_t)
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

        'ema_alpha': cfg.mt.alpha,
        'warmup_length': cfg.mt.warmup_length,
        'lamda_ccost_max': cfg.mt.lambda_ccost_max,
        'use_softmax': cfg.mt.use_softmax,
        'ccost_method': cfg.mt.ccost_method,

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
        "max_acc_student": maximum_tracker.max["student_acc"],
        "max_f1_student": maximum_tracker.max["student_f1"],
        'max_map_student': maximum_tracker.max['student_map'],
        "max_acc_teacher": maximum_tracker.max["teacher_acc"],
        "max_f1_teacher": maximum_tracker.max["teacher_f1"],
        'max_map_teacher': maximum_tracker.max['teacher_map'],
    }

    tensorboard.add_hparams(hparams, final_metrics)

    tensorboard.flush()
    tensorboard.close()


if __name__ == '__main__':
    run()
