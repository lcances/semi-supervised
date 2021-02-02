#!/usr/bin/env python
# coding: utf-8

# # import
# In[2]:


# In[3]:


import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import time

import numpy
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torchlibrosa.augmentation import SpecAugmentation

from SSL.util.model_loader import load_model
from SSL.util.loaders import load_dataset, load_optimizer, load_callbacks, load_preprocesser
from SSL.util.checkpoint import CheckPoint, mSummaryWriter
from SSL.util.mixup import MixUpBatchShuffle
from SSL.util.utils import reset_seed, get_datetime, track_maximum, DotDict

from metric_utils.metrics import BinaryAccuracy, FScore, ContinueAverage


# # Arguments

# In[1]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--from_config", default="", type=str)
parser.add_argument("-d", "--dataset_root", default="../../datasets", type=str)
parser.add_argument("-D", "--dataset", default="audioset-unbalanced", type=str)

group_t = parser.add_argument_group("Commun parameters")
group_t.add_argument("-m", "--model", default="wideresnet28_2", type=str)
group_t.add_argument("--supervised_ratio", default=1.0, type=float)
group_t.add_argument("--batch_size", default=128,type=int)
group_t.add_argument("--nb_epoch", default=15, type=int)
group_t.add_argument("--learning_rate", default=0.001, type=float)
group_t.add_argument("--resume", action="store_true", default=False)
group_t.add_argument("--seed", default=1234, type=int)

group_mixup = parser.add_argument_group("Mixup parameters")
group_mixup.add_argument("--mixup", action="store_true", default=False)
group_mixup.add_argument("--mixup_alpha", type=float, default=0.4)
group_mixup.add_argument("--mixup_max", action="store_true", default=False)
group_mixup.add_argument("--mixup_label", action="store_true", default=False)

group_sa = parser.add_argument_group("Spec augmentation")
group_sa.add_argument("--specAugment", action="store_true", default=False)
group_sa.add_argument("--sa_time_drop_width", type=int, default=32)
group_sa.add_argument('--sa_time_stripes_mum', type=int, default=2)
group_sa.add_argument("--sa_freq_drop_width", type=int, default=4)
group_sa.add_argument("--sa_freq_stripes_num", type=int, default=2)

group_u = parser.add_argument_group("Datasets parameters")
group_u.add_argument("-t", "--train_folds", nargs="+", default=[1, 2, 3, 4], type=int)
group_u.add_argument("-v", "--val_folds", nargs="+", default=[5], type=int)

group_l = parser.add_argument_group("Logs")
group_l.add_argument("--checkpoint_root", default="../../model_save/", type=str)
group_l.add_argument("--tensorboard_root", default="../../tensorboard/", type=str)
group_l.add_argument("--checkpoint_path", default="supervised", type=str)
group_l.add_argument("--tensorboard_path", default="supervised", type=str)
group_l.add_argument("--log_suffix", default="", type=str)

parser.add_argument("-N", "--nb_gpu", default=2, type=int)


args = parser.parse_args()

args.tensorboard_path = os.path.join(args.tensorboard_root, args.dataset, args.tensorboard_path)
args.checkpoint_path = os.path.join(args.checkpoint_root, args.dataset, args.checkpoint_path)


# In[2]:


vars(args)


# # initialisation

# In[8]:


reset_seed(args.seed)


# 
# trainer = SupervisedTrainer("cnn03", "esc10")
# trainer.init_trainer(
#     parameters=vars(args),
#     seed = args.seed,
#     num_workers=0,
#     pin_memory=True,
#     verbose = 2
# )

# In[9]:


# from SSL.trainers.esc import SupervisedTrainer
from SSL.trainers.trainers import Trainer

class SupervisedTrainer(Trainer):
    def __init__(self, model: str, dataset: str):
        super().__init__(model, "supervised", dataset)

trainer = SupervisedTrainer(args.model, args.dataset)


# # Prepare the dataset

# In[10]:


trainer.load_transforms()


# In[11]:


parameters = dict(
    dataset=args.dataset,

    dataset_root = args.dataset_root,
    supervised_ratio = args.supervised_ratio,
    batch_size = args.batch_size * args.nb_gpu,
    train_folds = args.train_folds,
    val_folds = args.val_folds,
    
    num_workers=10,
    pin_memory=True,

    verbose = 2,
)

trainer.load_dataset(parameters)


# # Prep model

# In[15]:


from types import MethodType
from torch.cuda import empty_cache
from torchsummary import summary


def create_model(self, nb_gpu: int = 1):
    print("Create the model")
    empty_cache()

    model_func = load_model(self.dataset, self.model_str)
    self.model = model_func(
        input_shape=self.input_shape,
        num_classes=self.num_classes,
    )
    self.model = self.model.cuda()
    
    if nb_gpu > 1:
        self.model = nn.DataParallel(self.model)

    s = summary(self.model, self.input_shape)
    
trainer.create_model = MethodType(create_model, trainer)
trainer.create_model(args.nb_gpu)


# # Training initialization

# ## Losses

# In[17]:


def init_loss(self):
    self.loss_ce = nn.BCEWithLogitsLoss(reduction="mean")

trainer.init_loss = MethodType(init_loss, trainer)


# In[18]:


trainer.init_loss()


# ## optimizer & callbacks

# In[19]:


parameters=DotDict(
    learning_rate=args.learning_rate,
)
trainer.init_optimizer(parameters)


# In[20]:


parameters=DotDict(
    nb_epoch=args.nb_epoch,
    optimizer=trainer.optimizer,
)
trainer.init_callbacks(parameters)


# # Logs and checkpoint

# In[21]:


# Logs
parameters=DotDict(
    supervised_ratio=args.supervised_ratio
)
trainer.init_logs(parameters, suffix=args.log_suffix)


# In[22]:


# Checkpoint
parameters=DotDict(
    supervised_ratio=args.supervised_ratio
)
trainer.init_checkpoint(parameters, suffix=args.log_suffix)


# ## Metrics

# In[23]:


# Metrics
def init_metrics(self):
    self.metrics = DotDict(
        fscore_fn=FScore(),
        acc_fn=BinaryAccuracy(),
        avg_fn=ContinueAverage(),
    )
    self.maximum_tracker = track_maximum()

trainer.init_metrics = MethodType(init_metrics, trainer)
trainer.init_metrics()


# ## training function

# In[24]:


def set_printing_form(self):
    UNDERLINE_SEQ = "\033[1;4m"
    RESET_SEQ = "\033[0m"

    header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<8.8} {:<6.6} - {:<9.9} {:<12.12}| {:<9.9}- {:<6.6}"
    value_form  = "{:<8.8} {:<6} - {:<6} - {:<8.8} {:<6.4f} - {:<9.9} {:<10.4f}| {:<9.4f}- {:<6.4f}"

    self.header = header_form.format(
        ".               ", "Epoch", "%", "Losses:", "ce", "metrics: ", "acc", "F1 ","Time"
    )

    self.train_form = value_form
    self.val_form = UNDERLINE_SEQ + value_form + RESET_SEQ


# # init mixup and SpecAugment

# In[25]:


# Spec augmenter
spec_augmenter = SpecAugmentation(time_drop_width=args.sa_time_drop_width,
                                  time_stripes_num=args.sa_time_stripes_mum,
                                  freq_drop_width=args.sa_freq_drop_width,
                                  freq_stripes_num=args.sa_freq_stripes_num)

# Mixup
mixup_fn = MixUpBatchShuffle(alpha=args.mixup_alpha, apply_max=args.mixup_max, mix_labels=args.mixup_label)



batch_summed = []


def train_fn(self, epoch: int):
    # aliases
    M = self.metrics
    T = self.tensorboard.add_scalar
    nb_batch = len(self.train_loader)

    start_time = time.time()
    print("")

    self.reset_metrics()
    self.model.train()

    for i, (X, y) in enumerate(self.train_loader):
        
        if args.specAugment:
            X = X.view(-1, 1, *X.shape[1:])
            X = spec_augmenter(X)
            X = X.squeeze(1)

        if args.mixup:
            X, y = mixup_fn(X, y)
        
        X = X.cuda().float()
        y = y.cuda().float()
        
        logits = self.model(X)
        loss = self.loss_ce(logits, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.set_grad_enabled(False):
            acc = M.acc_fn(logits, y).mean
            fscore = M.fscore_fn(logits, y).mean
            avg_ce = M.avg_fn(loss.item()).mean
            
            summed = torch.sum(y, axis=0)
            batch_summed.append(summed)

            # logs
            print(self.train_form.format(
                "Training: ",
                epoch + 1,
                int(100 * (i + 1) / nb_batch),
                "", avg_ce,
                "", acc, fscore,
                time.time() - start_time
            ), end="\r")

    T("train/Lce", avg_ce, epoch)
    T("train/f1", fscore, epoch)
    T("train/acc", acc, epoch)


# In[27]:


def val_fn(self, epoch: int):
    # aliases
    M = self.metrics
    T = self.tensorboard.add_scalar
    nb_batch = len(self.val_loader)

    start_time = time.time()
    print("")

    self.reset_metrics()
    self.model.eval()

    with torch.set_grad_enabled(False):
        for i, (X, y) in enumerate(self.val_loader):
            X = X.cuda().float()
            y = y.cuda().float()

            logits = self.model(X)
            loss = self.loss_ce(logits, y)

            acc = M.acc_fn(logits, y).mean
            fscore = M.fscore_fn(logits, y).mean
            avg_ce = M.avg_fn(loss.item()).mean

            # logs
            print(self.val_form.format(
                "Validation: ",
                epoch + 1,
                int(100 * (i + 1) / nb_batch),
                "", avg_ce,
                "", acc, fscore,
                time.time() - start_time
            ), end="\r")

    T("val/Lce", avg_ce, epoch)
    T("val/f1", fscore, epoch)
    T("val/acc", acc, epoch)

    T("hyperparameters/learning_rate", self._get_lr(), epoch)

    T("max/acc", self.maximum_tracker("acc", acc), epoch)
    T("max/f1", self.maximum_tracker("f1", fscore), epoch)

    self.checkpoint.step(acc)
    for c in self.callbacks:
        c.step()
        pass


# In[28]:


def test_fn(self):
    # aliases
    M = self.metrics
    T = self.tensorboard.add_scalar
    nb_batch = len(self.val_loader)

    # Load best epoch
    self.checkpoint.load_best()

    start_time = time.time()
    print("")

    self.reset_metrics()
    self.model.eval()

    with torch.set_grad_enabled(False):
        for i, (X, y) in enumerate(self.test_loader):
            X = X.cuda()
            y = y.cuda()

            logits = self.model(X)
            loss = self.loss_ce(logits, y)

            acc = M.acc_fn(pred_arg, y).mean
            fscore = M.fscore_fn(pred, y).mean
            avg_ce = M.avg_fn(loss.item()).mean

            # logs
            print(self.val_form.format(
                "Testing: ",
                1,
                int(100 * (i + 1) / nb_batch),
                "", avg_ce,
                "", acc, fscore,
                time.time() - start_time
            ), end="\r")


# In[29]:


trainer.set_printing_form = MethodType(set_printing_form, trainer)
trainer.train_fn = MethodType(train_fn, trainer)
trainer.val_fn = MethodType(val_fn, trainer)
trainer.test_fn = MethodType(test_fn, trainer)


# # Training

# In[30]:


# Resume if wish
if args.resume:
    trainer.checkpoint.load_last()


# In[31]:


# Fit function
trainer.set_printing_form()
print(trainer.header)

start_epoch = trainer.checkpoint.epoch_counter
end_epoch = args.nb_epoch

for e in range(start_epoch, args.nb_epoch):
    trainer.train_fn(e)
    trainer.val_fn(e)
    
    trainer.tensorboard.flush()


# # Evaluate
total_pred = []
total_targets = []

trainer.model.eval()

nb_batch = len(trainer.val_loader)

S = nn.Sigmoid()

with torch.set_grad_enabled(False):
    for i, (X, y) in enumerate(trainer.val_loader):
        X = X.cuda().float()
        y = y.cuda().float()

        logits = trainer.model(X)
        
        total_pred.append(S(logits).cpu())
        total_targets.append(y.cpu())
        
        print("%d / %d" % (i, nb_batch), end="\r")

total_pred_ = numpy.vstack(total_pred)
total_targets_ = numpy.vstack(total_targets)

# # Compute mAP
mAP = metrics.average_precision_score(total_targets_, total_pred_, average=None)

# # Compute mAUC
from sklearn import metrics
import tqdm

metrics_auc = []
for i in tqdm.tqdm(range(527)):
    y = total_targets_[:,i]
    pred = total_pred_[:,i]

    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    metrics_auc.append(metrics.auc(fpr, tpr))

mAUC = metrics_auc

# # Computing d-prime
# https://stats.stackexchange.com/questions/492673/understanding-and-implementing-the-dprime-measure-in-python

from scipy.stats import norm
Z = norm.ppf

def calc_dprime(y_true, y_pred):
    return numpy.sqrt(2) * Z(metrics.roc_auc_score(y_true,y_pred))

dprimes = []
for i in tqdm.tqdm(range(527)):
    y = total_targets[:,i]
    pred = total_pred[:,i]

    dprimes.append(calc_dprime(y, pred))

dprimes = numpy.mean(dprimes)

# Save all the parameters
tensorboard_params = {}
for key, value in args.__dict__.items():
    tensorboard_params[key] = str(value)
trainer.tensorboard.add_hparams(tensorboard_params, {})

##########
hparams = {}
for key, value in args.__dict__.items():
    hparams[key] = str(value)

final_metrics = {
    "mAP": mAP.mean(),
    "mAUC": numpy.mean(metrics_auc),
    "mDPrime": numpy.mean(dprimes),
}
tensorboard.add_hparams(hparams, final_metrics)
tensorboard.flush()
tensorboard.close()
