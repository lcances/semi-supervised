import sys
sys.executable


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

from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from SSL.util.model_loader import load_model
from SSL.util.loaders import load_dataset, load_optimizer, load_callbacks, load_preprocesser
from SSL.util.checkpoint import CheckPoint, mSummaryWriter
from SSL.util.utils import reset_seed, get_datetime, track_maximum, DotDict
from SSL.util.mixup import MixUpBatchShuffle

from metric_utils.metrics import BinaryAccuracy, FScore, ContinueAverage


# # Arguments

# In[5]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--from_config", default="", type=str)
parser.add_argument("-d", "--dataset_root", default="../../datasets", type=str)
parser.add_argument("-D", "--dataset", default="audioset-unbalanced", type=str)

group_t = parser.add_argument_group("Commun parameters")
group_t.add_argument("-m", "--model", default="wideresnet28_2", type=str)
group_t.add_argument("--supervised_ratio", default=1.0, type=float)
group_t.add_argument("--batch_size", default=128, type=int)
group_t.add_argument("--nb_epoch", default=500_000, type=int) # nb iteration
group_t.add_argument("--learning_rate", default=0.003, type=float)
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
group_sa.add_argument('--sa_time_stripes_num', type=int, default=1)
group_sa.add_argument("--sa_freq_drop_width", type=int, default=4)
group_sa.add_argument("--sa_freq_stripes_num", type=int, default=1)

group_u = parser.add_argument_group("Datasets parameters")
group_u.add_argument("-t", "--train_folds", nargs="+", default=[1, 2, 3, 4], type=int)
group_u.add_argument("-v", "--val_folds", nargs="+", default=[5], type=int)

group_l = parser.add_argument_group("Logs")
group_l.add_argument("--checkpoint_root", default="../../model_save/", type=str)
group_l.add_argument("--tensorboard_root", default="../../tensorboard/", type=str)
group_l.add_argument("--checkpoint_path", default="supervised", type=str)
group_l.add_argument("--tensorboard_path", default="supervised", type=str)
group_l.add_argument("--log_suffix", default="", type=str)

parser.add_argument("-N", "--nb_gpu", default=1, type=int)
parser.add_argument('-c', '--nb_cpu', default=5, type=int)

args = parser.parse_args()

args.tensorboard_path = os.path.join(args.tensorboard_root, args.dataset, args.tensorboard_path)
args.checkpoint_path = os.path.join(args.checkpoint_root, args.dataset, args.checkpoint_path)


# In[6]:


vars(args)


# # initialisation

# In[7]:


reset_seed(args.seed)
# In[8]:


# from SSL.trainers.esc import SupervisedTrainer
from SSL.trainers.trainers import Trainer

class SupervisedTrainer(Trainer):
    
    def __init__(self, model: str, dataset: str):
        super().__init__(model, "supervised", dataset)

trainer = SupervisedTrainer(args.model, args.dataset)


# # Prepare the dataset

# In[9]:


trainer.load_transforms()


# In[10]:


parameters = dict(
    dataset=args.dataset,

    dataset_root = args.dataset_root,
    supervised_ratio = args.supervised_ratio,
    batch_size = args.batch_size,
    train_folds = args.train_folds,
    val_folds = args.val_folds,
    
    num_workers=args.nb_cpu,
    pin_memory=True,

    verbose = 2,
)

trainer.load_dataset(parameters)


# # Prep model

# In[11]:


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
    
    s = summary(self.model, self.input_shape)
    
    if nb_gpu > 1:
        self.model = nn.DataParallel(self.model)
    
trainer.create_model = MethodType(create_model, trainer)
trainer.create_model(args.nb_gpu)


# # Training initialization

# ## Losses

# In[12]:


def init_loss(self):
    self.loss_ce = nn.BCEWithLogitsLoss(reduction="mean")

trainer.init_loss = MethodType(init_loss, trainer)


# In[13]:


trainer.init_loss()


# ## optimizer & callbacks

# In[14]:


parameters=DotDict(
    learning_rate=args.learning_rate,
)
trainer.init_optimizer(parameters)


# In[15]:


parameters=DotDict(
    nb_epoch=args.nb_epoch,
    optimizer=trainer.optimizer,
)
trainer.init_callbacks(parameters)


# # Logs and checkpoint

# In[16]:


# Prepare suffix
# normale training parameters
sufix_title = ''
sufix_title += f'_{args.learning_rate}-lr'
sufix_title += f'_{args.supervised_ratio}-sr'
sufix_title += f'_{args.nb_epoch}-e'
sufix_title += f'_{args.batch_size}-bs'
sufix_title += f'_{args.seed}-seed'

# mixup parameters
if args.mixup:
    sufix_title += '_mixup'
    if args.mixup_max: sufix_title += "-max"
    if args.mixup_label: sufix_title += "-label"
    sufix_title += f"-{args.mixup_alpha}-a"
    
# SpecAugment parameters
if args.specAugment:
    sufix_title += '_specAugment'
    sufix_title += f'-{args.sa_time_drop_width}tdw'
    sufix_title += f'-{args.sa_time_stripes_num}tsn'
    sufix_title += f'-{args.sa_freq_drop_width}fdw'
    sufix_title += f'-{args.sa_freq_stripes_num}fsn'


# In[17]:


sufix_title


# In[18]:


# Logs
parameters=DotDict(
    supervised_ratio=args.supervised_ratio
)

trainer.init_logs(parameters, suffix=sufix_title)


# In[19]:


# Checkpoint
parameters=DotDict(
    supervised_ratio=args.supervised_ratio
)
trainer.init_checkpoint(parameters, suffix=sufix_title)


# ## Metrics

# In[20]:


# Metrics
from metric_utils.metrics import Metrics
from sklearn import metrics


class MAP(Metrics):
    def __init__(self, epsilon=1e-10):
        super().__init__(epsilon)

    def __call__(self, y_pred, y_true):
        super().__call__(y_pred, y_true)
        aps = metrics.average_precision_score(y_true, y_pred, average=None)
        aps = numpy.nan_to_num(aps)
        
        self.values.append(aps.mean())
        return self

def init_metrics(self):
    self.metrics = DotDict(
        fscore_fn=FScore(),
        acc_fn=BinaryAccuracy(),
        avg_fn=ContinueAverage(),
        mAP_fn=MAP()
    )
    self.time_average_fn = ContinueAverage()
    
    self.maximum_tracker = track_maximum()

trainer.init_metrics = MethodType(init_metrics, trainer)
trainer.init_metrics()


# ## training function

# In[21]:


def set_printing_form(self):
    UNDERLINE_SEQ = "\033[1;4m"
    RESET_SEQ = "\033[0m"

    header_form = "Type            Epoch -       /       - Losses: bce       - Metrics: acc         F1           mAP           - Remaining time "
    header_form = "{:<16.16} {:<5.5} - {:<5.5} / {:<5.5} - {:<7.7} {:<9.9} - {:<8.8} {:<12.12} {:<12.12} {:<12.12} - {:<6.6}"
    value_form  = "{:<16.16} {:<5} - {:>5} / {:<5} - {:7.7} {:<9.4f} - {:<8.8} {:<12.3e} {:<12.3e} {:<12.3e} - {:<6.4f}"

    self.header = header_form.format(
        ".               ", "Epoch", "", "", "Losses:", "ce", "metrics: ", "acc", "F1", "mAP", "Time"
    )

    self.train_form = value_form
    self.val_form = UNDERLINE_SEQ + value_form + RESET_SEQ
    
trainer.set_printing_form = MethodType(set_printing_form, trainer)


# # init mixup and SpecAugment

# In[22]:


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes. 
        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [1, 2]    # dim 2: frequency; dim 3: time

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 3

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(input[n], total_width)

            return input


    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 1:
                e[bgn : bgn + distance, :] = 0
            elif self.dim == 2:
                e[:, bgn : bgn + distance] = 0


class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width, 
        freq_stripes_num):
        """Spec augmetation. 
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D. 
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method 
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(SpecAugmentation, self).__init__()

        self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width, 
            stripes_num=time_stripes_num)

        self.freq_dropper = DropStripes(dim=1, drop_width=freq_drop_width, 
            stripes_num=freq_stripes_num)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x


# In[23]:


# # Spec augmenter
spec_augmenter = SpecAugmentation(time_drop_width=args.sa_time_drop_width,
                                  time_stripes_num=args.sa_time_stripes_num,
                                  freq_drop_width=args.sa_freq_drop_width,
                                  freq_stripes_num=args.sa_freq_stripes_num)

# Mixup
mixup_fn = MixUpBatchShuffle(alpha=args.mixup_alpha, apply_max=args.mixup_max, mix_labels=args.mixup_label)


# In[24]:


batch_summed = []


# def calc_class_dist(y):
#     with torch.set_grad_enabled(False):
#         summed = torch.sum(y, axis=0)
#         summed = summed[summed > 0]
#         if len(summed) <= 0:
#             return False

#         ratio = min(summed) / max(summed)
#         if ratio < 0.16:
#             return False
        
#         return True

def train_fn(self, epoch, X, y, start_time) -> Union[float, float]:
    # aliases
    M = self.metrics
    T = self.tensorboard.add_scalar

    self.model.train()

    y_ = y.detach().clone() # keep a copy outside the graph and in cpu to compute the mAP

    X = X.cuda().float()
    y = y.cuda().float()

    if args.mixup:
        X, y = mixup_fn(X, y)

    if args.specAugment:
        X = spec_augmenter(X)

    logits = self.model(X)
    loss = self.loss_ce(logits, y)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    with torch.set_grad_enabled(False):

        pred = torch.sigmoid(logits)

        fscore = M.fscore_fn(pred, y)
        acc = M.acc_fn(pred, y)
        avg_ce = M.avg_fn(loss.item())

        end_time = time.time()
        running_mean_time = self.time_average_fn(end_time - start_time)

        # logs
        print(self.train_form.format(
            "Training: ",
            epoch + 1,
            e, args.nb_epoch,
            "", avg_ce.mean(size=1000),
            "", acc.mean(size=1000), fscore.mean(size=1000), 0.0,
            time.time() - start_time,
        ), end="\r")

    T("train/Lce", avg_ce.mean(size=1000), epoch)
    T("train/f1", fscore.mean(size=1000), epoch)
    T("train/acc", acc.mean(size=1000), epoch)
    
    return avg_ce.value, fscore.value

trainer.train_fn = MethodType(train_fn, trainer)


# In[25]:


def val_fn(self, epoch: int)  -> Union[float, float]:
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

            pred = torch.sigmoid(logits)
            fscore = M.fscore_fn(pred, y)            
            acc = M.acc_fn(pred, y)
            mAP = M.mAP_fn(pred.cpu().reshape(-1), y.cpu().reshape(-1))
            avg_ce = M.avg_fn(loss.item())

            # logs
            print(self.val_form.format(
                "Validation: ",
                epoch + 1,
                i, nb_batch,
                "", avg_ce.mean(size=1000),
                "", acc.mean(size=1000), fscore.mean(size=1000), mAP.mean(size=1000),
                time.time() - start_time
            ), end="\r")

    T("val/Lce", avg_ce.mean(size=1000), epoch)
    T("val/f1", fscore.mean(size=1000), epoch)
    T("val/acc", acc.mean(size=1000), epoch)
    T("val/mAP", mAP.mean(size=1000), epoch)

    T("hyperparameters/learning_rate", self._get_lr(), epoch)

    T("max/acc", self.maximum_tracker("acc", acc.mean(size=1000)), epoch)
    T("max/f1", self.maximum_tracker("f1", fscore.mean(size=1000)), epoch)
    T('max/mAP', self.maximum_tracker('mAP', mAP.mean(size=1000)), epoch)
    
    return avg_ce, fscore
    
trainer.val_fn = MethodType(val_fn, trainer)


# In[26]:


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

            pred = torch.sigmoid(logits)
            y_one_hot = y # F.one_hot(y, num_classes=self.num_classes)
            fscore = M.fscore_fn(pred, y_one_hot).mean
            
            acc = M.acc_fn(logits, y).mean
            
            avg_ce = M.avg_fn(loss.item()).mean

            # logs
            print(self.val_form.format(
                "Testing: ",
                1,
                i, nb_batch,
                "", avg_ce,
                "", acc, fscore,
                time.time() - start_time
            ), end="\r")
            
    return avg_ce, fscore
            
trainer.test_fn = MethodType(test_fn, trainer)


# # Training

# In[27]:


# Resume if wish
if args.resume:
    trainer.checkpoint.load_last()


# In[ ]:


# Fit function
trainer.set_printing_form()
print(trainer.header)

start_epoch = trainer.checkpoint.epoch_counter
end_epoch = args.nb_epoch

train_iterator = iter(trainer.train_loader)
start_time = time.time()

for e in range(start_epoch, args.nb_epoch):
    # Perform train 
    train_avg_ce, train_fscore = trainer.train_fn(e, *train_iterator.next(), start_time)
    
    # Validation every 10 000 iteration
    if e % 10_000 == 0 and e != 0:
        val_avg_ce, val_fscore = trainer.val_fn(e)
        print('')
        trainer.checkpoint.step(val_fscore.value)
    
    # Apply the different callbacks
#     for c in trainer.callbacks:
#         c.step()
    
    if e % 1000 == 0:
        trainer.tensorboard.flush()
    
trainer.save_hparams(vars(args))
