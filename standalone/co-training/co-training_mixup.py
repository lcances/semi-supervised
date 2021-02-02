import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.parallel import DataParallel
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
# from torch.utils.tensorboard import SummaryWriter
from advertorch.attacks import GradientSignAttack


# In[3]:


from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage, Ratio
from SSL.util.checkpoint import CheckPoint, mSummaryWriter
from SSL.util.utils import reset_seed, get_datetime, ZipCycle, track_maximum
from SSL.util.model_loader import load_model
from SSL.util.loaders import load_dataset, load_optimizer, load_callbacks, load_preprocesser
from SSL.util.mixup import MixUpBatchShuffle, MixUp

from SSL.ramps import Warmup, sigmoid_rampup
from SSL.losses import loss_cot, loss_diff, loss_sup


# In[27]:


os.path.abspath(".")


# # Arguments

# In[4]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--from_config", default="", type=str)
parser.add_argument("-d", "--dataset_root", default="../../datasets/", type=str)
parser.add_argument("-D", "--dataset", default="esc10", type=str)
parser.add_argument("-N", "--nb_gpu", default=1, type=int)

group_t = parser.add_argument_group("Commun parameters")
group_t.add_argument("--model", default="wideresnet28_2", type=str)
group_t.add_argument("--supervised_ratio", default=0.1, type=float)
group_t.add_argument("--batch_size", default=64, type=int)
group_t.add_argument("--nb_epoch", default=300, type=int)
group_t.add_argument("--learning_rate", default=5e-4, type=float)
group_t.add_argument("--resume", action="store_true", default=False)
group_t.add_argument("--seed", default=1234, type=int)

group_m = parser.add_argument_group("Model parameters")
group_m.add_argument("--num_classes", default=10, type=int)

group_u = parser.add_argument_group("ESC and UBS8K parameters")
group_u.add_argument("-t", "--train_folds", nargs="+", default=[1, 2, 3, 4,], type=int)
group_u.add_argument("-v", "--val_folds", nargs="+", default=[5], type=int)

group_h = parser.add_argument_group('hyperparameters')
group_h.add_argument("--lambda_cot_max", default=1, type=float)
group_h.add_argument("--lambda_diff_max", default=0.5, type=float)
group_h.add_argument("--warmup_length", default=160, type=int)
group_h.add_argument("--epsilon", default=0.02, type=float)

group_mixup = parser.add_argument_group("Mixup parameters")
group_mixup.add_argument("--mixup", action="store_true", default=False)
group_mixup.add_argument("--mixup_alpha", type=float, default=0.4)
group_mixup.add_argument("--mixup_max", action="store_true", default=False)
group_mixup.add_argument("--mixup_label", action="store_true", default=False)

group_l = parser.add_argument_group("Logs")
group_l.add_argument("--checkpoint_root", default="../../model_save/", type=str)
group_l.add_argument("--tensorboard_root", default="../../tensorboard/", type=str)
group_l.add_argument("--checkpoint_path", default="deep-co-training_mixup", type=str)
group_l.add_argument("--tensorboard_path", default="deep-co-training_mixup", type=str)
group_l.add_argument("--tensorboard_sufix", default="", type=str)

args = parser.parse_args()

tensorboard_path = os.path.join(args.tensorboard_root, args.dataset, args.tensorboard_path)
checkpoint_path = os.path.join(args.checkpoint_root, args.dataset, args.checkpoint_path)


# # Initialization

# In[5]:


reset_seed(args.seed)


# # Prepare the dataset

# In[6]:


train_transform, val_transform = load_preprocesser(args.dataset, "dct")


# In[7]:


train_transform


# In[8]:


manager, train_loader, val_loader = load_dataset(
    args.dataset,
    "dct",
    
    dataset_root = args.dataset_root,
    supervised_ratio = args.supervised_ratio,
    batch_size = args.batch_size,
    train_folds = args.train_folds,
    val_folds = args.val_folds,

    train_transform=train_transform,
    val_transform=val_transform,
    
    num_workers=2,
    pin_memory=True,

    verbose = 2
)


# In[9]:


input_shape = train_loader._iterables[0].dataset[0][0].shape
input_shape


# # Models

# In[10]:


torch.cuda.empty_cache()
model_func = load_model(args.dataset, args.model)

commun_args = dict(
    manager=manager,
    num_classes=args.num_classes,
    input_shape=list(input_shape),
)

m1, m2 = model_func(**commun_args), model_func(**commun_args)

if args.nb_gpu > 1:
    m1 = DataParallel(m1)
    m2 = DataParallel(m2)

m1 = m1.cuda()
m2 = m2.cuda()


# In[11]:


from torchsummary import summary

s = summary(m1, tuple(input_shape))


# # training parameters

# In[12]:


tensorboard_root=f"{args.model}/{args.supervised_ratio}/{get_datetime()}_{model_func.__name__}"
checkpoint_root = f"{args.model}/{args.supervised_ratio}/{model_func.__name__}"

# dct parameters
sufix_title = f'_{args.lambda_cot_max}-lcm'
sufix_title = f'_{args.lambda_diff_max}-ldm'
sufix_title = f'_{args.warmup_length}-wl'
sufix_title = f'_{args.epsilon}-eps'

# mixup parameters
if args.mixup:
    sufix_title += "_mixup"
    if args.mixup_max: sufix_title += "-max"
    if args.mixup_label: sufix_title += "-label"
    sufix_title += f"-{args.mixup_alpha}-a"
    
# normale training parameters
sufix_title += f'_{args.learning_rate}-lr'
sufix_title += f'_{args.supervised_ratio}-sr'
sufix_title += f'_{args.nb_epoch}-e'
sufix_title += f'_{args.batch_size}-bs'
sufix_title += f'_{args.seed}-seed'

tensorboard_title = tensorboard_root + sufix_title
checkpoint_title = checkpoint_root + sufix_title


# In[13]:


tensorboard = mSummaryWriter(log_dir="%s/%s" % (tensorboard_path, tensorboard_title), comment=model_func.__name__)
print(os.path.join(tensorboard_path, tensorboard_title))


# ## Optimizer & callbacks

# In[14]:


optim_args = dict(
    learning_rate=args.learning_rate,
)

optimizer = load_optimizer(args.dataset, "dct", model1=m1, model2=m2, **optim_args)
callbacks = load_callbacks(args.dataset, "dct", optimizer=optimizer, nb_epoch=args.nb_epoch)


# ## Adversarial generator

# In[15]:


# adversarial generation
adv_generator_1 = GradientSignAttack(
    m1, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    eps=args.epsilon, clip_min=-np.inf, clip_max=np.inf, targeted=False
)

adv_generator_2 = GradientSignAttack(
    m2, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    eps=args.epsilon, clip_min=-np.inf, clip_max=np.inf, targeted=False
)


# In[16]:


# Losses
# see losses.py

# define the warmups & add them to the callbacks (for update)
lambda_cot = Warmup(args.lambda_cot_max, args.warmup_length, sigmoid_rampup)
lambda_diff = Warmup(args.lambda_diff_max, args.warmup_length, sigmoid_rampup)
callbacks += [lambda_cot, lambda_diff]

# checkpoints
checkpoint = CheckPoint([m1, m2], optimizer, mode="max", name="%s/%s_m1.torch" % (checkpoint_path, checkpoint_title))

# metrics
metrics_fn = dict(
    ratio_s=[Ratio(), Ratio()],
    ratio_u=[Ratio(), Ratio()],
    acc_s=[CategoricalAccuracy(), CategoricalAccuracy()],
    acc_u=[CategoricalAccuracy(), CategoricalAccuracy()],
    f1_s=[FScore(), FScore()],
    f1_u=[FScore(), FScore()],
    
    avg_total=ContinueAverage(),
    avg_sup=ContinueAverage(),
    avg_cot=ContinueAverage(),
    avg_diff=ContinueAverage(),
)
maximum_tracker = track_maximum()


def reset_metrics():
    for item in metrics_fn.values():
        if isinstance(item, list):
            for f in item:
                f.reset()
        else:
            item.reset()
            


# In[17]:


reset_metrics()


# ## Metrics and hyperparameters

# In[18]:


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# # Training functions

# In[19]:


UNDERLINE_SEQ = "\033[1;4m"
RESET_SEQ = "\033[0m"

header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<8.8} {:<6.6} | {:<6.6} | {:<6.6} | {:<6.6} - {:<9.9} {:<9.9} | {:<9.9}- {:<6.6}"
value_form  = "{:<8.8} {:<6} - {:<6} - {:<8.8} {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f} - {:<9.9} {:<9.4f} | {:<9.4f}- {:<6.4f}"

header = header_form.format(
    "", "Epoch", "%", "Losses:", "Lsup", "Lcot", "Ldiff", "total", "metrics: ", "acc_s1", "acc_u1","Time"
)

train_form = value_form
val_form = UNDERLINE_SEQ + value_form + RESET_SEQ

print(header)


# In[20]:


mixup_s_fn = MixUp(alpha=args.mixup_alpha, apply_max=args.mixup_max, mix_labels=args.mixup_label)
mixup_u_fn = MixUpBatchShuffle(alpha=args.mixup_alpha, apply_max=args.mixup_max, mix_labels=args.mixup_label)


# In[21]:


def train(epoch):
    start_time = time.time()
    print("")

    reset_metrics()
    m1.train()
    m2.train()

    for batch, (S1, S2, U) in enumerate(train_loader):
        x_s1, y_s1 = S1
        x_s2, y_s2 = S2
        x_u, y_u = U
        
        # Apply mixup if needed, otherwise no mixup.
        if args.mixup:
#             x_s1, y_s1 = mixup_s_fn(x_s1, x_s2, y_s1, y_s2)
#             x_s2, y_s2 = mixup_s_fn(x_s2, x_s1, y_s2, y_s1)
            x_u, y_u = mixup_u_fn(x_u, y_u)

        x_s1, x_s2, x_u = x_s1.cuda(), x_s2.cuda(), x_u.cuda()
        y_s1, y_s2, y_u = y_s1.cuda(), y_s2.cuda(), y_u.cuda()

        with autocast():
            logits_s1 = m1(x_s1)
            logits_s2 = m2(x_s2)
            logits_u1 = m1(x_u)
            logits_u2 = m2(x_u)

        # pseudo labels of U
        pred_u1 = torch.argmax(logits_u1, 1)
        pred_u2 = torch.argmax(logits_u2, 1)

        # ======== Generate adversarial examples ========
        # fix batchnorm ----
        m1.eval()
        m2.eval()

        #generate adversarial examples ----
        adv_data_s1 = adv_generator_1.perturb(x_s1, y_s1)
        adv_data_u1 = adv_generator_1.perturb(x_u, pred_u1)

        adv_data_s2 = adv_generator_2.perturb(x_s2, y_s2)
        adv_data_u2 = adv_generator_2.perturb(x_u, pred_u2)

        m1.train()
        m2.train()

        # predict adversarial examples ----
        with autocast():
            adv_logits_s1 = m1(adv_data_s2)
            adv_logits_s2 = m2(adv_data_s1)

            adv_logits_u1 = m1(adv_data_u2)
            adv_logits_u2 = m2(adv_data_u1)

        # ======== calculate the differents loss ========
        # zero the parameter gradients ----
        for p in m1.parameters(): p.grad = None # zero grad
        for p in m2.parameters(): p.grad = None

        # losses ----
        with autocast():
            l_sup = loss_sup(logits_s1, logits_s2, y_s1, y_s2)

            l_cot = loss_cot(logits_u1, logits_u2)

            l_diff = loss_diff(
                logits_s1, logits_s2, adv_logits_s1, adv_logits_s2,
                logits_u1, logits_u2, adv_logits_u1, adv_logits_u2
            )

            total_loss = l_sup + lambda_cot() * l_cot + lambda_diff() * l_diff
        total_loss.backward()
        optimizer.step()

        # ======== Calc the metrics ========
        with torch.set_grad_enabled(False):
            # accuracies ----
            pred_s1 = torch.argmax(logits_s1, dim=1)
            pred_s2 = torch.argmax(logits_s2, dim=1)

            acc_s1 = metrics_fn["acc_s"][0](pred_s1, y_s1)
            acc_s2 = metrics_fn["acc_s"][1](pred_s2, y_s2)
            acc_u1 = metrics_fn["acc_u"][0](pred_u1, y_u)
            acc_u2 = metrics_fn["acc_u"][1](pred_u2, y_u)

            # ratios  ----
            adv_pred_s1 = torch.argmax(adv_logits_s1, 1)
            adv_pred_s2 = torch.argmax(adv_logits_s2, 1)
            adv_pred_u1 = torch.argmax(adv_logits_u1, 1)
            adv_pred_u2 = torch.argmax(adv_logits_u2, 1)

            ratio_s1 = metrics_fn["ratio_s"][0](adv_pred_s1, y_s1)
            ratio_s2 = metrics_fn["ratio_s"][1](adv_pred_s2, y_s2)
            ratio_u1 = metrics_fn["ratio_u"][0](adv_pred_u1, y_u)
            ratio_u2 = metrics_fn["ratio_u"][1](adv_pred_u2, y_u)
            # ========

            avg_total = metrics_fn["avg_total"](total_loss.item())
            avg_sup = metrics_fn["avg_sup"](l_sup.item())
            avg_diff = metrics_fn["avg_diff"](l_diff.item())
            avg_cot = metrics_fn["avg_cot"](l_cot.item())

            # logs
            print(train_form.format(
                "Training: ",
                epoch + 1,
                int(100 * (batch + 1) / len(train_loader)),
                "", avg_sup.mean, avg_cot.mean, avg_diff.mean, avg_total.mean,
                "", acc_s1.mean, acc_u1.mean,
                time.time() - start_time
            ), end="\r")


    # using tensorboard to monitor loss and acc\n",
    tensorboard.add_scalar('train/total_loss', avg_total.mean, epoch)
    tensorboard.add_scalar('train/Lsup', avg_sup.mean, epoch )
    tensorboard.add_scalar('train/Lcot', avg_cot.mean, epoch )
    tensorboard.add_scalar('train/Ldiff', avg_diff.mean, epoch )
    tensorboard.add_scalar("train/acc_1", acc_s1.mean, epoch )
    tensorboard.add_scalar("train/acc_2", acc_s2.mean, epoch )

    tensorboard.add_scalar("detail_acc/acc_s1", acc_s1.mean, epoch)
    tensorboard.add_scalar("detail_acc/acc_s2", acc_s2.mean, epoch)
    tensorboard.add_scalar("detail_acc/acc_u1", acc_u1.mean, epoch)
    tensorboard.add_scalar("detail_acc/acc_u2", acc_u2.mean, epoch)

    tensorboard.add_scalar("detail_ratio/ratio_s1", ratio_s1.mean, epoch)
    tensorboard.add_scalar("detail_ratio/ratio_s2", ratio_s2.mean, epoch)
    tensorboard.add_scalar("detail_ratio/ratio_u1", ratio_u1.mean, epoch)
    tensorboard.add_scalar("detail_ratio/ratio_u2", ratio_u2.mean, epoch)

    # Return the total loss to check for NaN
    return total_loss.item()


# In[22]:


def test(epoch, msg = ""):
    start_time = time.time()
    print("")

    reset_metrics()
    m1.eval()
    m2.eval()

    with torch.set_grad_enabled(False):
        for batch, (X, y) in enumerate(val_loader):
            x = X.cuda()
            y = y.cuda()

            with autocast():
                logits_1 = m1(x)
                logits_2 = m2(x)

                # losses ----
                l_sup = loss_sup(logits_1, logits_2, y, y)

            # ======== Calc the metrics ========
            # accuracies ----
            pred_1 = torch.argmax(logits_1, dim=1)
            pred_2 = torch.argmax(logits_2, dim=1)

            acc_1 = metrics_fn["acc_s"][0](pred_1, y)
            acc_2 = metrics_fn["acc_s"][1](pred_2, y)

            avg_sup = metrics_fn["avg_sup"](l_sup.item())

            # logs
            print(val_form.format(
                "Validation: ",
                epoch + 1,
                int(100 * (batch + 1) / len(train_loader)),
                "", avg_sup.mean, 0.0, 0.0, avg_sup.mean,
                "", acc_1.mean, 0.0,
                time.time() - start_time
            ), end="\r")

    tensorboard.add_scalar("val/acc_1", acc_1.mean, epoch)
    tensorboard.add_scalar("val/acc_2", acc_2.mean, epoch)
        
    tensorboard.add_scalar("max/acc_1", maximum_tracker("acc_1", acc_1.mean), epoch )
    tensorboard.add_scalar("max/acc_2", maximum_tracker("acc_2", acc_2.mean), epoch )
    
    tensorboard.add_scalar("detail_hyperparameters/lambda_cot", lambda_cot(), epoch)
    tensorboard.add_scalar("detail_hyperparameters/lambda_diff", lambda_diff(), epoch)
    tensorboard.add_scalar("detail_hyperparameters/learning_rate", get_lr(optimizer), epoch)

    # Apply callbacks
    for c in callbacks:
        c.step()

    # call checkpoint
    checkpoint.step(acc_1.mean)


# In[23]:


if args.resume:
    checkpoint.load_last()
start_epoch = checkpoint.epoch_counter

print(header)
for epoch in range(0, args.nb_epoch):
    total_loss = train(epoch)
    
    if np.isnan(total_loss):
        print("Losses are NaN, stoping the training here")
        break
        
    test(epoch)

    tensorboard.flush()


# In[24]:


hparams = {}
for key, value in args.__dict__.items():
    hparams[key] = str(value)

final_metrics = {
    "max_acc_1": maximum_tracker.max["acc_1"],
    "max_acc_2": maximum_tracker.max["acc_2"],
} 

tensorboard.add_hparams(hparams, final_metrics)

tensorboard.flush()
tensorboard.close()
