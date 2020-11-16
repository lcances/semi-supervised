from typing import Union
from SSL.ramps import Warmup, sigmoid_rampup
from torchsummary.torchsummary import summary
from SSL.util.model_loader import load_model
import time
import torch
from torch.cuda.amp import autocast
from torch.cuda.memory import empty_cache
import torch.nn as nn
import torch.nn.functional as F
from SSL.util.utils import DotDict
from SSL.trainers.trainers import Trainer
from SSL.losses import JensenShanon
from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage
from SSL.util.utils import track_maximum


class MeanTeacherTrainer(Trainer):
    def __init__(self, model: str, dataset: str,
                 ema_alpha: float = 0.999,
                 teacher_noise_db: int = 0,
                 warmup_length: int = 50,
                 lambda_ccost_max: float = 1,
                 use_softmax: bool = False,
                 ccost_method: str = "mse"):
        super().__init__(model, "mean-teacher", dataset)
        self.ema_alpha = ema_alpha
        self.teacher_noise_db = teacher_noise_db
        self.warmup_length = warmup_length
        self.lambda_ccost_max = lambda_ccost_max
        self.use_softmax = use_softmax
        self.ccost_method = ccost_method

        self.softmax_fn = lambda x: x
        if self.use_softmax:
            self.softmax_fn = nn.Softmax(dim=1)

    def create_model(self):
        print("Creating teacher and student model ...")
        empty_cache()

        model_func = load_model(self.dataset, self.model_str)
        model_params = dict(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
        )

        self.student = model_func(**model_params)
        self.teacher = model_func(**model_params)

        self.student = self.student.cuda()
        self.teacher = self.teacher.cuda()

        summary(self.student, self.input_shape)

    def set_printing_form(self):
        UNDERLINE_SEQ = "\033[1;4m"
        RESET_SEQ = "\033[0m"

        header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<10.8} {:<8.6} {:<8.6} {:<8.6} {:<8.6} {:<8.6} {:<8.6} | {:<10.8} {:<8.6} {:<8.6} {:<8.6} {:<8.6} {:<8.6} - {:<8.6}"
        value_form  = "{:<8.8} {:<6d} - {:<6d} - {:<10.8} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} | {:<10.8} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} - {:<8.4f}"
        header = header_form.format(".               ", "Epoch",  "%", "Student:", "ce", "ccost", "acc_s", "f1_s", "acc_u", "f1_u", "Teacher:", "ce", "acc_s", "f1_s", "acc_u", "f1_u" , "Time")

        self.train_form = value_form
        self.val_form = UNDERLINE_SEQ + value_form + RESET_SEQ

    def init_metrics(self, parameters: DotDict):
        self.metrics = DotDict(
          fscore_ss=FScore(),
          fscore_su=FScore(),
          fscore_ts=FScore(),
          fscore_tu=FScore(),
          acc_ss=CategoricalAccuracy(),
          acc_su=CategoricalAccuracy(),
          acc_ts=CategoricalAccuracy(),
          acc_tu=CategoricalAccuracy(),
          avg_Sce=ContinueAverage(),
          avg_Tce=ContinueAverage(),
          avg_ccost=ContinueAverage(),
        )
        self.maximum_tracker = track_maximum()

        self.scores = DotDict()
        for key in self.metrics:
            self.scores[key] = []

    def init_callbacks(self, parameters: DotDict):
        super().init_callbacks(parameters)

        self.lambda_ccost = Warmup(self.lambda_ccost_max, self.warmup_length, sigmoid_rampup)
        self.callbacks += [self.lambda_ccost]

    def init_checkpoint(self, parameters: DotDict):
        super().init_checkpoint(parameters)
        self.checkpoint.model = [self.student, self.teacher]

    def init_loss(self, parameters: DotDict):
        self.loss_ce = nn.CrossEntropyLoss(reduction="mean")

        if self.ccost_method.lower() == "mse":
            self.loss_cc = nn.MSELoss(reduction="mean")
        elif self.ccost_method.lower() == "js":
            self.loss_cc = JensenShanon

    def train_fn(self, epoch: int):
        # aliases
        M = self.metrics
        T = self.tensorboard.add_scalar
        nb_batch = len(self.train_loader)

        start_time = time.time()
        print("")

        self.reset_metrics()
        self.student.train()

        for i, (S, U) in enumerate(self.train_loader):
            x_s, y_s = S
            x_u, y_u = U

            x_s, x_u = x_s.cuda(), x_u.cuda()
            y_s, y_u = y_s.cuda(), y_u.cuda()

            # Predictions
            with autocast():
                ss_logits = self.student(x_s)
                su_logits = self.student(x_u)
                ts_logits = self.teacher(self._noise_fn(x_s))
                tu_logits = self.teacher(self._noise_fn(x_u))

                # Calculate supervised loss (only student on S)
                loss = self.loss_ce(ss_logits, y_s)

                # Calculate consistency cost (mse(student(x), teacher(x)))
                # x is S + U
                student_logits = torch.cat((ss_logits, su_logits), dim=0)
                teacher_logits = torch.cat((ts_logits, tu_logits), dim=0)
                ccost = self.loss_cc(
                    self.softmax_fn(student_logits),
                    self.softmax_fn(teacher_logits),
                )

                total_loss = loss + self.lambda_ccost() * ccost

            for p in self.student.parameters():
                p.grad = None
            total_loss.backward()
            self.optimizer.step()

            with torch.set_grad_enabled(False):
                # Teacher prediction (for metrics purpose)
                _teacher_loss = self.loss_ce(ts_logits, y_s)

                # Update teacher
                self._update_teacher_model(epoch*nb_batch + i)

                # Compute the metrics for the student
                fscores, accs = self._calc_metrics(
                    ss_logits, su_logits,
                    ts_logits, tu_logits,
                    y_s, y_u,
                )

                # Running average of the two losses
                student_running_loss = M.avg_Sce(loss.item()).mean
                teacher_running_loss = M.avg_Tce(_teacher_loss.item()).mean
                running_ccost = M.avg_ccost(ccost.item()).mean

                # logs
                print(self.train_form.format(
                    "Training: ", epoch + 1, int(100 * (i + 1) / nb_batch),
                    "", student_running_loss, running_ccost,
                    accs.ss, fscores.ss, accs.su, fscores.su,
                    "", teacher_running_loss,
                    accs.ts, fscores.ts, accs.tu, fscores.tu,
                    time.time() - start_time
                ), end="\r")

        T("train/student_acc_s", accs.ss, epoch)
        T("train/student_acc_u", accs.su, epoch)
        T("train/student_f1_s", fscores.ss, epoch)
        T("train/student_f1_u", fscores.su, epoch)

        T("train/teacher_acc_s", accs.ts, epoch)
        T("train/teacher_acc_u", accs.tu, epoch)
        T("train/teacher_f1_s", fscores.ts, epoch)
        T("train/teacher_f1_u", fscores.tu, epoch)

        T("train/student_loss", student_running_loss, epoch)
        T("train/teacher_loss", teacher_running_loss, epoch)
        T("train/consistency_cost", running_ccost, epoch)

    def val_fn(self, epoch: int):
        # aliases
        M = self.metrics
        T = self.tensorboard.add_scalar
        nb_batch = len(self.val_loader)

        start_time = time.time()
        print("")

        self.reset_metrics()
        self.student.eval()

        with torch.set_grad_enabled(False):
            for i, (X, y) in enumerate(self.val_loader):
                X = X.cuda()
                y = y.cuda()

                # Predictions
                with autocast():
                    student_logits = self.student(X)
                    teacher_logits = self.teacher(X)

                    # Calculate supervised loss (only student on S)
                    loss = self.loss_ce(student_logits, y)
                    _teacher_loss = self.loss_ce(teacher_logits, y)  # for metrics only
                    ccost = self.loss_cc(
                        self.softmax_fn(student_logits),
                        self.softmax_fn(teacher_logits))

                # Compute the metrics
                fscores, accs = self._calc_metrics(
                    student_logits, student_logits, 
                    teacher_logits, teacher_logits,
                    y, y
                )

                # Running average of the two losses
                student_running_loss = M.avg_Sce(loss.item()).mean
                teacher_running_loss = M.avg_Tce(_teacher_loss.item()).mean
                running_ccost = M.avg_ccost(ccost.item()).mean

                # logs
                print(self.val_form.format(
                    "Validation: ", epoch + 1, int(100 * (i + 1) / nb_batch),
                    "", student_running_loss, running_ccost, accs.ss, fscores.ss, 0.0, 0.0,
                    "", teacher_running_loss, accs.ts, fscores.ts, 0.0, 0.0,
                    time.time() - start_time
                ), end="\r")

        T("val/student_acc", accs.ss, epoch)
        T("val/student_f1", fscores.ss, epoch)
        T("val/teacher_acc", accs.ts, epoch)
        T("val/teacher_f1", fscores.ts, epoch)
        T("val/student_loss", student_running_loss, epoch)
        T("val/teacher_loss", teacher_running_loss, epoch)
        T("val/consistency_cost", running_ccost, epoch)

        T("hyperparameters/learning_rate", self._get_lr(), epoch)
        T("hyperparameters/lambda_cost_max", self.lambda_cost(), epoch)

        T("max/student_acc", self.maximum_tracker("student_acc", accs.ss), epoch)
        T("max/teacher_acc", self.maximum_tracker("teacher_acc", accs.ts), epoch)
        T("max/student_f1", self.maximum_tracker("student_f1", fscores.ss), epoch)
        T("max/teacher_f1", self.maximum_tracker("teacher_f1", fscores.ts), epoch)

    def test_fn(self):
        pass

    def _update_teacher_model(self, epoch):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (epoch + 1), self.ema_alpha)

        for param, ema_param in zip(self.student.parameters(), self.teacher.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha)

    def _noise_fn(self, x):
        if self.teacher_noise_db == 0:
            return x

        n_db = self.teacher_noise_db
        return x + (torch.rand(x.shape).cuda() * n_db + n_db)

    def _calc_metrics(self,
                      ss_logits, su_logits, ts_logits, tu_logits,
                      y_s, y_u) -> Union[DotDict, DotDict]:

        with torch.set_grad_enable(False):
            S = nn.Softmax(dim=1)
            A = lambda x: torch.argmax(x, dim=1)
            M = self.metrics

            one_hot_s = F.one_hot(y_s, self.num_classes)
            one_hot_u = F.one_hot(y_u, self.num_classes)

            fscores = DotDict(
                fscore_ss=M.fscore_ss(S(ss_logits), one_hot_s).mean,
                fscore_su=M.fscore_su(S(su_logits), one_hot_u).mean,
                fscore_ts=M.fscore_ts(S(ts_logits), one_hot_s).mean,
                fscore_tu=M.fscore_tu(S(tu_logits), one_hot_u).mean,
            )

            accs = DotDict(
                acc_ss=M.acc_ss(A(ss_logits), y_s).mean,
                acc_su=M.acc_su(A(su_logits), y_u).mean,
                acc_ts=M.acc_ts(A(ts_logits), y_s).mean,
                acc_tu=M.acc_tu(A(tu_logits), y_u).mean,
            )

            return fscores, accs
