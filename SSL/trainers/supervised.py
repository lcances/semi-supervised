import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from SSL.util.utils import DotDict
from SSL.trainers.trainers import Trainer
from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage
from SSL.util.utils import track_maximum


class SupervisedTrainer(Trainer):
    def __init__(self, model: str, dataset: str):
        super().__init__(model, "supervised", dataset)

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

    def init_metrics(self, parameters: DotDict):
        self.metrics = DotDict(
            fscore_fn=FScore(),
            acc_fn=CategoricalAccuracy(),
            avg_fn=ContinueAverage(),
        )
        self.maximum_tracker = track_maximum()

        self.set_printing_form()

    def init_loss(self, parameters: dict):
        self.loss_ce = nn.CrossEntropyLoss(reduction="mean")

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
            X = X.cuda()
            y = y.cuda()

            logits = self.model(X)
            loss = self.loss_ce(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.set_grad_enabled(False):
                pred = torch.softmax(logits, dim=1)
                pred_arg = torch.argmax(logits, dim=1)
                y_one_hot = F.one_hot(y, num_classes=self.num_classes)

                acc = M.acc_fn(pred_arg, y).mean
                fscore = M.fscore_fn(pred, y_one_hot).mean
                avg_ce = M.avg_fn(loss.item()).mean

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
                X = X.cuda()
                y = y.cuda()

                logits = self.model(X)
                loss = self.loss_ce(logits, y)

                # metrics
                pred = torch.softmax(logits, dim=1)
                pred_arg = torch.argmax(logits, dim=1)
                y_one_hot = F.one_hot(y, num_classes=self.num_classes)

                acc = M.acc_fn(pred_arg, y).mean
                fscore = M.fscore_fn(pred, y_one_hot).mean
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

    def test_fn(self):
        super().test_fn()
        # aliases
        M = self.metrics
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

                # metrics
                pred = torch.softmax(logits, dim=1)
                pred_arg = torch.argmax(logits, dim=1)
                y_one_hot = F.one_hot(y, num_classes=self.num_classes)

                acc = M.acc_fn(pred_arg, y).mean
                fscore = M.fscore_fn(pred, y_one_hot).mean
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
