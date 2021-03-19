from torch.utils.tensorboard import SummaryWriter
import torch
import os


class CheckPoint:
    def __init__(self, model: list, optimizer,
                 mode: str = "max", name: str = "best",
                 verbose: bool = True):
        self.mode = mode

        self.name = name
        self.verbose = verbose

        self.model = model
        self.optimizer = optimizer
        self.best_state = dict()
        self.last_state = dict()
        self.best_metric = None
        self.epoch_counter = 0

        # Preparation
        if not isinstance(self.model, list):
            self.model = [self.model]

        self.create_directory()
        self._init_message()

    def create_directory(self):
        os.makedirs(os.path.dirname(self.name), exist_ok=True)

    def _init_message(self):
        if self.verbose:
            print('checkpoint initialise at: ', os.path.abspath(self.name))
            print('name: ', os.path.basename(self.name))
            print('mode: ', self.mode)

    def step(self, new_value, iter: int = None):
        if self.epoch_counter == 0:
            self.best_metric = new_value

        # Save last epoch
        self.last_state = self._get_state(new_value, iter)
        torch.save(self.last_state, self.name + ".last")

        # save best epoch
        if self._check_is_better(new_value):
            if self.verbose:
                print("\n better performance: saving ...")

            self.best_metric = new_value
            self.best_state = self._get_state(new_value)
            torch.save(self.best_state, self.name + ".best")

        self.epoch_counter += 1

    def _get_state(self, new_value=None, iter: int = None) -> dict:
        state = {
            "state_dict": [m.state_dict() for m in self.model],
            "optimizer": self.optimizer.state_dict(),
        }
        state['epoch'] = self.epoch_counter if iter is None else iter

        if new_value is not None:
            state["best_metric"] = new_value

        return state

    def save(self):
        torch.save(self._get_state, self.name + ".last")

    def load(self, path):
        data = torch.load(path)
        self._load_helper(data, self.last_state)
        self._load_helper(data, self.best_state)

    def load_best(self):
        if not os.path.isfile(self.name + ".best"):
            return

        data = torch.load(self.name + ".best")
        self._load_helper(data, self.best_state)

    def load_last(self):
        if not os.path.isfile(self.name + ".last"):
            print(f"File {self.name}.last doesn't exist")
            return

        data = torch.load(self.name + ".last")
        self._load_helper(data, self.last_state)
        print("Last save loaded ...")

    def _load_helper(self, state, destination):
        print(list(state.keys()))
        for k, v in state.items():
            destination[k] = v

        self.optimizer.load_state_dict(destination["optimizer"])
        self.epoch_counter = destination["epoch"]
        self.best_metric = destination["best_metric"]

        # Path to fit with previous version of checkpoint
        if not isinstance(destination["state_dict"], list):
            destination["state_dict"] = [destination["state_dict"]]

        for i in range(len(self.model)):
            self.model[i].load_state_dict(destination["state_dict"][i])

    def _check_is_better(self, new_value):
        if self.best_metric is None:
            self.best_metrics = new_value
            return True

        return self.best_metric < new_value

    
class mSummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix=''):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        self.history = dict()
        
    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        super().add_scalar(tag, scalar_value, global_step, walltime)
        
        if tag not in self.history:
            self.history[tag] = [scalar_value]
        else:
            self.history[tag].append(scalar_value)