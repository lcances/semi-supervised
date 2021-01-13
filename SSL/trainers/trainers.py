import os
from SSL.util.model_loader import load_model
from SSL.util.utils import get_datetime, DotDict, reset_seed, track_maximum
from torch.cuda import empty_cache
from torch.utils.tensorboard.writer import SummaryWriter
from SSL.util.loaders import load_callbacks
from SSL.util.loaders import load_dataset, load_optimizer, load_preprocesser
from SSL.util.checkpoint import CheckPoint
from SSL.util.checkpoint import mSummaryWriter as SummaryWriter
from torchsummary import summary


class Trainer:
    def __init__(self, model: str, framework: str, dataset: str):
        # models
        self.model_str = model
        self.model = None

        self.framework = framework
        self.dataset = dataset

        # Dataset related variables
        self.manager = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_transform = None
        self.val_transform = None
        self.input_shape = None
        self.num_classes = self._auto_num_classes()

        # Logs related
        abs_path = os.path.dirname(os.path.abspath(__file__))
        self.tensorboard_path = os.path.join(abs_path, "..", "..", "tensorboard", dataset, framework )
        self.checkpoint_path = os.path.join(abs_path, "..", "..", "model_save", dataset, framework)
        self.tensorboard = None
        self.checkpoint = None

        # Training related variables
        self.optimizer = None
        self.callbacks = None
        self.parameters = None
        self.metrics = DotDict()
        self.maximum_tracker = track_maximum()

        # Other parameters
        self.parameters = None
        self.extra_hparams = DotDict()

    def init_trainer(self,
                     parameters: dict,
                     num_workers: int = 0,
                     pin_memory: bool = True,
                     verbose: int = 1,):
        """ Mandatory parameters
        dataset_root
        supervised_ratio
        batch_size
        seed
        learning_rate
        nb_epoch
        """
        parameters = DotDict(**parameters)
        self.parameters = parameters
        # Reset the seed
        reset_seed(parameters.seed)

        # Load transformation function
        self.load_transforms()

        # Load the dataset and get input shape
        self.load_dataset(parameters)

        # Create the model with the proper input shape and num classes
        self.create_model()

        # Init the different components required for training
        self.init_loss(parameters)
        self.init_optimizer(parameters)
        self.init_callbacks(parameters)
        self.init_logs(parameters)
        self.init_checkpoint(parameters)
        self.init_metrics(parameters)

        print(">>> Trainer is ready")

    def load_transforms(self):
        print("Load the transformation")
        transforms = load_preprocesser(self.dataset, self.framework)
        self.train_transform, self.val_transform = transforms

    def load_dataset(self, parameters: DotDict):
        print("Load the dataset")
        outputs = load_dataset(
            framework=self.framework,
            train_transform=self.train_transform,
            val_transform=self.val_transform,
            **parameters
        )

        self.manager, self.train_loader, self.val_loader = outputs
        self.input_shape = self._get_input_shape()

    def _get_input_shape(self):
        return tuple(self.train_loader.dataset[0][0].shape)

    def create_model(self):
        print("Create the model")
        empty_cache()

        model_func = load_model(self.dataset, self.model_str)
        self.model = model_func(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
        )
        self.model = self.model.cuda()

        s = summary(self.model, self.input_shape)

    def init_checkpoint(self, parameters: DotDict):
        print("Prepare the checkpoint system")
        title_element = (
            self.model_str,
            parameters.supervised_ratio,
            self.model_str,
            parameters.supervised_ratio,
        )

        checkpoint_title = "%s/%sS/%s_%.1fS" % title_element
        self.checkpoint = CheckPoint(
            self.model, self.optimizer,
            mode="max",
            name=f"{self.checkpoint_path}/{checkpoint_title}")

    def init_optimizer(self, parameters: DotDict):
        print("Initialize optimizer")
        self.optimizer = load_optimizer(
            self.dataset,
            self.framework,
            learning_rate=parameters.learning_rate,
            model=self.model,
        )

    def init_callbacks(self, parameters: DotDict):
        print("Initialize callbacks")
        self.callbacks = load_callbacks(
            self.dataset,
            self.framework,
            optimizer=self.optimizer,
            nb_epoch=parameters.nb_epoch,
        )

    def init_loss(self, parameters: DotDict):
        print("Initialize loss function")
        raise NotImplementedError

    def init_metrics(self, parameters: DotDict):
        print("Initialize metrics")
        raise NotImplementedError()

    def reset_metrics(self):
        for key, fn in self.metrics.items():
            fn.reset()

    # =========================================================================
    #       TRAINING METHODS
    # =========================================================================
    def set_printing_form(self):
        pass

    def train_fn(self, epoch: int):
        if self.train_loader is None:
            print("No training function define")
            return

    def val_fn(self, epoch: int):
        if self.val_loader is None:
            print("No validation function define")
            return

    def test_fn(self):
        if self.test_loader is None:
            print("No testing function define")
            return

    def fit(self, resume: bool = False):
        start_epoch = 0

        self.set_printing_form()

        if resume:
            start_epoch = self.checkpoint.epoch_counter

        for e in range(start_epoch, self.parameters.nb_epoch):
            self.train_fn(e)
            self.val_fn(e)

            self.tensorboard.flush()

        self.test_fn()

        self.save_hparams(self.parameters)

    # =========================================================================
    #       LOG METHODS
    # =========================================================================
    def init_logs(self, parameters: DotDict):
        print("Prepare the log system")
        title_element = (
            self.model_str,
            parameters.supervised_ratio,
            get_datetime(),
            self.model_str,
            parameters.supervised_ratio,
        )

        tensorboard_title = "%s/%sS/%s_%s_%.1fS" % title_element

        self.tensorboard = SummaryWriter(log_dir="%s/%s" % (self.tensorboard_path, tensorboard_title))

    def save_hparams(self, parameters: dict):
        hparams = dict(
            **self.extra_hparams, **parameters
        )

        hparams["model"] = self.model_str

        # Transform every Iterable into a str
        for key, value in parameters.items():
            if isinstance(value, (list, tuple, set)):
                hparams[key] = "["+",".join(map(str, value))+"]"
            else:
                hparams[key] = str(value)
        
        final_metrics = self.maximum_tracker.max

        self.tensorboard.add_hparams(hparams, final_metrics)

    def close(self):
        self.tensorboard.flush()
        self.tensorboard.close()

    # =========================================================================
    #       UTILITY METHODS
    # =========================================================================
    def _auto_num_classes(self) -> int:
        if self.dataset.lower() in ["esc10", "ubs8k"]:
            return 10

        elif self.dataset.lower() in ["speechcommand"]:
            return 35

        elif self.dataset.lower() in ["speechcommand10"]:
            return 12

        elif self.dataset.lower() in ["esc50"]:
            return 50

        else:
            raise ValueError(f"Dataset {self.dataset} is not available")

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]
