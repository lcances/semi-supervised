import os
from SSL.util.model_loader import load_model
from SSL.util.utils import get_datetime, DotDict, save_source_as_img
from torch.cuda import empty_cache
from torch.nn import parameter
from torch.utils.tensorboard.writer import SummaryWriter
from SSL.util.loaders import load_callbacks, load_dataset, load_optimizer, load_preprocesser


class Trainer:
    def __init__(self, model: str, framework: str, dataset: str):
        # models
        self.model_str = model
        self.model = None

        self.framework = framework
        self.dataset = dataset

        self.train_transform = None
        self.val_transform = None
        self.input_shape = None
        self.input_classes = self._auto_num_classes()

        self.manager = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.tensorboard = None
        self.checkpoint = None

        abs_path = os.path.dirname(os.path.abspath(__file__))
        self.tensorboard_path = os.path.join(abs_path, "..", "..", "tensorboard")
        self.checkpoint_path = os.path.join(abs_path, "..", "..", "model_save")

        self.optimizer = None

    def init_trainer(self,
                     dataset_root: str,
                     supervised_ratio: float,
                     batch_size: int,
                     train_folds: tuple,
                     val_folds: tuple,
                     num_workers: int = 0,
                     pin_memory: bool = True,
                     verbose: int = 1):
        fn_arguments = DotDict(**locals())

        # Load transformation function
        self.load_transforms()

        # Load the dataset and get input shape
        self.load_dataset(**fn_arguments)

        # Create the model with the proper input shape and num classes
        self.create_model()

        # Init the different components required for training
        self.init_logs(**fn_arguments)
        self.init_optimizer(**fn_arguments)
        self.init_checkpoint(**fn_arguments)
        self.init_callbacks(**fn_arguments)

    def load_transforms(self):
        transforms = load_preprocesser(self.dataset, self.framework)
        self.train_transform, self.val_transform = transforms

    def load_dataset(self, **parameters):
        outputs = load_dataset(
            self.dataset,
            self.framework,

            train_transform=self.train_transform,
            val_transform=self.val_transform,

            **parameters
        )

        self.manager, self.train_loader, self.val_loader = outputs
        self.input_shape = tuple(self.train_loader.dataset[0][0].shape)

    def create_model(self):
        empty_cache()

        model_func = load_model(self.dataset, self.model_str)
        self.model = model_func(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
        )

    def init_logs(self, parameters: dict):
        title_element = (
            self.model_str,
            parameters["supervised_ratio"],
            get_datetime(),
            self.model_str,
            parameter["supervised_ratio"],
        )

        tensorboard_title = "%s/%sS/%s_%s_%.1fS" % title_element

        self.tensorboard = SummaryWriter(log_dir="%s/%s" % (self.tensorboard_path, tensorboard_title))

    def init_checkpoint(self, parameters: dict):
        title_element = (
            self.model_str,
            parameters["supervised_ratio"],
            get_datetime(),
            self.model_str,
            parameter["supervised_ratio"],
        )

        checkpoint_title = "%s/%sS/%s_%s_%.1fS" % title_element
        self.checkpoint(self.model, self.optimizer,
                        mode="max",
                        name=f"{self.checkpoint_path}, {checkpoint_title}")

    def init_optimizer(self, parameters: dict):
        self.optimizer = load_optimizer(
            self.dataset,
            self.framework,
            learning_rate=parameters["learning_rate"],
            model=self.model,
        )

    def init_callbacks(self, parameters: dict):
        self.callbacs = load_callbacks(
            self.dataset,
            self.framework,
            optimizer=self.optimizer,
            nb_epoch=parameters["nb_epoch"],
        )

    def init_metrics(self, parameters: dict):
        raise NotImplementedError()

    def reset_metrics(self):
        for m in self.metrics:
            m.reset()

    def train_fn(self):
        if self.train_loader is None:
            print("No training function define")
            return

    def val_fn(self):
        if self.val_loader is None:
            print("No validation function define")
            return

    def test_fn(self):
        if self.test_loader is None:
            print("No testing function define")
            return

    def _auto_num_classes(self) -> int:
        if self.dataset.lower() in ["esc10, ubs8k"]:
            return 10

        elif self.dataset.lower() in ["speechcommand"]:
            return 35

        elif self.dataset.lower() in ["esc50"]:
            return 50

        else:
            raise ValueError(f"Dataset {self.dataset} is not available")
