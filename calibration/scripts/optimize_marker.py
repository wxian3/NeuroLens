from os import path
from datetime import timedelta
import logging
import hydra
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate
from calibration.config import Config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

LOGGER = logging.getLogger(__name__)
CONF_FP: str = path.join("..", "..", "conf")


class EmptyDataset(Dataset):
    """
    Dummy dataset object.

    Since the training data is generated on the fly, no
    actual data loader is required. This dataset type
    makes this training paradigm compatible with PyTorch
    lightning.
    """

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return []


@hydra.main(config_path=CONF_FP, config_name="calibration_config")
def cli(cfg: Config):
    """
    Optimize the marker appearance and keypoint detector.

    See README.md for further information. Example call:
    `pdm run optimize_marker -- exp_name=[your experiment name]`.
    """
    LOGGER.info(
        f"Optimizing marker of size {cfg.model.marker.size}x{cfg.model.marker.size}."
    )
    LOGGER.info("Instantiating model...")
    model = instantiate(cfg.model, _recursive_=False)
    log_dir = path.join(
        path.abspath(path.dirname(__file__)), "..", "..", "experiments", cfg.exp_name
    )
    LOGGER.info(f"Writing logs and tensorboard to `{log_dir}`.")
    train_loader = DataLoader(
        EmptyDataset(cfg.model.batch_size * cfg.trainer.max_steps),
        batch_size=cfg.model.batch_size,
    )
    trainer = instantiate(
        cfg.trainer,
        default_root_dir=log_dir,
        callbacks=[ModelCheckpoint(train_time_interval=timedelta(minutes=10.0))],
    )
    trainer.fit(model, train_dataloaders=train_loader)
    return np.mean(model.loss_deque)


if __name__ == "__main__":
    cli()
