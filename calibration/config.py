from dataclasses import dataclass, field
from enum import Enum
from os import path
from glob import glob
from natsort import natsorted
from typing import List, Optional, Dict, Tuple, Union
import cv2

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


@dataclass
class ConfigCalibration:
    subs_fac: int = 5


@dataclass
class ConfigMarker:
    _target_: str = "calibration.marker.Marker"
    size: int = MISSING
    working_size: int = MISSING
    seed: int = 1
    # 95% of points are within 2 * std_thresh.
    std_thresh: float = 0.20
    # std_thresh: float = 0.0


@dataclass
class ConfigRandomMotionBlur:
    _target_: str = "kornia.augmentation.RandomMotionBlur"
    p: float = 0.0
    kernel_size: int = 5
    angle: float = 90.0
    direction: float = 1.0
    border_type: str = "reflect"


@dataclass
class ConfigRandomAffine:
    _target_: str = "kornia.augmentation.RandomAffine"
    p: float = 0.0
    degrees: float = 180.0
    translate: Tuple[float, float] = (0.05, 0.05)
    scale: Tuple[float, float, float, float] = (0.95, 1.05, 0.95, 1.05)
    shear: float = 0.02
    return_transform: bool = True
    padding_mode: str = "reflection"


@dataclass
class ConfigRandomGaussianBlur:
    _target_: str = "kornia.augmentation.RandomGaussianBlur"
    p: float = 0.3
    # We aim to find a model that can reliably detect the marker
    # up to 1/2 it's size (in pixels).
    # For down-scaling by factor two, σ=3.3 is a good value.
    # The half filter size should be
    # around 3σ, so we need a full filter size of 2*3*3.3, so around
    # 19.
    # Must be an odd, positive integer.
    kernel_size: int = (11, 11)
    # sigma: Tuple[float, float] = (3.3, 3.3)
    sigma: Tuple[float, float] = (1.6, 1.6)
    border_type: str = "reflect"


@dataclass
class ConfigColorJitter:
    _target_: str = "kornia.augmentation.ColorJitter"
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1
    hue: float = 0.1
    return_transform: bool = False
    same_on_batch: bool = False
    p: float = 0.0


@dataclass
class ConfigRandomElasticTransform:
    _target_: str = "calibration.transformations.RandomElasticTransform"
    # Reactivate once a fix or mitigation is found:
    # https://fb.workplace.com/groups/349226332644221/permalink/949543442612504/
    kernel_size: Tuple[int, int] = (63, 63)
    sigma: Tuple[float, float] = (32.0, 32.0)
    alpha: Tuple[float, float] = (1.0, 1.0)
    align_corners: bool = False
    mode: str = "bilinear"
    padding_mode: str = "mirror"
    return_transform: bool = True
    same_on_batch: bool = False
    p: float = 0.0


@dataclass
class ConfigRandomGaussianNoise:
    _target_: str = "kornia.augmentation.RandomGaussianNoise"
    mean: float = 0.0
    std: float = 0.1
    p: float = 1.0


@dataclass
class ConfigImageFormation:
    _target_: str = "calibration.image_formation.ImageFormation"
    # print_dist: ConfigRandomElasticTransform = ConfigRandomElasticTransform()
    motion_dist: ConfigRandomMotionBlur = ConfigRandomMotionBlur()
    affine_dist: ConfigRandomAffine = ConfigRandomAffine()
    blur_dist: ConfigRandomGaussianBlur = ConfigRandomGaussianBlur()
    color_dist: ConfigColorJitter = ConfigColorJitter()
    noise_dist: ConfigRandomGaussianNoise = ConfigRandomGaussianNoise()
    working_size: int = MISSING


@dataclass
class ConfigModel:
    _target_: str = "calibration.model.Model"
    marker: ConfigMarker = MISSING
    image_formation: ConfigImageFormation = ConfigImageFormation()
    batch_size: int = 20
    log_every: float = 30.0
    lr: float = 1e-3
    lr_fcn_fac: float = 1.0
    lr_marker_fac: float = 100.0
    n_latent: int = 200
    n_hidden: int = 2
    reg_weight: float = 0.0


@dataclass
class ConfigTrainer:
    _target_: str = "pytorch_lightning.Trainer"
    logger: bool = True
    enable_checkpointing: bool = True
    default_root_dir: str = ""
    max_steps: int = 100000
    log_every_n_steps: int = 50
    enable_model_summary: bool = True
    accelerator: str = "gpu"
    gpus: Tuple[int] = (0,)
    auto_select_gpus: bool = True


@dataclass
class ConfigTarget:
    exp_name: str = MISSING
    aruco_id: str = "DICT_4X4_1000"
    margin_in_cm: float = 0.4
    approx_square_length_in_cm: float = 1.25
    aruco_length_in_squares: int = 4
    page_size_pt: Tuple[float, float] = (595.2755905511812, 841.8897637795277)


@dataclass
class Config:
    gpus: int = MISSING
    disable_tqdm: bool = MISSING
    model: ConfigModel = MISSING
    calibration: ConfigCalibration = ConfigCalibration()
    trainer: ConfigTrainer = ConfigTrainer()
    exp_name: str = MISSING
    target: ConfigTarget = ConfigTarget()
    dbg: bool = False
    vis: bool = False
    video_fp: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="calibration_base_config", node=Config)


def get_latest_checkpoint(exp_name: str, cfg: Config):
    return path.abspath(
        natsorted(
            glob(
                path.join(
                    path.dirname(__file__),
                    "..",
                    "experiments",
                    cfg.target.exp_name,
                    "lightning_logs",
                    "version_0",
                    "checkpoints",
                    "*.ckpt",
                )
            )
        )[-1]
    )
