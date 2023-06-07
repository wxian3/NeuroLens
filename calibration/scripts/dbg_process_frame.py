import cv2
import logging
from os import path
import hydra
import imageio
from ..config import Config, ARUCO_DICT
from ..keypoint_detection import process_frame


LOGGER = logging.getLogger(__name__)
CONF_FP: str = path.join("..", "..", "conf")


@hydra.main(config_path=CONF_FP, config_name="calibration_config")
def cli(cfg: Config):
    """
    Process a single frame for debugging purposes.

    Uses `tmp.png` in the project folder. Enable debugging visualizations
    by adding `dbg=true`. Example call:
    `pdm run dbg_process_frame -- target.exp_name=[your experiment name] dbg=true`.
    """
    frame = imageio.imread(path.join(path.dirname(__file__), "..", "..", "tmp.png"))[
        :, :, ::-1
    ]
    process_frame(frame, cfg)
