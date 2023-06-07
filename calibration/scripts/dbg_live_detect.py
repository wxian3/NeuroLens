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
    [debugging] Script to detect a board live and show markers.

    Press `s` to save a frame to `tmp.png` for later processing, press
    `q` to exit. Example run:
    `pdm run dbg_live_detect -- target.exp_name=[your experiment name]`.
    You can also read a pre-recorded video using the option
    `video_fp=[path to video]`.
    """
    LOGGER.info(
        f"Starting live detection for board with marker `{cfg.target.aruco_id}`..."
    )
    LOGGER.info("Loading marker...")
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[cfg.target.aruco_id])
    arucoParams = cv2.aruco.DetectorParameters_create()
    arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    LOGGER.info("Starting live view...")
    if cfg.video_fp is not None:
        print(path.abspath(cfg.video_fp))
        cap = cv2.VideoCapture(path.abspath(cfg.video_fp))
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Cannot open camera!")
    LOGGER.info("Live view running!")
    while True:
        ret, frame = cap.read()
        if not ret:
            raise Exception("Can't read frame (stream end?). Exiting ...")
        frame_orig = frame.copy()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(
            frame, arucoDict, parameters=arucoParams
        )
        if len(corners) > 0:
            keypoint_coords_xy, keypoints_valid, _, frame = process_frame(frame, cfg)
        # Display the resulting frame
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code == ord("q"):
            break
        elif key_code == ord("s"):
            imageio.imsave(
                path.join(path.dirname(__file__), "..", "..", "tmp.png"),
                frame_orig[:, :, ::-1],
            )
    LOGGER.info("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
