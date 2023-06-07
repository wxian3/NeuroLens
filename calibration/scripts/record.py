import sys
import cv2
import numpy as np
import logging
import os
from os import path
from calibration.persistence import NumpyEncoder
import hydra
import imageio
import json
import shutil
import click
from ..persistence import NumpyEncoder
from ..config import Config
from ..keypoint_detection import process_frame


LOGGER = logging.getLogger(__name__)
CONF_FP: str = path.join("..", "..", "conf")


@hydra.main(config_path=CONF_FP, config_name="calibration_config")
def cli(cfg: Config):
    """
    Run target detection live or for a video and store the results in `data/live`.

    By default uses the video camera 0. You can also specify a pre-recorded video
    by using the parameter `video_fp=[path to your.mp4]`. Example call:
    `pdm run record -- target.exp_name=[your experiment name] [video_fp=[path to .mp4]]`.
    Add `vis=true` to see a live visualization of the detections.
    """
    LOGGER.info(
        f"Starting live detection for board with marker `{cfg.target.aruco_id}`..."
    )
    storage_fp = path.abspath(path.join(CONF_FP, "..", "..", "data", "live"))
    if path.exists(storage_fp):
        LOGGER.info(f"Data collection path exists ({storage_fp}).")
        if click.confirm("Do you want to delete it to collect new data?"):
            shutil.rmtree(storage_fp)
        else:
            LOGGER.error("Can't continue. Aborting...")
            sys.exit(1)
    os.makedirs(storage_fp)
    LOGGER.info("Starting live view...")
    if cfg.video_fp is not None:
        cap = cv2.VideoCapture(path.abspath("../../../testcap.mp4"))
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Cannot open camera!")
    if cfg.vis:
        LOGGER.info("Starting witness view...")
        cv2.namedWindow("coverage")
    coverage_vis = None
    LOGGER.info("Live view running!")
    frame_coords = []
    board_coords = []
    # TODO: make robust w.r.t. inaccurate marker detections / improve marker accuracy.
    # TODO: reject detections where the center is too far from expected.
    while True:
        ret, frame = cap.read()
        if coverage_vis is None:
            coverage_vis = np.zeros_like(frame)
        if not ret:
            LOGGER.error("Can't read frame (stream end?). Exiting ...")
            break
        frame_orig = frame.copy()
        try:
            (
                keypoint_coords_xy,
                keypoints_valid,
                keypoint_board_coords,
                vis_frame,
            ) = process_frame(frame, cfg)
        except Exception as ex:
            LOGGER.error(ex)
            continue
        this_frame_coords = []
        this_board_coords = []
        resize_fac = 3
        frame = cv2.resize(
            frame, None, fx=resize_fac, fy=resize_fac, interpolation=cv2.INTER_LINEAR
        )
        for center_coord, valid, board_coord in zip(
            keypoint_coords_xy, keypoints_valid, keypoint_board_coords
        ):
            clr = (0, 255, 0) if valid else (0, 0, 255)
            frame[
                int(center_coord[1] * resize_fac)
                - 3 : int(center_coord[1] * resize_fac)
                + 3,
                int(center_coord[0] * resize_fac)
                - 3 : int(center_coord[0] * resize_fac)
                + 3,
                :,
            ] = clr
            text = str("(%d, %d)" % (int(board_coord[0]), int(board_coord[1])))
            text_width, text_height = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1
            )[0]
            cv2.putText(
                frame,
                text,
                (
                    int(center_coord[0] * resize_fac),
                    int(center_coord[1] * resize_fac) - text_height,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 255, 0),
                1,
            )
            if valid:
                coverage_vis[int(center_coord[1]), int(center_coord[0]), 1] = min(
                    255,
                    50 + coverage_vis[int(center_coord[1]), int(center_coord[0]), 1],
                )
                this_frame_coords.append(center_coord)
                this_board_coords.append(board_coord)
        if len(this_frame_coords) > 0:
            frame_coords.append(this_frame_coords)
            board_coords.append(this_board_coords)
            cv2.imwrite(
                path.join(storage_fp, "%05d.png" % (len(board_coords) - 1)), frame_orig
            )
            cv2.imwrite(
                path.join(storage_fp, "vis-%05d.png" % (len(board_coords) - 1)),
                vis_frame,
            )
        # Display the resulting frame and update coverage.
        if cfg.vis:
            cv2.imshow("frame", frame)
            cv2.imshow("coverage", coverage_vis)
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
    if cfg.vis:
        cv2.destroyAllWindows()
    LOGGER.info("Writing results...")
    imageio.imwrite(path.join(storage_fp, "coverage.png"), coverage_vis)
    with open(path.join(storage_fp, "points.json"), "w") as outf:
        json.dump(
            {
                "frame_coordinates_xy": frame_coords,
                "board_coordinates_xyz": board_coords,
                "resolution_wh": (coverage_vis.shape[1], coverage_vis.shape[0]),
            },
            outf,
            cls=NumpyEncoder,
            indent=4,
            sort_keys=False,
        )
