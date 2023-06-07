import logging
import numpy as np
from os import path
from calibration.target import calculate_parameters

import hydra
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader as RLImageReader
import imageio
import tempfile

from ..model import Model
from ..config import Config, ARUCO_DICT, get_latest_checkpoint

LOGGER = logging.getLogger(__name__)
CONF_FP: str = path.join("..", "..", "conf")


@hydra.main(config_path=CONF_FP, config_name="calibration_config")
def cli(cfg: Config):
    """
    Assemble a calibration pattern using an optimized marker.

    For more information, please see README.md. Example run:
    `pdm run assemble_pattern -- target.exp_name=[your experiment name]`.
    """
    LOGGER.info(
        f"Assembling calibration target with section from experiment `{cfg.target.exp_name}`."
    )
    latest_checkpoint = get_latest_checkpoint(cfg.target.exp_name, cfg)
    with tempfile.TemporaryDirectory() as tmp_dir:
        LOGGER.info(f"Loading optimized marker from `{latest_checkpoint}`...")
        model = Model.load_from_checkpoint(latest_checkpoint)
        marker, center = model.marker(1)
        marker = (marker[0].cpu().detach().numpy().transpose(1, 2, 0) * 255.0).astype(
            np.uint8
        )
        assert center[0, 0] == marker.shape[1] // 2
        assert center[0, 1] == marker.shape[0] // 2
        imageio.imsave(path.join(tmp_dir, "marker.png"), marker)
        marker_im = RLImageReader(path.join(tmp_dir, "marker.png"))
        if cfg.dbg:
            cv2.namedWindow("[dbg] Optimized Marker")
            cv2.imshow("[dbg] Optimized Marker", marker)
            cv2.waitKey(0)
            cv2.destroyWindow("[dbg] Optimized Marker")
        LOGGER.info(f"Loading ArUco marker with ID `{cfg.target.aruco_id}`...")
        ar_markers = [
            cv2.aruco.drawMarker(
                cv2.aruco.getPredefinedDictionary(ARUCO_DICT[cfg.target.aruco_id]),
                idx,
                200,
            )
            for idx in range(23, 27)
        ]
        for marker_idx, ar_marker in enumerate(ar_markers):
            imageio.imsave(
                path.join(tmp_dir, "ar_marker_%d.png" % (marker_idx)), ar_marker
            )
        ar_marker_ims = [
            RLImageReader(path.join(tmp_dir, "ar_marker_%d.png" % (idx)))
            for idx in range(4)
        ]
        if cfg.dbg:
            cv2.namedWindow("[dbg] ArUco Marker")
            cv2.imshow("[dbg] ArUco Marker", ar_marker)
            cv2.waitKey(0)
            cv2.destroyWindow("[dbg] ArUco Marker")
        pdf_path = path.abspath(
            path.join(
                path.dirname(__file__),
                "..",
                "..",
                "experiments",
                cfg.target.exp_name,
                "target.pdf",
            )
        )
        LOGGER.info(f"Generating PDF at `{pdf_path}`...")
        canv = canvas.Canvas(pdf_path, pagesize=cfg.target.page_size_pt)
        params = calculate_parameters(cfg)
        LOGGER.info(
            f"Squares printed: {params.n_squares_x} X {params.n_squares_y} (x X y)."
        )
        LOGGER.info("Drawing markers...")
        for x in range(0, params.n_squares_x):
            for y in range(0, params.n_squares_y):
                corner_coord_x = params.pattern_start_x_pt + x * params.square_length_pt
                corner_coord_y = params.pattern_start_y_pt + y * params.square_length_pt
                canv.drawImage(
                    marker_im,
                    corner_coord_x,
                    corner_coord_y,
                    width=params.square_length_pt,
                    height=params.square_length_pt,
                )
        LOGGER.info("Drawing ArUco tags...")
        for marker_idx, marker_im in enumerate(ar_marker_ims):
            tag_width_px = ar_marker.shape[1]
            tag_height_px = ar_marker.shape[0]
            assert tag_width_px == tag_height_px, "Non-square markers not supported!"
            canv.drawImage(
                marker_im,
                params.tag_start_x_pt[marker_idx],
                params.tag_start_y_pt[marker_idx],
                width=params.tag_square_length_pt,
                height=params.tag_square_length_pt,
            )
        canv.showPage()
        canv.save()
        LOGGER.info("Done.")
        import time

        time.sleep(2)


if __name__ == "__main__":
    cli()
