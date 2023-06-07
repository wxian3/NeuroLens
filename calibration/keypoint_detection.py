import logging
from os import path
import cv2
import math
import numpy as np
import torch

from .target import calculate_parameters
from .model import Model
from .config import Config, get_latest_checkpoint, ARUCO_DICT


LOGGER = logging.getLogger(__name__)
MODEL = None
ARUCO_PS = None
TARGET_PS = None


def process_frame(frame, cfg: Config):
    """
    Process a frame and run the keypoint detector over it.

    First locates the ArUco markers. Then infers a rough position
    for all keypoints. Runs the keypoint detector on all of these
    areas and returns their centers, the validity, coordinates
    and a visualization.

    Returns centers_found, centers_valid, centers_coords, vis_frame.
    """
    global MODEL, ARUCO_PS, TARGET_PS
    if MODEL is None:
        LOGGER.info(f"Loading model from experiment `{cfg.target.exp_name}`...")
        latest_checkpoint = get_latest_checkpoint(cfg.target.exp_name, cfg)
        MODEL = Model.load_from_checkpoint(latest_checkpoint).cuda()
        LOGGER.info("Loading successful.")
    if ARUCO_PS is None:
        LOGGER.info(
            f"Setting up ArUco parameters for marker `{cfg.target.aruco_id}`..."
        )
        ARUCO_PS = (
            cv2.aruco.Dictionary_get(ARUCO_DICT[cfg.target.aruco_id]),
            cv2.aruco.DetectorParameters_create(),
        )
        ARUCO_PS[1].cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        LOGGER.info("Setup successful.")
    if TARGET_PS is None:
        LOGGER.info("Calculating target parameters...")
        TARGET_PS = calculate_parameters(cfg)
        LOGGER.info("Done.")
    centers_found = []
    centers_valid = []
    centers_coords = []
    # TODO: resize frame before detection?
    (corners, ids, _) = cv2.aruco.detectMarkers(
        frame, ARUCO_PS[0], parameters=ARUCO_PS[1]
    )
    vis_frame = frame.copy()
    corners_marker = np.array(
        [
            [0.0, 0.0],  # tl
            [cfg.model.marker.working_size, 0.0],  # tr
            [cfg.model.marker.working_size, cfg.model.marker.working_size],  # br
            [0.0, cfg.model.marker.working_size],  # bl
        ],
        dtype=np.float32,
    )
    if len(corners) > 0:
        if cfg.dbg:
            cv2.namedWindow("[dbg] markers")
            dbg_frame = frame.copy()
            cv2.aruco.drawDetectedMarkers(dbg_frame, corners)
            cv2.imshow("[dbg] markers", dbg_frame)
            cv2.waitKey(0)
            cv2.destroyWindow("[dbg] markers")
        cv2.aruco.drawDetectedMarkers(vis_frame, corners, ids)
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            if markerID not in [23, 24, 25, 26]:
                continue
            corners = markerCorner.reshape((4, 2))
            # The marker is printed upside-down (that's why tl-tr-br-bl is swapped
            # to bl-br-tr-tl).
            corners_grid = np.array(
                [
                    [
                        TARGET_PS.tag_x_grid[markerID - 23],
                        TARGET_PS.tag_y_grid[markerID - 23]
                        + cfg.target.aruco_length_in_squares,
                    ],  # bl
                    [
                        TARGET_PS.tag_x_grid[markerID - 23]
                        + cfg.target.aruco_length_in_squares,
                        TARGET_PS.tag_y_grid[markerID - 23]
                        + cfg.target.aruco_length_in_squares,
                    ],  # br
                    [
                        TARGET_PS.tag_x_grid[markerID - 23]
                        + cfg.target.aruco_length_in_squares,
                        TARGET_PS.tag_y_grid[markerID - 23],
                    ],  # tr
                    [
                        TARGET_PS.tag_x_grid[markerID - 23],
                        TARGET_PS.tag_y_grid[markerID - 23],
                    ],  # tl
                ],
                dtype=np.float32,
            )
            LOGGER.debug(f"ArUco marker grid position: {corners_grid}.")
            # Define a grid coordinate system over the target as:
            # +------+------+------+------+
            # |      |      |      |      |
            # +------X------+------X------+
            # |      |      |      |      |
            # +------+------+------+------+
            # |      |      |      |      |
            # +------X------+------X------+
            # |      |      |      |      |
            # +------+------+------+------+
            # with the four X's marking the corners of the ArUco
            # marker.
            hom_grid_frame, _ = cv2.findHomography(corners_grid, corners)
            # debugging:
            # corners_grid_hom = np.array(
            #     [
            #         [TARGET_PS.tag_x_grid, TARGET_PS.tag_y_grid, 1],  # tl
            #         [TARGET_PS.tag_x_grid + 1, TARGET_PS.tag_y_grid, 1],  # tr
            #         [TARGET_PS.tag_x_grid + 1, TARGET_PS.tag_y_grid + 1, 1],  # br
            #         [TARGET_PS.tag_x_grid, TARGET_PS.tag_y_grid + 1, 1],  # bl
            #     ],
            #     dtype=np.float32,
            # )
            # corners_hom = (hom_grid_frame @ corners_grid_hom.T).T
            # corners_calc = corners_hom[:, :2] / corners_hom[:, 2:3]
            with BatchedKeypointDetector(
                centers_found, centers_valid, centers_coords, cfg
            ) as bkd:
                for x in range(
                    max(
                        0,
                        TARGET_PS.tag_x_grid[markerID - 23]
                        - TARGET_PS.x_part_length_grid,
                    ),
                    min(
                        TARGET_PS.n_squares_x,
                        TARGET_PS.tag_x_grid[markerID - 23]
                        + cfg.target.aruco_length_in_squares
                        + TARGET_PS.x_part_length_grid,
                    ),
                ):
                    for y in range(
                        max(
                            0,
                            TARGET_PS.tag_y_grid[markerID - 23]
                            - TARGET_PS.y_part_length_grid,
                        ),
                        min(
                            TARGET_PS.n_squares_y,
                            TARGET_PS.tag_y_grid[markerID - 23]
                            + cfg.target.aruco_length_in_squares
                            + TARGET_PS.y_part_length_grid,
                        ),
                    ):
                        if (
                            x >= TARGET_PS.tag_x_grid[markerID - 23]
                            and x
                            < TARGET_PS.tag_x_grid[markerID - 23]
                            + cfg.target.aruco_length_in_squares
                            and y >= TARGET_PS.tag_y_grid[markerID - 23]
                            and y
                            < TARGET_PS.tag_y_grid[markerID - 23]
                            + cfg.target.aruco_length_in_squares
                        ):
                            # This is the ArUco marker section.
                            continue
                        corners_grid_hom = np.array(
                            [
                                [x, y, 1.0],  # tl
                                [x + 1, y, 1.0],  # tr
                                [x + 1, y + 1, 1.0],  # br
                                [x, y + 1, 1.0],  # bl
                            ],
                            dtype=np.float32,
                        )
                        corners_frame_hom = (hom_grid_frame @ corners_grid_hom.T).T
                        corners_frame = (
                            corners_frame_hom[:, :2] / corners_frame_hom[:, 2:3]
                        )
                        valid = True
                        for corner_x, corner_y in corners_frame:
                            if (
                                corner_x < 0
                                or corner_y < 0
                                or corner_x > frame.shape[1]
                                or corner_y > frame.shape[0]
                            ):
                                valid = False
                                break
                        line_thickness = 2
                        cv2.line(
                            vis_frame,
                            tuple(corners_frame[0].astype(np.int32)),
                            tuple(corners_frame[1].astype(np.int32)),
                            (0, 255, 0),
                            thickness=line_thickness,
                        )
                        cv2.line(
                            vis_frame,
                            tuple(corners_frame[1].astype(np.int32)),
                            tuple(corners_frame[2].astype(np.int32)),
                            (0, 255, 0),
                            thickness=line_thickness,
                        )
                        cv2.line(
                            vis_frame,
                            tuple(corners_frame[2].astype(np.int32)),
                            tuple(corners_frame[3].astype(np.int32)),
                            (0, 255, 0),
                            thickness=line_thickness,
                        )
                        cv2.line(
                            vis_frame,
                            tuple(corners_frame[3].astype(np.int32)),
                            tuple(corners_frame[0].astype(np.int32)),
                            (0, 255, 0),
                            thickness=line_thickness,
                        )
                        if not valid:
                            LOGGER.debug(
                                f"Marker at position {x}, {y} (x, y) not fully visible: {corners_frame}."
                            )
                            continue
                        hom_boxframe_marker, _ = cv2.findHomography(
                            corners_frame, corners_marker
                        )
                        # We're not using `np.linalg.inv(hom_boxframe_marker)` here,
                        # which would be an option but is less numerically stable and
                        # accurate than optimizing for the new homography.
                        hom_marker_boxframe, _ = cv2.findHomography(
                            corners_marker, corners_frame
                        )
                        flat_marker = cv2.warpPerspective(
                            frame,
                            hom_boxframe_marker,
                            (
                                cfg.model.marker.working_size,
                                cfg.model.marker.working_size,
                            ),
                            flags=cv2.INTER_NEAREST,
                        )
                        flat_marker_pt = (
                            torch.from_numpy(flat_marker)
                            .cuda()
                            .permute(2, 0, 1)[None, ...]
                            / 255.0
                        )
                        bkd.process_marker(flat_marker_pt, hom_marker_boxframe, x, y)
        for center_coord, center_valid in zip(centers_found, centers_valid):
            clr = (255, 0, 0) if center_valid else (0, 0, 255)
            vis_frame[
                int(center_coord[1]) - 3 : int(center_coord[1]) + 3,
                int(center_coord[0]) - 3 : int(center_coord[0]) + 3,
                :,
            ] = clr
        if cfg.dbg:
            # Plot all detections in the frame.
            frame_dbg = frame.copy()
            for center_coord, center_valid in zip(centers_found, centers_valid):
                clr = (255, 0, 0) if center_valid else (0, 0, 255)
                frame_dbg[
                    int(center_coord[1]) - 3 : int(center_coord[1]) + 3,
                    int(center_coord[0]) - 3 : int(center_coord[0]) + 3,
                    :,
                ] = clr
            cv2.namedWindow("[dbg] frame with centers")
            cv2.imshow("[dbg] frame with centers", frame_dbg)
            cv2.waitKey(0)
            cv2.destroyWindow("[dbg] frame with centers")
    return centers_found, centers_valid, centers_coords, vis_frame


class BatchedKeypointDetector:
    def __init__(self, center_list, valid_list, centers_board_coords, cfg):
        self._markers_flat = []
        self._homographies = []
        self._coordinates = []
        self.center_list = center_list
        self.valid_list = valid_list
        self.centers_board_coords = centers_board_coords
        self.cfg = cfg

    def process_marker(self, marker_pt, hom_marker_boxframe, x, y):
        self._markers_flat.append(marker_pt)
        self._homographies.append(hom_marker_boxframe)
        self._coordinates.append([x, y, 0])
        if len(self._markers_flat) >= MODEL.batch_size:
            self.run_forward()

    def run_forward(self):
        marker_batch = torch.cat(self._markers_flat, dim=0)
        model_centers, model_vars = MODEL.forward(detection=marker_batch)
        center_coords = model_centers.cpu().detach().numpy()
        variances = model_vars.cpu().detach().numpy()
        LOGGER.debug(f"Center coordinates: {center_coords}.")
        # if cfg.dbg and False:
        #     flat_marker_dbg = flat_marker.copy()
        #     flat_marker_dbg[
        #         int(center_coords[0]) - 3 : int(center_coords[0]) + 3,
        #         int(center_coords[1]) - 3 : int(center_coords[1]) + 3,
        #         :,
        #     ] = (255, 0, 0)
        #     LOGGER.info(f"[dbg] Variance: {variance}.")
        #     cv2.namedWindow("[dbg] normalized marker")
        #     cv2.imshow("[dbg] normalized marker", flat_marker_dbg)
        #     cv2.waitKey(0)
        #     cv2.destroyWindow("[dbg] normalized marker")
        for center_coord, variance, hom_marker_boxframe, board_coord in zip(
            center_coords, variances, self._homographies, self._coordinates
        ):
            # Check for reliability.
            reliable = not (
                center_coord[0] < 0.0
                or center_coord[0] >= self.cfg.model.marker.working_size
                or center_coord[1] < 0.0
                or center_coord[1] > self.cfg.model.marker.working_size
                or math.sqrt(variance) > self.cfg.model.marker.std_thresh
            )
            center_coords_hom = np.array(
                [[center_coord[1], center_coord[0], 1.0]], dtype=np.float32
            )
            center_coords_frame_hom = (hom_marker_boxframe @ center_coords_hom.T).T
            center_coords_frame = (
                center_coords_frame_hom[0, :2] / center_coords_frame_hom[0, 2:3]
            )
            self.center_list.append(center_coords_frame)
            self.valid_list.append(reliable)
            self.centers_board_coords.append(board_coord)
        self._homographies = []
        self._markers_flat = []
        self._coordinates = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if len(self._markers_flat) > 0:
            self.run_forward()
