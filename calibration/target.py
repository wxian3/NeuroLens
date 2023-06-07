import math
import logging
from collections import namedtuple

from reportlab.lib.units import cm

from .config import Config

target_params = namedtuple(
    "TargetParams",
    "n_squares_x, n_squares_y, pattern_start_x_pt, pattern_start_y_pt, "
    "square_length_pt, tag_start_x_pt, tag_start_y_pt, tag_x_grid, tag_y_grid, "
    "tag_square_length_pt, x_part_length_grid, y_part_length_grid",
)
LOGGER = logging.getLogger(__name__)


def calculate_parameters(cfg: Config):
    width_pt, height_pt = cfg.target.page_size_pt
    margin_size_pt = cfg.target.margin_in_cm * cm
    start_x_pt = margin_size_pt
    end_x_pt = width_pt - margin_size_pt
    start_y_pt = margin_size_pt
    end_y_pt = height_pt - margin_size_pt
    print_area_width_pt = abs(end_x_pt - start_x_pt)
    print_area_height_pt = abs(end_y_pt - start_y_pt)
    LOGGER.info(
        f"Print area size: {print_area_width_pt}pt X {print_area_height_pt}pt (width X height)."
    )
    approx_square_length_pt = cfg.target.approx_square_length_in_cm * cm
    squares_length_width_rounded_pt = print_area_width_pt / round(
        print_area_width_pt / approx_square_length_pt
    )
    squares_length_height_rounded_pt = print_area_height_pt / round(
        print_area_height_pt / approx_square_length_pt
    )
    square_length_pt = min(
        squares_length_width_rounded_pt, squares_length_height_rounded_pt
    )
    n_squares_x = int(math.floor(print_area_width_pt / square_length_pt))
    n_squares_y = int(math.floor(print_area_height_pt / square_length_pt))
    unused_x_pt = print_area_width_pt - n_squares_x * square_length_pt
    pattern_start_x_pt = start_x_pt + 0.5 * unused_x_pt
    unused_y_pt = print_area_height_pt - n_squares_y * square_length_pt
    pattern_start_y_pt = start_y_pt + 0.5 * unused_y_pt  # - square_length_pt
    assert cfg.target.aruco_length_in_squares % 2 == 0
    tag_start_x_pt = []
    tag_start_y_pt = []
    tag_x_grid = []
    tag_y_grid = []
    x_part_length_grid = int(
        round((n_squares_x - 2 * cfg.target.aruco_length_in_squares) // 4)
    )
    y_part_length_grid = int(
        round((n_squares_y - 2 * cfg.target.aruco_length_in_squares) / 4)
    )
    for id_x in range(2):
        for id_y in range(2):
            tag_x_grid.append(
                x_part_length_grid
                + (cfg.target.aruco_length_in_squares + 2 * x_part_length_grid) * id_x
            )
            tag_start_x_pt.append(
                pattern_start_x_pt + tag_x_grid[-1] * square_length_pt
            )

            tag_y_grid.append(
                y_part_length_grid
                + (cfg.target.aruco_length_in_squares + 2 * y_part_length_grid) * id_y
            )
            tag_start_y_pt.append(
                pattern_start_y_pt + tag_y_grid[-1] * square_length_pt
            )
    tag_square_length_pt = cfg.target.aruco_length_in_squares * square_length_pt
    return target_params(
        n_squares_x,
        n_squares_y,
        pattern_start_x_pt,
        pattern_start_y_pt,
        square_length_pt,
        tag_start_x_pt,
        tag_start_y_pt,
        tag_x_grid,
        tag_y_grid,
        tag_square_length_pt,
        x_part_length_grid,
        y_part_length_grid,
    )
