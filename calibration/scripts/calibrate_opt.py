import torch
import cv2
import numpy as np
import logging
from os import path
from calibration.persistence import NumpyEncoder
import hydra
import imageio
import json
import sys
import tqdm
import flow_vis
from torchmin import least_squares, minimize

from ..persistence import NumpyEncoder
from ..config import Config
from ..keypoint_detection import process_frame
from ..camera import Camera
from ..networks import LensNet
from ..util import batched_func


LOGGER = logging.getLogger(__name__)
CONF_FP: str = path.join("..", "..", "conf")


@hydra.main(config_path=CONF_FP, config_name="calibration_config")
def cli(cfg: Config):
    """
    Experimental calibration using `torchmin` for fancier optimizers than SGD.

    Does not work at the moment because it gets stuck in local minima.
    """
    LOGGER.info("Loading detection data...")
    storage_fp = path.abspath(path.join(CONF_FP, "..", "..", "data", "live"))
    with open(path.join(storage_fp, "points.json"), "r") as inf:
        cam_info = json.load(inf)
    n_points = 0
    n_frames = 0
    board_coords_val = []
    frame_coords_val = []
    subs_fac = 100
    LOGGER.info("Converting representation for initialization...")
    for idx, (im_board_coords, im_frame_coords) in enumerate(
        zip(cam_info["board_coordinates_xyz"], cam_info["frame_coordinates_xy"])
    ):
        if idx % subs_fac == 0:
            board_coords_val.append(np.array(im_board_coords, dtype=np.float32))
            frame_coords_val.append(np.array(im_frame_coords, dtype=np.float32))
            n_points += frame_coords_val[-1].shape[0]
            n_frames += 1
    LOGGER.info(f"Using {n_points} points from {n_frames} frames.")
    LOGGER.info("Finding initialization...")
    (
        retval,
        camera_matrix,
        dist_coeffs,
        r_vecs_obj_to_cam,
        t_vecs_obj_to_cam,
        _,  # stdDeviationsIntrinsics,
        _,  # stdDeviationsExtrinsics,
        _,  # perViewErrors,
    ) = cv2.calibrateCameraExtended(
        board_coords_val,
        frame_coords_val,
        cam_info["resolution_wh"],
        None,
        None,
        flags=cv2.CALIB_RATIONAL_MODEL,
    )

    # TODO: make robust to outliers.
    # TODO: test https://github.com/rfeinman/pytorch-minimize
    LOGGER.info(f"Overall RMS reprojection error after initialization: {retval}.")
    LOGGER.info("Initializing lens net...")
    RTs_val = []
    dev = torch.device("cuda")
    lens_net = LensNet().to(dev)
    # lens_net = None
    K = torch.from_numpy(camera_matrix).to(torch.float32).to(dev).requires_grad_()
    cams_val = []
    residuals_val = []
    LOGGER.info("Preparing validation data...")
    for cam_idx, (r_vecs, t_vecs) in enumerate(
        zip(r_vecs_obj_to_cam, t_vecs_obj_to_cam)
    ):
        RT = (
            torch.from_numpy(
                np.hstack((cv2.Rodrigues(r_vecs)[0], t_vecs)),
            )
            .to(torch.float32)
            .to(dev)
        ).requires_grad_()
        RTs_val.append(RT)
        cams_val.append(Camera(cam_info["resolution_wh"], K, lens_net, RT))
        proj_opencv = cv2.projectPoints(
            board_coords_val[cam_idx],
            r_vecs_obj_to_cam[cam_idx],
            t_vecs_obj_to_cam[cam_idx],
            camera_matrix,
            np.zeros_like(dist_coeffs),
        )[0][:, 0, :]
        residuals_val.append(
            np.square(np.linalg.norm(frame_coords_val[cam_idx] - proj_opencv, axis=1))
        )
        # proj_calib = cams[-1].project_points(
        #     torch.from_numpy(board_coords[cam_idx]).to(torch.float32).to(dev)
        # )
        # proj_calib = proj_calib.detach().cpu().numpy()
        # import pdb
        # pdb.set_trace()
    RMSE_opencv_val = np.sqrt(np.mean(np.concatenate(residuals_val, axis=0)))
    LOGGER.info(f"RMSE OpenCV: {RMSE_opencv_val}")
    LOGGER.info("Assembling training set...")
    RTs_train = []
    cams_train = []
    board_coords_train = []
    frame_coords_train = []
    n_points_train = 0
    n_frames_train = 0
    for idx, (im_board_coords, im_frame_coords) in enumerate(
        zip(cam_info["board_coordinates_xyz"], cam_info["frame_coordinates_xy"])
    ):
        if idx % subs_fac == 0:
            board_coords_train.append(np.array(im_board_coords, dtype=np.float32))
            frame_coords_train.append(np.array(im_frame_coords, dtype=np.float32))
            n_points_train += frame_coords_train[-1].shape[0]
            n_frames_train += 1
            _, r_vecs, t_vecs = cv2.solvePnP(
                board_coords_train[-1],
                frame_coords_train[-1],
                camera_matrix,
                dist_coeffs,
            )
            RT = (
                torch.from_numpy(
                    np.hstack((cv2.Rodrigues(r_vecs)[0], t_vecs)),
                )
                .to(torch.float32)
                .to(dev)
            ).requires_grad_()
            RTs_train.append(RT)
            cams_train.append(Camera(cam_info["resolution_wh"], K, lens_net, RT))
            board_coords_train[-1] = (
                torch.from_numpy(board_coords_train[-1]).to(torch.float32).to(dev)
            )
            frame_coords_train[-1] = (
                torch.from_numpy(frame_coords_train[-1]).to(torch.float32).to(dev)
            )
    LOGGER.info(
        f"Using {n_points_train} points for training from {n_frames_train} frames."
    )
    LOGGER.info(f"Lens has {sum(p.numel() for p in lens_net.parameters())} parameters.")
    LOGGER.info("Setting up optimizer...")
    # Find x0.
    x0 = [K] + list(lens_net.parameters()) + RTs_train
    x0 = [p.flatten() for p in x0]
    x0 = torch.cat(x0, dim=0)

    def update_cams(params):
        # Update all the parameters.
        for cam_idx, cam in enumerate(cams_train):
            starting_point = 0
            # Update K.
            cam.K = params[starting_point : starting_point + cam.K.numel()].reshape(
                cam.K.shape
            )
            starting_point += cam.K.numel()
            # Update lens net.
            for param in cam.lensnet.parameters():
                if cam_idx == 0:
                    param = params[
                        starting_point : starting_point + param.numel()
                    ].reshape(param.shape)
                starting_point += param.numel()
            starting_point += cam_idx * cam.RT.numel()
            cam.RT = params[starting_point : starting_point + cam.RT.numel()].reshape(
                cam.RT.shape
            )

    # Create closure producing the residuals.
    def forw_closure(params):
        update_cams(params)
        # Run all points through the model.
        loss_vals = []
        for cam_idx, (cam, board_coord_mat, frame_coord_mat) in tqdm.tqdm(
            enumerate(zip(cams_train, board_coords_train, frame_coords_train)),
            total=len(cams_train),
        ):
            projected = cam.project_points(board_coord_mat)
            loss_vals.append(
                torch.linalg.norm(projected - frame_coord_mat, dim=1, ord=2)
            )
        return torch.cat(loss_vals, dim=0).sum()

    res = minimize(
        forw_closure,
        x0,
        "l-bfgs",
        max_iter=1000,
        disp=2,
        options={"disp": 2, "gtol": 0, "xtol": 0},
    )
    # res = least_squares(
    #     forw_closure,
    #     x0,
    #     method="trf",
    #     # loss="huber",
    #     verbose=2,
    #     max_nfev=10,
    # )
    update_cams(res["x"])
    loss_vals = []
    for cam_idx, (cam, board_coord_mat, frame_coord_mat) in enumerate(
        zip(cams_val, board_coords_val, frame_coords_val)
    ):
        board_coord_mat = torch.from_numpy(board_coord_mat).to(torch.float32).to(dev)
        frame_coord_mat = torch.from_numpy(frame_coord_mat).to(torch.float32).to(dev)
        projected = cam.project_points(board_coord_mat)
        loss = torch.square(
            torch.linalg.norm(projected - frame_coord_mat, dim=1)
        )  # RMSE
        loss_vals.append(loss.detach().clone())
    LOGGER.info(f"RMSE (val): {torch.sqrt(torch.cat(loss_vals, dim=0).mean())}")
    vis = vis_lens(cam)
    imageio.imwrite("../../../opt.jpg", vis)
    import pdb

    pdb.set_trace()


def colorize(uv_im, max_mag=None):
    hsv = np.zeros((uv_im.shape[0], uv_im.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(uv_im[..., 0], uv_im[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    print(mag.max())
    if max_mag is None:
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    else:
        mag = np.clip(mag, 0.0, max_mag)
        mag = mag / max_mag * 255.0
        hsv[..., 2] = mag.astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def vis_lens(camera: Camera):
    i, j = np.meshgrid(
        np.linspace(0, camera.resolution_w_h[0] - 1, camera.resolution_w_h[0]),
        np.linspace(0, camera.resolution_w_h[1] - 1, camera.resolution_w_h[1]),
        indexing="ij",
    )
    i = i.T
    j = j.T
    P_sensor = (
        torch.from_numpy(np.stack((i, j), axis=-1))
        .to(torch.float32)
        .to(camera.K.device)
    )
    with torch.no_grad():
        camera_directions_w_lens = batched_func(
            camera.get_rays_view, P_sensor.reshape((-1, 2)), 10000
        )
        camera_directions_w_lens = camera_directions_w_lens.reshape(
            (P_sensor.shape[0], P_sensor.shape[1], 3)
        )[:, :, :2]
        # This camera does not need a lens and RT does not matter either since we're
        # working in view space only.
        camera_no_lens = Camera(camera.resolution_w_h, camera.K)
        camera_directions_wo_lens = batched_func(
            camera_no_lens.get_rays_view, P_sensor.reshape((-1, 2)), 10000
        ).reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
        direction_diff = camera_directions_w_lens - camera_directions_wo_lens
        flow_color = colorize(
            direction_diff.detach().cpu().numpy(),
            max_mag=0.1,
        )
        # flow_color = flow_vis.flow_to_color(
        #     direction_diff.detach().cpu().numpy(), clip_flow=None, convert_to_bgr=False
        # )
    return flow_color


if __name__ == "__main__":
    cli()
