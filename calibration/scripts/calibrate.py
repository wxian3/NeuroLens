import torch
import cv2
import numpy as np
import logging
from os import path
import hydra
import imageio
import json
import tqdm

from ..config import Config
from ..camera import Camera
from ..networks import LensNet
from ..util import batched_func


LOGGER = logging.getLogger(__name__)
CONF_FP: str = path.join("..", "..", "conf")


@hydra.main(config_path=CONF_FP, config_name="calibration_config")
def cli(cfg: Config):
    """
    Calibrate a lens using previously acquired data in `data/live`.

    Uses gradient-descent based optimization with RLS to be robust w.r.t.
    outliers to calibrate the lens. Does an automatic comparison with the
    OpenCV model.
    """
    LOGGER.info("Loading detection data...")
    storage_fp = path.abspath(path.join(CONF_FP, "..", "..", "data", "live"))
    with open(path.join(storage_fp, "points.json"), "r") as inf:
        cam_info = json.load(inf)
    n_points = 0
    n_frames = 0
    board_coords_val = []
    frame_coords_val = []
    subs_fac = cfg.calibration.subs_fac
    LOGGER.info(
        "Converting representation for initialization (subsampling every %dth frame)...",
        subs_fac,
    )
    for idx, (im_board_coords, im_frame_coords) in enumerate(
        zip(cam_info["board_coordinates_xyz"], cam_info["frame_coordinates_xy"])
    ):
        if idx % subs_fac == 0 and len(im_board_coords) >= 6:
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

    LOGGER.info(f"Overall RMS reprojection error after initialization: {retval}.")
    # for image_idx in range(len(board_coords)):
    #     image = imageio.imread(
    #         path.join(storage_fp, "%05d.png" % (image_idx * subs_fac))
    #     )
    #     cv2.imshow("image", image[:, :, ::-1])
    #     resize_fac = 3
    #     image = cv2.resize(
    #         image, None, fx=resize_fac, fy=resize_fac, interpolation=cv2.INTER_LINEAR
    #     )
    #     for frame_coord, board_coord in zip(
    #         frame_coords[image_idx], board_coords[image_idx]
    #     ):
    #         image[
    #             int(frame_coord[1] * resize_fac)
    #             - resize_fac : int(frame_coord[1] * resize_fac)
    #             + resize_fac,
    #             int(frame_coord[0] * resize_fac)
    #             - resize_fac : int(frame_coord[0] * resize_fac)
    #             + resize_fac,
    #             1,
    #         ] = 255
    #         text = str("%d, %d" % (int(board_coord[0]), int(board_coord[1])))
    #         text_width, text_height = cv2.getTextSize(
    #             text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1
    #         )[0]
    #         cv2.putText(
    #             image,
    #             text,
    #             (
    #                 int(frame_coord[0] * resize_fac),
    #                 int(frame_coord[1] * resize_fac) - text_height,
    #             ),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.3,
    #             (0, 255, 0),
    #             1,
    #         )
    #     cv2.imshow("detections", image[:, :, ::-1])
    #     proj = cv2.projectPoints(
    #         board_coords[image_idx],
    #         r_vecs_obj_to_cam[image_idx],
    #         t_vecs_obj_to_cam[image_idx],
    #         camera_matrix,
    #         dist_coeffs,
    #     )[0][:, 0, :]
    #     for frame_coord in proj:
    #         image[
    #             int(frame_coord[1] * resize_fac)
    #             - resize_fac : int(frame_coord[1] * resize_fac)
    #             + resize_fac,
    #             int(frame_coord[0] * resize_fac)
    #             - resize_fac : int(frame_coord[0] * resize_fac)
    #             + resize_fac,
    #             0,
    #         ] = 255
    #     cv2.imshow("projected", image[:, :, ::-1])
    #     cv2.waitKey(0)

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
        if idx % subs_fac != 0 and len(im_board_coords) >= 6:
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
    LOGGER.info("Setting up optimizer...")
    optimizer_train = torch.optim.Adam(
        [
            K,
        ]
        + list(lens_net.parameters())
        + RTs_train,
        lr=1e-4,
    )
    optimizer_val_RT = torch.optim.Adam(
        RTs_val,
        lr=1e-4,
    )
    LOGGER.info(f"Lens has {sum(p.numel() for p in lens_net.parameters())} parameters.")
    LOGGER.info("Optimizing...")
    rnge = tqdm.trange(300)
    weights = {}
    weights_normalizer = 1.0
    p_ord = 1
    weights_updated_every = 10
    for epoch_idx in rnge:
        loss_vals = []
        loss_vals_train = []
        for cam_idx, (cam, board_coord_mat, frame_coord_mat) in tqdm.tqdm(
            enumerate(zip(cams_train, board_coords_train, frame_coords_train)),
            total=len(cams_train),
        ):
            optimizer_train.zero_grad()
            projected = cam.project_points(board_coord_mat)
            if cam_idx not in weights:
                weights[cam_idx] = torch.ones(
                    (projected.shape[0], 1), dtype=torch.float32, device=dev
                )
            else:
                if epoch_idx > 0 and epoch_idx % weights_updated_every == 0:
                    norm_vals = torch.linalg.norm(
                        projected - frame_coord_mat, dim=1, ord=abs(p_ord - 2)
                    )
                    if p_ord - 2 < 0:
                        weights[cam_idx] = 1.0 / torch.maximum(
                            torch.tensor(1e-5, dtype=torch.float32, device=dev),
                            norm_vals,
                        )
                    else:
                        weights[cam_idx] = norm_vals
            weights_to_apply = (weights_normalizer * weights[cam_idx]).detach()
            loss = (
                weights_to_apply
                * torch.linalg.norm(projected - frame_coord_mat, dim=1, ord=2)
            ).sum()
            loss_vals_train.append(loss.item())
            loss.backward()
            optimizer_train.step()
        # Recalculate weights normalizer.
        if epoch_idx > weights_updated_every:
            weights_normalizer = (
                n_points_train
                / torch.sum(torch.cat(list(weights.values()), dim=0))
                * 100.0
            )
        print("weights_normalizer", weights_normalizer)
        # Evaluate.
        for cam_idx, (cam, board_coord_mat, frame_coord_mat) in enumerate(
            zip(cams_val, board_coords_val, frame_coords_val)
        ):
            optimizer_val_RT.zero_grad()
            board_coord_mat = (
                torch.from_numpy(board_coord_mat).to(torch.float32).to(dev)
            )
            frame_coord_mat = (
                torch.from_numpy(frame_coord_mat).to(torch.float32).to(dev)
            )
            projected = cam.project_points(board_coord_mat)
            loss = torch.square(
                torch.linalg.norm(projected - frame_coord_mat, dim=1)
            )  # RMSE
            loss_vals.append(loss.detach().clone())
            weights_to_apply = 1.0 / torch.maximum(
                torch.tensor(1e-5, dtype=torch.float32, device=dev),
                torch.linalg.norm(projected - frame_coord_mat, dim=1, ord=1),
            )
            loss = (
                weights_to_apply.detach()
                * torch.linalg.norm(projected - frame_coord_mat, dim=1, ord=2)
            ).sum()
            loss.backward()
            optimizer_val_RT.step()
        if epoch_idx % 10 == 0:
            vis = vis_lens(cam)
            imageio.imwrite("../../../%d.jpg" % (epoch_idx), vis)
        rnge.set_description(
            f"Loss: {np.mean(loss_vals_train)}, RMSE (val): {torch.sqrt(torch.cat(loss_vals, dim=0).mean())}"
        )


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
