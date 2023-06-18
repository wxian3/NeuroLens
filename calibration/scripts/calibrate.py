import json
import logging
from os import path
import cv2
import hydra
import imageio
import numpy as np
import torch
import torchvision.transforms as transforms
import tqdm
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from ..camera import Camera
from ..config import Config
from ..networks import iResNet, LensNet 
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
    storage_fp = path.join("..", "..", "data/target")

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
        tuple(cam_info["resolution_wh"]),
        None,
        None,
        flags=cv2.CALIB_RATIONAL_MODEL,
    )

    LOGGER.info(f"Overall RMS reprojection error after initialization: {retval}.")

    LOGGER.info("Initializing lens net...")
    RTs_val = []
    dev = torch.device("cuda")
    lens_net = iResNet().to(dev)
    iResNet().test_inverse()

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
            dist_coeffs,
        )[0][:, 0, :]
        residuals_val.append(
            np.square(np.linalg.norm(frame_coords_val[cam_idx] - proj_opencv, axis=1))
        )

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
    scale = torch.nn.Parameter(torch.tensor(1.305)).requires_grad_()
    optimizer_train = torch.optim.Adam(
        [
            K,
        ]
        + list(lens_net.parameters())
        + RTs_train
        + [scale],
        lr=1e-4,
    )
    optimizer_val_RT = torch.optim.Adam(
        RTs_val,
        lr=1e-4,
    )

    if is_eval:
        PATH = log_dir + "/lensnet_latest.pt"
        checkpoint = torch.load(PATH)
        lens_net.load_state_dict(checkpoint["model_state_dict"])
        optimizer_train.load_state_dict(checkpoint["optimizer_state_dict"])
        RTs_train = checkpoint["RTs_train"]
        RTs_val = checkpoint["RTs_train"]
        K = checkpoint["K"]

        loss_vals = []
        rows, cols = 3, 4
        gridspec_kw = {"wspace": 0.0, "hspace": 0.0}
        fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(12, 9))
        bleed = 0
        fig.subplots_adjust(
            left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed)
        )
        for cam_idx, (cam, board_coord_mat, frame_coord_mat, ax) in enumerate(
            zip(cams_val, board_coords_val, frame_coords_val, axarr.ravel())
        ):
            board_coord_mat = (
                torch.from_numpy(board_coord_mat).to(torch.float32).to(dev)
            )
            frame_coord_mat = (
                torch.from_numpy(frame_coord_mat).to(torch.float32).to(dev)
            )
            projected = cam.project_points(board_coord_mat)
            val_error = torch.square(
                torch.linalg.norm(projected - frame_coord_mat, dim=1)
            )  # RMSE
            loss_vals.append(val_error.detach().clone())
            # visualize keypoints
            vis, camera_directions_w_lens = vis_lens(cam)
            # ax.imshow(vis[..., :3])
            ax.scatter(
                projected.detach().cpu().numpy()[:, 0],
                projected.detach().cpu().numpy()[:, 1],
                marker="o",
                s=5,
            )
            ax.scatter(
                frame_coord_mat.detach().cpu().numpy()[:, 0],
                frame_coord_mat.detach().cpu().numpy()[:, 1],
                marker="o",
                s=5,
            )
            ax.set_xlim((0, 512))
            ax.set_ylim((0, 512))
            ax.set_axis_off()

        fig.savefig(log_dir + "/vis_eval.png")
        plt.close(fig)

        rows, cols = 1, 4
        gridspec_kw = {"wspace": 0.0, "hspace": 0.0}
        fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(12, 3))
        bleed = 0
        fig.subplots_adjust(
            left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed)
        )

        vis, camera_directions_w_lens = vis_lens(cams_val[0])
        target_board_ABC = (
            imageio.imread(path.join(storage_fp, "target_rainbow_ABC.png"))[..., :3]
            / 255.0
        )
        target_board_path = path.join(storage_fp, "target_rainbow.png")
        target_board = imageio.imread(target_board_path)[..., :3]
        transform = transforms.ToTensor()
        image = transform(target_board).unsqueeze(0).to(dev)
        flow = camera_directions_w_lens.unsqueeze(0).to(dev) * 1.305
        output = F.grid_sample(
            image, flow, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        output_img = output[0].permute(1, 2, 0).cpu().numpy()
        axarr[0].imshow(target_board)
        axarr[1].imshow(output_img)
        axarr[2].imshow(target_board_ABC)

        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

        emap = abs(target_board_ABC - output_img)
        axarr[3].imshow(rgb2gray(emap), cmap=plt.get_cmap("gray"))

        for ax in axarr:
            ax.set_axis_off()

        fig.savefig(log_dir + "/img_eval.png")
        plt.close(fig)

        rmse = torch.sqrt(torch.cat(loss_vals, dim=0).mean())
        print("rmse lensnet: ", rmse.item())


    scheduler_type = "super"
    if scheduler_type == "linear":
        step_size = 20000  # 15000
        final_ratio = 0.01  # 0.05
        start_ratio = 0.15
        duration_ratio = 0.6

        def lambda_rule(ep):
            lr_l = 1.0 - min(
                1,
                max(0, ep - start_ratio * step_size)
                / float(duration_ratio * step_size),
            ) * (1 - final_ratio)
            return lr_l

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer_train, lr_lambda=lambda_rule
        )
        LOGGER.info(
            f"LR scheduler has step size {step_size}, final ratio {final_ratio}, start ratio {start_ratio}, duration ratio {duration_ratio}."
        )
    elif scheduler_type == "super":
        step_size = 12000
        max_lr = 1e-4
        pct_start = 0.05
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_train,
            max_lr=max_lr,
            total_steps=step_size,
            pct_start=pct_start,
        )
        LOGGER.info(
            f"LR scheduler has step size {step_size}, max lr {max_lr}, pct_start {pct_start}"
        )

    LOGGER.info(f"Lens has {sum(p.numel() for p in lens_net.parameters())} parameters.")
    writer = SummaryWriter(log_dir=log_dir)
    LOGGER.info(f"Writing logs and tensorboard to `{log_dir}`.")
    LOGGER.info("Optimizing...")
    start_epoch = 0
    if is_eval:
        start_epoch = checkpoint["epoch"]
    rnge = tqdm.trange(start_epoch, 500)


    # load data
    dense_matching = False
    if dense_matching:
        target_board_ABC = (
            imageio.imread(path.join(storage_fp, "target.png"))[..., :3]
            / 255.0
        )
        target_board_path = path.join(storage_fp, "target.png")
        target_board = imageio.imread(target_board_path)[..., :3]
        transform = transforms.ToTensor()
        image = transform(target_board).unsqueeze(0).to(dev)
        image_ABC = transform(target_board_ABC).unsqueeze(0).to(dev)
        l1_loss = torch.nn.L1Loss()

    for epoch_idx in rnge:
        loss_vals = []
        loss_vals_train = []
        loss_train = []

        for _, (cam, board_coord_mat, frame_coord_mat) in tqdm.tqdm(
            enumerate(zip(cams_train, board_coords_train, frame_coords_train)),
            total=len(cams_train),
        ):
            optimizer_train.zero_grad()
            projected = cam.project_points(board_coord_mat)
            loss = torch.linalg.norm(projected - frame_coord_mat, dim=1).mean()
            loss_vals_train.append(loss.item())

            proj_error = torch.square(
                torch.linalg.norm(projected - frame_coord_mat, dim=1)
            )
            loss_train.append(proj_error)

            # dense match loss
            if dense_matching:
                camera_directions_w_lens = cam.project_lens()
                flow = camera_directions_w_lens.unsqueeze(0) * scale
                output = F.grid_sample(
                    image[..., ::2, ::2],
                    flow,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                dense_loss = l1_loss(image_ABC[..., ::2, ::2], output)

                loss += dense_loss * 50

            loss.backward()
            optimizer_train.step()
            scheduler.step()

        avg_loss = np.array(loss_vals_train).mean()
        if dense_matching:
            writer.add_scalar("dense_loss", dense_loss, epoch_idx)
        writer.add_scalar("reproj_loss", avg_loss, epoch_idx)

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
            val_error = torch.square(
                torch.linalg.norm(projected - frame_coord_mat, dim=1)
            )  # RMSE
            loss_vals.append(val_error.detach().clone())
  
            loss = torch.linalg.norm(projected - frame_coord_mat, dim=1).mean()

            # dense match loss
            if dense_matching:
                camera_directions_w_lens = cam.project_lens()
                flow = camera_directions_w_lens.unsqueeze(0) * scale
                output = F.grid_sample(
                    image[..., ::2, ::2],
                    flow,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                dense_loss = l1_loss(image_ABC[..., ::2, ::2], output)

                loss += dense_loss * 50

            loss.backward()
            optimizer_val_RT.step()

            if epoch_idx % 10 == 0 and cam_idx % 100 == 0 and dense_matching:

                def rgb2gray(rgb):
                    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

                rows, cols = 1, 4
                gridspec_kw = {"wspace": 0.0, "hspace": 0.0}
                fig, axarr = plt.subplots(
                    rows, cols, gridspec_kw=gridspec_kw, figsize=(12, 3)
                )
                bleed = 0
                fig.subplots_adjust(
                    left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed)
                )

                axarr[0].imshow(target_board)
                output_img = output[0].permute(1, 2, 0).detach().cpu().numpy()
                axarr[1].imshow(output_img)
                axarr[2].imshow(target_board_ABC[::2, ::2, :])
                emap = abs(target_board_ABC[::2, ::2, :] - output_img)
                axarr[3].imshow(rgb2gray(emap), cmap=plt.get_cmap("gray"))
                for ax in axarr:
                    ax.set_axis_off()
                fig.savefig(log_dir + f"/dense_match_{epoch_idx // 10}.png")

                plt.close(fig)

        if epoch_idx % 10 == 0:
            # visualize keypoints

            rows, cols = 3, 4
            gridspec_kw = {"wspace": 0.0, "hspace": 0.0}
            fig, axarr = plt.subplots(
                rows, cols, gridspec_kw=gridspec_kw, figsize=(12, 9)
            )
            bleed = 0
            fig.subplots_adjust(
                left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed)
            )
            for cam_idx, (cam, board_coord_mat, frame_coord_mat, ax) in enumerate(
                zip(cams_val, board_coords_val, frame_coords_val, axarr.ravel())
            ):
                board_coord_mat = (
                    torch.from_numpy(board_coord_mat).to(torch.float32).to(dev)
                )
                frame_coord_mat = (
                    torch.from_numpy(frame_coord_mat).to(torch.float32).to(dev)
                )
                projected = cam.project_points(board_coord_mat)
                # vis, _ = vis_lens(cam)
                # ax.imshow(vis[..., :3])
                ax.scatter(
                    frame_coord_mat.detach().cpu().numpy()[:, 0],
                    frame_coord_mat.detach().cpu().numpy()[:, 1],
                    marker="o",
                    s=5,
                )
                ax.scatter(
                    projected.detach().cpu().numpy()[:, 0],
                    projected.detach().cpu().numpy()[:, 1],
                    marker="o",
                    s=5,
                )
                ax.set_axis_off()
                ax.grid(False)
                plt.tight_layout()
            fig.savefig(log_dir + f"/vis_{epoch_idx // 10}.png")
            vis, _ = vis_lens(cam)
            writer.add_image(
                "vis/lens", vis, global_step=epoch_idx, walltime=None, dataformats="HWC"
            )
            plt.close(fig)

        train_rmse = torch.sqrt(torch.cat(loss_train, dim=0).mean())
        writer.add_scalar("train_rmse", train_rmse.detach(), epoch_idx)
        rmse = torch.sqrt(torch.cat(loss_vals, dim=0).mean())
        writer.add_scalar("val_rmse", rmse, epoch_idx)
        lr = optimizer_train.param_groups[0]["lr"]
        writer.add_scalar("lr", lr, epoch_idx)
        print(f"RMSE (train): {train_rmse}, RMSE (val): {rmse}, lr: {lr}")
        if dense_matching:
            print(
                f"Epoch {epoch_idx}, reproj loss: {avg_loss}, dense loss: {dense_loss}"
            )
        else:
            print(f"Epoch {epoch_idx}, reproj loss: {avg_loss}")

        if epoch_idx > 0 and epoch_idx % 25 == 0 and not is_eval:
            PATH = log_dir + "/lensnet_latest.pt"
            torch.save(
                {
                    "epoch": epoch_idx,
                    "model_state_dict": lens_net.state_dict(),
                    "optimizer_state_dict": optimizer_train.state_dict(),
                    "K": K,
                    "RTs_train": RTs_train,
                    "RTs_val": RTs_val,
                    "loss": loss,
                },
                PATH,
            )


def colorize(uv_im, max_mag=None):
    hsv = np.zeros((uv_im.shape[0], uv_im.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(uv_im[..., 0], uv_im[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    # print(mag.max())
    if max_mag is None:
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    else:
        mag = np.clip(mag, 0.0, max_mag)
        mag = mag / max_mag * 255.0
        hsv[..., 2] = mag.astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def image_grid_vis(
    coords_x,
    coords_y,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, coord_x, coord_y in zip(axarr.ravel(), coords_x, coords_y):
        ax.scatter(coord_x, coord_y, marker="o")
        if not show_axes:
            ax.set_axis_off()
    plt.close(fig)


def project_lens(camera: Camera):
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
    # camera_directions_w_lens = batched_func(
    #     camera.get_rays_view, P_sensor.reshape((-1, 2)), 10000
    # )
    camera_directions_w_lens = camera.get_rays_view(P_sensor.reshape((-1, 2)))
    camera_directions_w_lens = camera_directions_w_lens.reshape(
        (P_sensor.shape[0], P_sensor.shape[1], 3)
    )[:, :, :2]

    return camera_directions_w_lens


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

        camera_no_lens = Camera(camera.resolution_w_h, camera.K)
        camera_directions_wo_lens = batched_func(
            camera_no_lens.get_rays_view, P_sensor.reshape((-1, 2)), 10000
        ).reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
        direction_diff = camera_directions_w_lens - camera_directions_wo_lens
        flow_color = colorize(
            direction_diff.detach().cpu().numpy(),
            max_mag=0.1,
        )
      
    return flow_color, camera_directions_w_lens


if __name__ == "__main__":
    cli()
