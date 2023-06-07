from typing import Optional
import torch
from torch import nn
import pytorch_lightning as pl
import time
from collections import deque
from ray import tune
from hydra.utils import instantiate

from . import standard_models
from .config import ConfigMarker, ConfigImageFormation
from .marker import Marker


class Model(pl.LightningModule):
    """
    This model represents the entire process from marker creation to recording to point detection.

    If called without an already detected and extracted marker, it will randomly project
    the marker and simulate the entire image formation pipeline, then run it through
    the center point detection and return the results (true center points and predicted center
    points) to enable optimization of the entire model.

    If a detection is provided, it uses the existing detection and provides the estimated
    marker center point.
    """

    def __init__(
        self,
        marker: ConfigMarker,
        image_formation: Optional[ConfigImageFormation] = None,
        batch_size=4,
        log_every=-1,
        lr=1e-2,
        lr_fcn_fac=1.0,
        lr_marker_fac=100.0,
        n_latent=200,
        n_hidden=2,
        reg_weight=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.marker = instantiate(marker)
        if image_formation is not None:
            self.image_formation = instantiate(image_formation)
        else:
            self.image_formation = None
        self.batch_size = batch_size
        # Use an efficient predictor to create a compact, latent representation.
        self.predictor = standard_models.mobilenet_v3_small(
            pretrained=True, progress=True
        )
        # Use an FCN to find the center coordinate.
        self.fcn = nn.Sequential(
            *(
                [
                    nn.Linear(1000, n_latent, True),
                    nn.ReLU(),
                ]
                + [nn.Linear(n_latent, n_latent, True), nn.ReLU()] * (n_hidden - 1)
                + [
                    nn.Linear(n_latent, 3),
                ]
            )
        )
        self._last_log_written = 0
        self.log_every = log_every
        self.lr = lr
        self.lr_fcn_fac = lr_fcn_fac
        self.lr_marker_fac = lr_marker_fac
        self.loss_deque = deque(maxlen=50)
        self.reg_weight = reg_weight
        self.gnllloss = nn.GaussianNLLLoss(reduction="sum")

    def forward(self, detection=None):
        if detection is None:
            # Go through the image formation process.
            self._marker, self._marker_center = self.marker(self.batch_size)
            self._detection, self._detection_center = self.image_formation(
                self._marker, self._marker_center
            )
        else:
            # Test time, we're looking for detections with unknown center.
            self._detection_center = None
            self._detection = detection
        clres = self.predictor(self._detection)
        fcnres = self.fcn(clres)
        coords = torch.sigmoid(fcnres[:, :2]) * self.marker.working_size
        # We are predicting the standard deviation and are limiting
        # it to a reasonable size to stabilize the optimization.
        sds = torch.sigmoid(fcnres[:, 2:3]) * (self.marker.working_size / 2)
        variances = torch.square(sds)
        return coords, variances

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.predictor.parameters(), "lr": self.lr},
                {"params": self.fcn.parameters(), "lr": self.lr * self.lr_fcn_fac},
                {
                    "params": self.marker.parameters(),
                    "lr": self.lr * self.lr_marker_fac,
                },
            ]
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        coords, variances = self.forward()
        # L1 loss.
        # self._loss = torch.sum((coords - self._detection_center).abs())
        # Gaussian negative log likelihood, homoscedastic.
        self._loss = self.gnllloss(coords, self._detection_center, variances)
        distances = torch.linalg.vector_norm(coords - self._detection_center, dim=1)
        if self.reg_weight > 0.0:
            self._regularizer = torch.sum((1.0 - self._marker).abs()) * self.reg_weight
        else:
            self._regularizer = 0.0
        self.maybe_log(
            coords, self._loss, self._regularizer, distances, variances, batch_idx
        )
        self.loss_deque.append(self._loss.item())
        if len(self.loss_deque) >= 50:
            self.loss_deque.popleft()
        if tune.is_session_enabled():
            tune.report(iterations=batch_idx, accuracy=self._loss.item())
        return self._loss + self._regularizer

    def maybe_log(self, coords, loss, regularizer, distances, variances, batch_idx):
        if time.time() - self._last_log_written > self.log_every:
            # Scalars.
            self.log("max/distance", distances.max())
            self.log("max/variance", variances.max())
            self.log("average/distance", distances.mean())
            self.log("average/variance", variances.mean())
            self.log("average/95perc_conf_dist", 2.0 * torch.sqrt(variances).mean())
            self.log("train/loss", loss)
            self.log("train/regularizer", regularizer)
            if hasattr(self, "_marker_grad") and self._marker_grad is not None:
                self.logger.experiment.add_histogram(
                    "grad/marker", self.marker.marker_mem.grad, batch_idx
                )
            # Histograms.
            self.logger.experiment.add_histogram(
                "val/marker", self.marker.marker_mem, batch_idx
            )
            self.logger.experiment.add_histogram("val/distances", distances, batch_idx)
            self.logger.experiment.add_histogram("val/variances", variances, batch_idx)
            # Add images.
            self.logger.experiment.add_image(
                f"images/marker",
                (self._marker[0] * 255.0).to(torch.uint8),
                batch_idx,
                dataformats="CWH",
            )
            self.logger.experiment.add_image(
                f"images/detections",
                (self._detection * 255.0).to(torch.uint8),
                batch_idx,
                dataformats="NCHW",
            )
            center_vis = self._detection.clone()
            for idx in range(self._detection.shape[0]):
                center_vis[
                    idx,
                    :,
                    self._detection_center[idx, 0].long()
                    - 3 : self._detection_center[idx, 0].long()
                    + 3,
                    self._detection_center[idx, 1].long()
                    - 3 : self._detection_center[idx, 1].long()
                    + 3,
                ] = 0.0
                resize_fac = self.marker.working_size / self.marker.size
                center_vis[
                    idx,
                    :,
                    (self._marker_center[idx, 0] * resize_fac).long()
                    - 3 : (self._marker_center[idx, 0] * resize_fac).long()
                    + 3,
                    (self._marker_center[idx, 1] * resize_fac).long()
                    - 3 : (self._marker_center[idx, 1] * resize_fac).long()
                    + 3,
                ] = 0.0
                center_vis[
                    idx,
                    :,
                    coords[idx, 0].long() - 3 : coords[idx, 0].long() + 3,
                    coords[idx, 1].long() - 3 : coords[idx, 1].long() + 3,
                ] = 0.0
                # red detection center.
                center_vis[
                    idx,
                    0,
                    self._detection_center[idx, 0].long()
                    - 3 : self._detection_center[idx, 0].long()
                    + 3,
                    self._detection_center[idx, 1].long()
                    - 3 : self._detection_center[idx, 1].long()
                    + 3,
                ] = 1.0
                # green: original marker center.
                center_vis[
                    idx,
                    1,
                    (self._marker_center[idx, 0] * resize_fac).long()
                    - 3 : (self._marker_center[idx, 0] * resize_fac).long()
                    + 3,
                    (self._marker_center[idx, 1] * resize_fac).long()
                    - 3 : (self._marker_center[idx, 1] * resize_fac).long()
                    + 3,
                ] = 1.0
                # blue: model estimation for the center.
                center_vis[
                    idx,
                    2,
                    coords[idx, 0].long() - 3 : coords[idx, 0].long() + 3,
                    coords[idx, 1].long() - 3 : coords[idx, 1].long() + 3,
                ] = 1.0
            self.logger.experiment.add_image(
                f"images/centers",
                (center_vis * 255.0).to(torch.uint8),
                batch_idx,
                dataformats="NCHW",
            )
            self._last_log_written = time.time()

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        self._marker_grad = self.marker.marker_mem.grad
