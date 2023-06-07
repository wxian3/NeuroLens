from typing import Optional, Tuple
import numpy as np
import torch

import torch.nn.functional as F

from .networks import LensNet


# The following methods (rotation_6d_to_matrix and matrix_to_rotation_6d) fall under the PyTorch3D license.


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)
    Returns:
        6D rotation representation, of size (*, 6)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def zero_K_gradient(grad: torch.Tensor):
    # We zero the gradients for regions that we want to keep fixed.
    grad[0, 1] = 0.0
    grad[1, 0] = 0.0
    grad[2, :] = 0.0


class Camera:
    def __init__(
        self,
        resolution_w_h: Tuple[int, int],
        K: torch.Tensor,
        lensnet: Optional[LensNet] = None,
        RT=None,
    ):
        """
        Parameters
        ==========

        resolution_w_h: (int, int). Resolution width and height.
        K: torch.tensor. Intrinsic matrix of the form
            f_x  0   c_x
             0  f_y  c_y
             0   0    1
        lensnet: LensNet. Distortion model. Can be None
            if not existent (pinhole model).
        RT: torch.tensor. Extrinsic w2c matrix of the form
            r11 r12 r13 t1
            r21 r22 r23 t2
            r31 r32 r33 t3
        """
        assert resolution_w_h[0] > 0
        assert resolution_w_h[1] > 0
        assert len(resolution_w_h) == 2
        self.resolution_w_h = resolution_w_h
        assert isinstance(K, torch.Tensor)
        assert K.ndim == 2
        assert K.shape[0] == 3
        assert K.shape[1] == 3
        assert K[0, 0] > 0.0
        assert K[0, 1] == 0.0
        assert K[1, 0] == 0.0
        assert K[1, 1] > 0.0
        assert K[2, 0] == 0.0
        assert K[2, 1] == 0.0
        assert K[2, 2] == 1.0
        self.K = K
        self.K.register_hook(zero_K_gradient)
        assert lensnet is None or isinstance(lensnet, LensNet)
        self.lensnet = lensnet
        assert RT is None or isinstance(RT, torch.Tensor)
        if RT is None:
            self.RT = torch.eye(3, 4, dtype=torch.float32, device=K.device)
        else:
            assert RT.ndim == 2
            assert RT.shape[0] == 3
            assert RT.shape[1] == 4
            self.RT = RT

    @property
    def RT(self):
        rot_mat = rotation_6d_to_matrix(self.R[None, ...])[0]
        return torch.cat((rot_mat, self.T), dim=1)

    @RT.setter
    def RT(self, value):
        self.R = matrix_to_rotation_6d(value[:, :3])
        self.T = value[:, 3:4]

    @staticmethod
    def homogenize(X: torch.Tensor):
        assert X.ndim == 2
        assert X.shape[1] in (2, 3)
        return torch.cat(
            (X, torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)), dim=1
        )

    @staticmethod
    def dehomogenize(X: torch.Tensor):
        assert X.ndim == 2
        assert X.shape[1] in (3, 4)
        return X[:, :-1] / X[:, -1:]

    def world_to_view(self, P_world: torch.Tensor):
        assert P_world.ndim == 2
        assert P_world.shape[1] == 3
        P_world_hom = self.homogenize(P_world)
        P_view = (self.RT @ P_world_hom.T).T
        return P_view

    def view_to_world(self, P_view: torch.Tensor):
        assert P_view.ndim == 2
        assert P_view.shape[1] == 3
        P_view_shifted = P_view - self.RT[:, 3][None, ...]
        P_world = (self.RT[:, :3].T @ P_view_shifted.T).T
        return P_world

    def view_to_sensor(self, P_view: torch.Tensor):
        P_view_outsidelens_direction = self.dehomogenize(P_view)  # x' = x/z, y' = y/z
        if self.lensnet is not None:
            P_view_insidelens_direction = self.lensnet.forward(
                P_view_outsidelens_direction, sensor_to_frustum=False
            )
        else:
            P_view_insidelens_direction = P_view_outsidelens_direction
        P_view_insidelens_direction_hom = self.homogenize(P_view_insidelens_direction)
        P_sensor = self.dehomogenize((self.K @ P_view_insidelens_direction_hom.T).T)
        return P_sensor

    def get_rays_view(self, P_sensor: torch.Tensor):
        assert P_sensor.ndim == 2
        assert P_sensor.shape[1] == 2
        P_sensor_hom = self.homogenize(P_sensor)
        P_view_insidelens_direction_hom = (torch.inverse(self.K) @ P_sensor_hom.T).T
        P_view_insidelens_direction = self.dehomogenize(P_view_insidelens_direction_hom)
        if self.lensnet is not None:
            P_view_outsidelens_direction = self.lensnet.forward(
                P_view_insidelens_direction, sensor_to_frustum=True
            )
        else:
            P_view_outsidelens_direction = P_view_insidelens_direction
        return self.homogenize(P_view_outsidelens_direction)

    def get_rays_world(self, P_sensor: torch.tensor):
        rays_view = self.get_rays_view(P_sensor)
        origins_view = torch.zeros_like(rays_view)
        origins_world = self.view_to_world(origins_view)
        rays_world = (self.RT[:, :3].T @ rays_view.T).T
        return origins_world, rays_world

    def sensor_to_view(self, P_sensor_and_depth: torch.Tensor):
        assert P_sensor_and_depth.ndim == 2
        assert P_sensor_and_depth.shape[1] == 3
        P_sensor, P_depth = P_sensor_and_depth[:, :2], P_sensor_and_depth[:, 2:3]
        rays = self.get_rays_view(P_sensor)
        P_view = rays * P_depth
        return P_view

    def project_points(self, P_world: torch.Tensor):
        P_view_outsidelens = self.world_to_view(P_world)
        P_sensor = self.view_to_sensor(P_view_outsidelens)
        return P_sensor

    def unproject_points(self, P_sensor: torch.Tensor):
        P_view_outsidelens = self.sensor_to_view(P_sensor)
        P_world = self.view_to_world(P_view_outsidelens)
        return P_world
