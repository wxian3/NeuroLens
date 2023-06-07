import numpy as np
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def invert_function(x, func):
    from scipy import interpolate
    from scipy.interpolate import dfitpack
    y = func(x)
    dy = np.concatenate(([0], np.diff(y)))
    y = y[dy>=0]
    x = x[dy>=0]
    try:
        inter = interpolate.InterpolatedUnivariateSpline(y, x)
    # dfitpack.error
    except Exception: # pragma: no cover
        inter = lambda x: x
    return inter


class LensDistortion():  # pragma: no cover
    def __init__(self):
        offset = np.array([0, 0])
        scale = 1

    def imageFromDistorted(self, points):
        # return the points as they are
        return points

    def distortedFromImage(self, points):
        # return the points as they are
        return points

    def undistortImage(self, image):
        return image


class NoDistortion(LensDistortion):
    """
    The default model for the lens distortion which does nothing.
    """
    pass

class Vignetting(LensDistortion):
    def __init__(self, k1=None, k2=None, k3=None, projection=None):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.projection = projection
        if projection is not None:
            self.image_width_px = projection['image_width_px']
            self.image_height_px = projection['image_height_px']
            self.center_x_px = projection['center_x_px']
            self.center_y_px = projection['center_y_px']
            self.scale = np.min([self.image_width_px, self.image_height_px]) / 2
            self.offset = np.array([self.center_x_px, self.center_y_px])

    def _convert_radius(self, r):
        return 1 + self.k1*r**2 + self.k2*r**4 + self.k3*r**6

    def compute_intensity(self, points, intensity):
        # ensure that the points are provided as an array
        # and rescale the points to that the center is at 0 and the border at 1
        points = (np.array(points)-self.offset)/self.scale
        # calculate the radius form the center
        r = np.linalg.norm(points, axis=-1)[..., None]
        r = r / r.max()
        # scale the intensities
        r_t =  torch.from_numpy(self._convert_radius(r)).T
        intensity = intensity * r_t
        return intensity

    def vignet(self, image):
        x, y = torch.meshgrid([torch.arange(0, self.image_width_px),
                    torch.arange(0, self.image_height_px)], indexing='ij')
        x = x.float() + 0.5
        y = y.float() + 0.5
        coord = torch.cat((y.unsqueeze(-1), x.unsqueeze(-1)), 2).view(-1, 2)
        image = image.permute(2,0,1)
        vignet_image = self.compute_intensity(coord, image.view(3, -1)).reshape(3, self.image_width_px, self.image_height_px)

        return vignet_image.permute(1,2,0)


class BrownLensDistortion(LensDistortion):
    r"""
    The most common distortion model is the Brown's distortion model. In CameraTransform, we only consider the radial part
    of the model, as this covers all common cases and the merit of tangential components is disputed. This model relies on
    transforming the radius with even polynomial powers in the coefficients :math:`k_1, k_2, k_3`. This distortion model is
    e.g. also used by OpenCV or AgiSoft PhotoScan.
    Adjust scale and offset of x and y to be relative to the center:
    .. math::
        x' &= \frac{x-c_x}{f_x}\\
        y' &= \frac{y-c_y}{f_y}
    Transform the radius from the center with the distortion:
    .. math::
        r &= \sqrt{x'^2 + y'^2}\\
        r' &= r \cdot (1 + k_1 \cdot r^2 + k_2 \cdot r^4 + k_3 \cdot r^6)\\
        x_\mathrm{distorted}' &= x' / r \cdot r'\\
        y_\mathrm{distorted}' &= y' / r \cdot r'
    Readjust scale and offset to obtain again pixel coordinates:
    .. math::
        x_\mathrm{distorted} &= x_\mathrm{distorted}' \cdot f_x + c_x\\
        y_\mathrm{distorted} &= y_\mathrm{distorted}' \cdot f_y + c_y
    """
    projection = None

    def __init__(self, k1=None, k2=None, k3=None, projection=None):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.projection = projection
        if projection is not None:
            self.image_width_px = projection['image_width_px']
            self.image_height_px = projection['image_height_px']
            self.center_x_px = projection['center_x_px']
            self.center_y_px = projection['center_y_px']

        self._init_inverse()

    def _init_inverse(self):
        r = np.arange(0, 2, 0.1)
        self._convert_radius_inverse = invert_function(r, self._convert_radius)

        if self.projection is not None:
            self.scale = np.min([self.image_width_px, self.image_height_px]) / 2
            self.offset = np.array([self.center_x_px, self.center_y_px])
        else:
            self.scale = 1
            self.offset = np.array([0, 0])

    def _convert_radius(self, r):
        return r*(1 + self.k1*r**2 + self.k2*r**4 + self.k3*r**6)

    def imageFromDistorted(self, points):
        # ensure that the points are provided as an array
        # and rescale the points to that the center is at 0 and the border at 1
        points = (np.array(points)-self.offset)/self.scale
        # calculate the radius form the center
        r = np.linalg.norm(points, axis=-1)[..., None]
        # transform the points
        points = points / r * self._convert_radius_inverse(r)
        # rescale back to the image
        return points * self.scale + self.offset

    def distortedFromImage(self, points):
        # ensure that the points are provided as an array
        # and rescale the points to that the center is at 0 and the border at 1
        points = (np.array(points)-self.offset)/self.scale
        # calculate the radius form the center
        r = np.linalg.norm(points, axis=-1)[..., None]
        # transform the points
        points = points / r * self._convert_radius(r)
        # rescale back to the image
        return points * self.scale + self.offset

    def distortImage(self, image):
        x, y = torch.meshgrid([torch.arange(0, self.image_width_px),
                    torch.arange(0, self.image_height_px)], indexing='ij')
        x = x.float() + 0.5
        y = y.float() + 0.5
        coord = torch.cat((y.unsqueeze(-1), x.unsqueeze(-1)), 2).view(-1, 2)
        grid = self.imageFromDistorted(coord).reshape(self.image_width_px, self.image_height_px, 2)
        image_size_xy = torch.tensor([self.image_height_px, self.image_width_px])
        grid = (grid / image_size_xy.view(1, 1, 2)) * 2 - 1
        image = image.permute(2,0,1)
        distort_image = F.grid_sample(image.float().unsqueeze(0), grid.float().unsqueeze(0), align_corners=False, padding_mode="border")[0]
        distort_image = distort_image.permute(1,2,0)

        return distort_image

class ABCDistortion(LensDistortion):
    r"""
    The ABC model is a less common distortion model, that just implements radial distortions. Here the radius is transformed
    using a polynomial of 4th order. It is used e.g. in PTGui.
    Adjust scale and offset of x and y to be relative to the center:
    .. math::
        s &= 0.5 \cdot \mathrm{min}(\mathrm{im}_\mathrm{width}, \mathrm{im}_\mathrm{height})\\
        x' &= \frac{x-c_x}{s}\\
        y' &= \frac{y-c_y}{s}
    Transform the radius from the center with the distortion:
    .. math::
        r &= \sqrt{x^2 + y^2}\\
        r' &= d \cdot r + c \cdot r^2 + b \cdot r^3 + a \cdot r^4\\
        d &= 1 - a - b - c
    Readjust scale and offset to obtain again pixel coordinates:
    .. math::
        x_\mathrm{distorted} &= x_\mathrm{distorted}' \cdot s + c_x\\
        y_\mathrm{distorted} &= y_\mathrm{distorted}' \cdot s + c_y
    """
    projection = None

    def __init__(self, a=None, b=None, c=None, projection=None):
        self.a = a
        self.b = b
        self.c = c
        self.projection = projection
        if projection is not None:
            self.image_width_px = projection['image_width_px']
            self.image_height_px = projection['image_height_px']
            self.center_x_px = projection['center_x_px']
            self.center_y_px = projection['center_y_px']

        self._init_inverse()

    def _init_inverse(self):
        self.d = 1 - self.a - self.b - self.c
        r = np.arange(0, 2, 0.1)
        self._convert_radius_inverse = invert_function(r, self._convert_radius)

        if self.projection is not None:
            self.scale = np.min([self.image_width_px, self.image_height_px]) / 2
            self.offset = np.array([self.center_x_px, self.center_y_px])
        else:
            self.scale = 1
            self.offset = np.array([0, 0])

    def _convert_radius(self, r):
        return self.d * r + self.c * r**2 + self.b * r**3 + self.a * r**4

    def imageFromDistorted(self, points):
        # ensure that the points are provided as an array
        # and rescale the points to that the center is at 0 and the border at 1
        points = (np.array(points)-self.offset)/self.scale
        # calculate the radius form the center
        r = np.linalg.norm(points, axis=-1)[..., None]
        # transform the points
        points = points / r * self._convert_radius_inverse(r)
        # rescale back to the image
        return points * self.scale + self.offset

    def distortedFromImage(self, points):
        # ensure that the points are provided as an array
        # and rescale the points to that the center is at 0 and the border at 1
        points = (np.array(points)-self.offset)/self.scale
        # calculate the radius form the center
        r = np.linalg.norm(points, axis=-1)[..., None]
        # transform the points
        points = points / r * self._convert_radius(r)
        # rescale back to the image
        return points * self.scale + self.offset

    def distortImage(self, image):
        x, y = torch.meshgrid([torch.arange(0, self.image_width_px),
                    torch.arange(0, self.image_height_px)], indexing='ij')
        x = x.float() + 0.5
        y = y.float() + 0.5
        coord = torch.cat((y.unsqueeze(-1), x.unsqueeze(-1)), 2).view(-1, 2)
        grid = self.imageFromDistorted(coord).reshape(self.image_width_px, self.image_height_px, 2)
        image_size_xy = torch.tensor([self.image_height_px, self.image_width_px])
        grid = (grid / image_size_xy.view(1, 1, 2)) * 2 - 1
        image = image.permute(2,0,1)
        distort_image = F.grid_sample(image.float().unsqueeze(0), grid.float().unsqueeze(0), align_corners=False, padding_mode="border")[0]
        distort_image = distort_image.permute(1,2,0)

        return distort_image
