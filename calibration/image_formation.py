import torch
import pytorch_lightning as pl
import kornia.augmentation as K

# from .transformations import RandomElasticTransform


class ImageFormation(pl.LightningModule):
    """
    This models the image formation process, including printing of the pattern.

    This transformation must randomly distort the input images with valid
    distortions and at the same time provide the position of the marker
    center (given those distortions).
    """

    def __init__(
        self,
        # print_dist: RandomElasticTransform,
        motion_dist: K.RandomMotionBlur,
        affine_dist: K.RandomAffine,
        blur_dist: K.RandomGaussianBlur,
        color_dist: K.ColorJitter,
        noise_dist: K.RandomGaussianNoise,
        working_size: int,
    ):
        """
        print_dist: K.RandomElasticTransform; model distortion of the
            pattern through printing.
        motion_dist: K.RandomMotionBlur; model motion blur during
            recording of the pattern.
        affine_dist: K.RandomAffine; model imperfect retrieval of the
            pattern.
        color_dist: K.ColorJitter; model color distortion of the recording.

        """
        super().__init__()
        # self.print_dist = print_dist
        self.motion_dist = motion_dist
        self.affine_dist = affine_dist
        self.blur_dist = blur_dist
        self.color_dist = color_dist
        self.noise_dist = noise_dist
        self.working_size = working_size
        self.resizer = K.Resize(self.working_size)

    def _affine_mul(self, mat1, mat2):
        # Pad mat2.
        mat2 = torch.cat(
            (
                mat2[:, 1:2],
                mat2[:, 0:1],
                torch.ones((mat2.shape[0], 1), dtype=mat2.dtype, device=mat2.device),
            ),
            dim=1,
        )
        multiplied = torch.matmul(mat1, mat2[:, :, None])
        normalized = multiplied[:, :2, 0] / multiplied[:, 2:3, 0]
        return torch.cat((normalized[:, 1:2], normalized[:, 0:1]), dim=1)

    def forward(self, pattern, center):
        # Assume the pattern is used in a grid.
        pattern_grid = pattern.repeat(1, 1, 3, 3)
        center = (
            center
            + torch.tensor(
                pattern.shape[2:4], dtype=torch.float32, device=pattern.device
            )[None, ...]
        )
        # print the pattern.
        # printed, transform = self.print_dist(pattern_grid)
        # # Apply shift to
        # center = self.print_dist.apply_to_coordinates(center)
        # Now we get to the recording stage.
        # We assume the pattern is moving, so it could have a
        # certain amount of motion blur.
        recorded = self.motion_dist(pattern_grid)
        # We assume the pattern is recorded using an arbitrary
        # affine transform; however we can normalize that away using
        # the AprilTag. Hence, we drop the homography for now.
        # However, we assume that we can't retrieve the pattern perfectly.
        # That's why we add a random affine transformation.
        recorded, transform = self.affine_dist(recorded)
        center = self._affine_mul(transform, center)
        # Add sensor noise.
        noised = self.noise_dist(recorded)
        # We simulate that this could have been recorded from further
        # away and later resized, so we have quite a bit of loss
        # of detail.
        # Rule of thumb for Gauss filter: filter half-width should be
        # about 3σ.
        # Convolving two times with Gaussian kernel of width σ is
        # same as convolving once with kernel of width σ√2.
        blurred = self.blur_dist(noised)
        # We assume that we don't have color-calibrated cameras (at this
        # stage), so we have to be robust to all kinds of color-shifts.
        color_distorted = self.color_dist(blurred)
        # Get only the center crop.
        cropped = color_distorted[
            :,
            :,
            pattern.shape[2] : -pattern.shape[2],
            pattern.shape[3] : -pattern.shape[3],
        ]
        center = (
            center
            - torch.tensor(
                pattern.shape[2:4], dtype=torch.float32, device=center.device
            )[None, ...]
        )
        if self.working_size != cropped.shape[2]:
            center *= self.working_size / cropped.shape[2]
            cropped = self.resizer(cropped)
        return cropped, center
