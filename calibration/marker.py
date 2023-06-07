import torch
import pytorch_lightning as pl


class Marker(pl.LightningModule):
    """
    This represents one keypoint marker on a calibration target pattern.
    """

    def __init__(self, size, working_size, seed, std_thresh=0.0):
        super().__init__()
        assert (
            size % 2 == 1
        ), "marker size must be an odd number to have a well-defined center."
        self.size = size
        self.working_size = working_size
        torch.random.manual_seed(seed)
        # We assume the pixel 'coordinate' to point to the center of the pixel. Hence,
        # the center of the pattern is axactly the coordinate of its center pixel.
        self.register_buffer(
            "center",
            torch.tensor(
                [size // 2, size // 2],
                dtype=torch.float32,
                requires_grad=False,
            ),
            persistent=True,
        )
        # This *is* the actual marker, ready to be optimized through back-propagation.
        # We initialize the memory with values in [-4, 4[ (which means that the values
        # range from close to 0 to 1 when run through a sigmoid) - we use the sigmoid
        # function to guarantee the bounds for the parameters even after gradient
        # updates.
        init = (
            torch.rand((3, size, size), dtype=torch.float32, requires_grad=True) * 8.0
            - 4.0
        )
        self.marker_mem = torch.nn.Parameter(
            data=init,
            requires_grad=True,
        )
        # Standard-deviation threshold for detection.
        self.std_thresh = std_thresh

    def forward(self, batch_size):
        return (
            torch.sigmoid(self.marker_mem)[None, ...].expand(
                batch_size,
                self.marker_mem.shape[0],
                self.marker_mem.shape[1],
                self.marker_mem.shape[2],
            ),
            self.center[None, ...].expand(batch_size, 2),
        )
