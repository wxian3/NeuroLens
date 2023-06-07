from torch import nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import pytorch_lightning as pl
from toolz import curry


def init_weights_zero(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)


@curry
def subnet_fc(c_in, c_out, zero_init=False):
    seq_block = nn.Sequential(nn.Linear(c_in, 256), nn.ReLU(), nn.Linear(256, c_out))
    if zero_init:
        seq_block.apply(init_weights_zero)
    return seq_block


class LensNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.save_hyperparameters()
        self.bidirectional_lens = Ff.SequenceINN(2)
        depth = 12
        for k in range(depth):
            self.bidirectional_lens.append(
                Fm.AllInOneBlock,
                subnet_constructor=subnet_fc(zero_init=(k == (depth - 1))),
                permute_soft=True,
            )

    def forward(self, rays, sensor_to_frustum=True):
        if sensor_to_frustum:
            return self.bidirectional_lens(rays)[0]
        else:
            return self.bidirectional_lens(rays, rev=True)[0]
