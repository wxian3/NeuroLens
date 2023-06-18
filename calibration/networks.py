import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import pytorch_lightning as pl
import torch
from toolz import curry
from torch import nn


def init_weights_zero(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)


@curry
def subnet_fc(c_in, c_out, zero_init=False):
    seq_block = nn.Sequential(nn.Linear(c_in, 256), nn.ELU(), nn.Linear(256, c_out))
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


class iResNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.tol = 1e-6
        self.inp_size_linear = (2,)

        torch.manual_seed(0)
        nodes = [Ff.graph_inn.InputNode(*self.inp_size_linear, name="input")]
        for i in range(5):
        
            nodes.append(
                Ff.graph_inn.Node(
                    nodes[-1],
                    Fm.IResNetLayer,
                    {
                        "hutchinson_samples": 1, 
                        "internal_size": 1024, 
                        "n_internal_layers": 4, 
                    },
                    conditions=[],
                    name=f"i_resnet_{i}",
                )
            )
        nodes.append(Ff.graph_inn.OutputNode(nodes[-1], name="output"))
        self.i_resnet_linear = Ff.GraphINN(nodes, verbose=False)

        for node in self.i_resnet_linear.node_list:
            if isinstance(node.module, Fm.IResNetLayer):
                node.module.lipschitz_correction()

    def forward(self, rays, sensor_to_frustum=True):
        if sensor_to_frustum:
            return self.i_resnet_linear(rays, jac=False)[0]
        else:
            return self.i_resnet_linear(rays, rev=True, jac=False)[0]

    def test_inverse(self):
        x = torch.randn(self.batch_size, *self.inp_size_linear)
        x = x * torch.randn_like(x)
        x = x + torch.randn_like(x)

        y = self.i_resnet_linear(x, jac=False)[0]
        x_hat = self.i_resnet_linear(y, rev=True, jac=False)[0]

        print("Check that inverse is close to input")
        assert torch.allclose(x, x_hat, atol=self.tol)


class LipBoundedPosEnc(nn.Module):
    def __init__(self, inp_features, n_freq, cat_inp=True):
        super().__init__()
        self.inp_feat = inp_features
        self.n_freq = n_freq
        self.cat_inp = cat_inp
        self.out_dim = 2 * self.n_freq * self.inp_feat
        if self.cat_inp:
            self.out_dim += self.inp_feat

    def forward(self, x):
        """
        :param x: (bs, npoints, inp_features)
        :return: (bs, npoints, 2 * out_features + inp_features)
        """
        assert len(x.size()) == 3
        bs, npts = x.size(0), x.size(1)
        const = (2 ** torch.arange(self.n_freq) * np.pi).view(1, 1, 1, -1)
        const = const.to(x)

        # Out shape : (bs, npoints, out_feat)
        cos_feat = torch.cos(const * x.unsqueeze(-1)).view(bs, npts, self.inp_feat, -1)
        sin_feat = torch.sin(const * x.unsqueeze(-1)).view(bs, npts, self.inp_feat, -1)
        out = torch.cat([sin_feat, cos_feat], dim=-1).view(
            bs, npts, 2 * self.inp_feat * self.n_freq
        )
        const_norm = (
            torch.cat([const, const], dim=-1)
            .view(1, 1, 1, self.n_freq * 2)
            .expand(-1, -1, self.inp_feat, -1)
            .reshape(1, 1, 2 * self.inp_feat * self.n_freq)
        )

        if self.cat_inp:
            out = torch.cat([out, x], dim=-1)
            const_norm = torch.cat(
                [const_norm, torch.ones(1, 1, self.inp_feat).to(x)], dim=-1
            )

            return out / const_norm / np.sqrt(self.n_freq * 2 + 2)
        else:

            return out / const_norm / np.sqrt(self.n_freq * 2)


class InvertibleResBlockLinear(nn.Module):
    def __init__(
        self, inp_dim, hid_dim, nblocks=1, nonlin="leaky_relu", pos_enc_freq=None
    ):
        super().__init__()
        self.dim = inp_dim
        self.nblocks = nblocks

        self.pos_enc_freq = pos_enc_freq
        if self.pos_enc_freq is not None:
            inp_dim_af_pe = self.dim * (self.pos_enc_freq * 2 + 1)
            self.pos_enc = LipBoundedPosEnc(self.dim, self.pos_enc_freq)
        else:
            self.pos_enc = lambda x: x
            inp_dim_af_pe = inp_dim

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.utils.spectral_norm(nn.Linear(inp_dim_af_pe, hid_dim)))
        for _ in range(self.nblocks):
            self.blocks.append(
                nn.utils.spectral_norm(
                    nn.Linear(hid_dim, hid_dim),
                )
            )
        self.blocks.append(
            nn.utils.spectral_norm(
                nn.Linear(hid_dim, self.dim),
            )
        )

        self.nonlin = nonlin.lower()
        if self.nonlin == "leaky_relu":
            self.act = nn.LeakyReLU()
        elif self.nonlin == "relu":
            self.act = nn.ReLU()
        elif self.nonlin == "elu":
            self.act = nn.ELU()
        elif self.nonlin == "softplus":
            self.act = nn.Softplus()
        else:
            raise NotImplementedError

    def forward_g(self, x):
        orig_dim = len(x.size())
        if orig_dim == 2:
            x = x.unsqueeze(0)

        y = self.pos_enc(x)
        for block in self.blocks[:-1]:
            y = self.act(block(y))
        y = self.blocks[-1](y)

        if orig_dim == 2:
            y = y.squeeze(0)

        return y

    def forward(self, x):
        return x + self.forward_g(x)

    def invert(self, y, verbose=False, iters=35):
        return self.fixed_point_invert(
            lambda x: self.forward_g(x), y, iters=iters, verbose=verbose
        )

    def fixed_point_invert(self, g, y, iters=35, verbose=False):
        with torch.no_grad():
            x = y
            dim = x.size(-1)
            for i in range(iters):
                x = y - g(x)
                if verbose:
                    err = (y - (x + g(x))).view(-1, dim).norm(dim=-1).mean()
                    err = err.detach().cpu().item()
                    print("iter:%d err:%s" % (i, err))
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 2
        self.out_dim = 2
        self.hidden_size = 512
        self.n_blocks = 5
        self.n_g_blocks = 5
        self.tol = 1e-6

        # Network modules
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.blocks.append(
                InvertibleResBlockLinear(
                    self.dim,
                    self.hidden_size,
                    nblocks=self.n_g_blocks,
                    nonlin="elu",
                    pos_enc_freq=5,
                )
            )

    def forward(self, rays, sensor_to_frustum=True):
        if sensor_to_frustum:
            rays = rays.unsqueeze(0)
            out = rays
            for block in self.blocks:
                out = block(out)
            return out[0]
        else:
            rays = rays.unsqueeze(0)
            x = rays
            for block in self.blocks[::-1]:
                x = block.invert(x, verbose=False, iters=35)
            return x[0]

    def test_inverse(self):
        x = torch.rand(7, self.dim) * 2 - 1

        y = self.forward(x)
        print(y.max(dim=0), y.min(dim=0))
