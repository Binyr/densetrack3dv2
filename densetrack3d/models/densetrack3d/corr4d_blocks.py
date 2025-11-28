import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from densetrack3d.utils.timer import CUDATimer

class Conv2dSamePadding(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        
        # print(f"pad_h: {pad_h}, pad_w: {pad_w}")
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            # self.padding,
            0,
            self.dilation,
            self.groups,
        )


class Corr4DMLP(nn.Module):
    def __init__(
        self,
        in_channel: int = 49,
        out_channels: tuple = (64, 128, 128),
        kernel_shapes: tuple = (3, 3, 2),
        strides: tuple = (2, 2, 2),
    ):
        super().__init__()
        self.in_channels = [in_channel] + list(out_channels[:-1])
        self.out_channels = out_channels
        self.kernel_shapes = kernel_shapes
        self.strides = strides

        # self.linear_trans = nn.Sequential(
        #     nn.Conv2d(in_channel, 64, 1, 1, 0),
        #     nn.GroupNorm(64 // 16, 64),
        #     nn.ReLU()
        # )

        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2dSamePadding(
                        in_channels=self.in_channels[i],
                        out_channels=self.out_channels[i],
                        kernel_size=self.kernel_shapes[i],
                        stride=self.strides[i],
                    ),
                    # nn.Conv2d(
                    #     in_channels=self.in_channels[i],
                    #     out_channels=self.out_channels[i],
                    #     kernel_size=self.kernel_shapes[i],
                    #     stride=self.strides[i],
                    # ),
                    nn.GroupNorm(out_channels[i] // 16, out_channels[i]),
                    nn.ReLU(),
                )
                for i in range(len(out_channels))
            ]
        )

    def forward(self, x):
        """
        x: (b, h, w, i, j)
        """
        b = x.shape[0]

        out1 = rearrange(x, "b h w i j -> b (i j) h w")
        out2 = rearrange(x, "b h w i j -> b (h w) i j")

        out = torch.cat([out1, out2], dim=0)  # (2 * b) c h w

        # out = self.linear_trans(out)

        for i in range(len(self.out_channels)):
            out = self.conv[i](out)

        out = torch.mean(out, dim=(2, 3))  # (2 * b, out_channels[-1])
        out1, out2 = torch.split(out, b, dim=0)  # (b, out_channels[-1])
        out = torch.cat([out1, out2], dim=-1)  # (b, 2*out_channels[-1])

        return out


class Corr4DCNN(nn.Module):
    def __init__(
        self,
        linear_in_c: int = 49,
        in_channel: int = 49,
        out_channels: tuple = (64, 128, 128),
        kernel_shapes: tuple = (3, 3, 2),
        strides: tuple = (2, 2, 2),
    ):
        super().__init__()
        self.in_channels = [in_channel] + list(out_channels[:-1])
        self.out_channels = out_channels
        self.kernel_shapes = kernel_shapes
        self.strides = strides

        # self.linear_trans = nn.Sequential(
        #     nn.Conv2d(49, 64, 1, 1, 0),
        #     # nn.GroupNorm(64 // 16, 64),
        #     # nn.ReLU()
        # )

        # cmdtop_params = {
        #     "in_channel": 49,
        #     "out_channels": (64, 128, 128),
        #     "kernel_shapes": (3, 3, 2),
        #     "strides": (2, 2, 2),
        # }

        self.linear_trans = nn.Conv2d(linear_in_c, 64, kernel_size=1)

        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.in_channels[i], 
                        out_channels=self.out_channels[i], 
                        kernel_size=self.kernel_shapes[i], 
                        stride=self.strides[i], 
                        padding=1 if i <= 1 else 0,
                    ),
                    # Conv2dSamePadding(
                    #     in_channels=self.in_channels[i],
                    #     out_channels=self.out_channels[i],
                    #     kernel_size=self.kernel_shapes[i],
                    #     stride=self.strides[i],
                    # ),
                    nn.GroupNorm(out_channels[i] // 16, out_channels[i]),
                    nn.ReLU(),
                )
                for i in range(len(out_channels))
            ]
        )

    def forward(self, x):
        """
        x: (b, h, w, i, j)
        """
        b = x.shape[0]

        out1 = rearrange(x, "b h w i j -> b (i j) h w")
        out2 = rearrange(x, "b h w i j -> b (h w) i j")

        out = torch.cat([out1, out2], dim=0)  # (2 * b) c h w

        # with CUDATimer(f"linear"):
        out = self.linear_trans(out)

        for i in range(len(self.out_channels)):
            # with CUDATimer(f"conv {i}"):
            out = self.conv[i](out)
            # print(f"conv {i} out shape: {out.shape}")

        out = torch.mean(out, dim=(2, 3))  # (2 * b, out_channels[-1])
        out1, out2 = torch.split(out, b, dim=0)  # (b, out_channels[-1])
        out = torch.cat([out1, out2], dim=-1)  # (b, 2*out_channels[-1])

        # out = rearrange(out, "(k b) c 1 1 -> b (k c)", b=b)
        return out


class Corr4DCNN2(nn.Module):
    def __init__(
        self,
        linear_in_c1: int = 49,
        linear_in_c2: int = 49,
        in_channel: int = 64,
        out_channels: tuple = (64, 128, 128),
        kernel_shapes: tuple = (3, 3, 2), # customed for 7x7x7x7
        strides: tuple = (2, 2, 2),
        paddings: tuple = (1, 1, 0),
        in_channel2: int = 64,
        out_channels2: tuple = (64, 128, 128),
        kernel_shapes2: tuple = (3, 3, 2), # customed for 7x7x7x7
        strides2: tuple = (2, 2, 2),
        paddings2: tuple = (1, 1, 0)
    ):
        super().__init__()
        self.in_channels = [in_channel] + list(out_channels[:-1])
        self.out_channels = out_channels
        self.kernel_shapes = kernel_shapes
        self.strides = strides

        self.in_channels2 = [in_channel2] + list(out_channels2[:-1])
        self.out_channels2 = out_channels2
        self.kernel_shapes2 = kernel_shapes2
        self.strides2 = strides2

        self.linear_trans1 = nn.Conv2d(linear_in_c1, in_channel, kernel_size=1)
        self.linear_trans2 = nn.Conv2d(linear_in_c2, in_channel2, kernel_size=1)

        self.conv1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.in_channels[i], 
                        out_channels=self.out_channels[i], 
                        kernel_size=self.kernel_shapes[i], 
                        stride=self.strides[i], 
                        padding=paddings[i],
                    ),
                    # Conv2dSamePadding(
                    #     in_channels=self.in_channels[i],
                    #     out_channels=self.out_channels[i],
                    #     kernel_size=self.kernel_shapes[i],
                    #     stride=self.strides[i],
                    # ),
                    # nn.GroupNorm(out_channels[i] // 16, out_channels[i]),
                    nn.GroupNorm(1, out_channels[i]),
                    # nn.GroupNorm(max(out_channels[i] // 16, 1), out_channels[i]),
                    nn.ReLU(),
                )
                for i in range(len(out_channels))
            ]
        )

        self.conv2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.in_channels2[i], 
                        out_channels=self.out_channels2[i], 
                        kernel_size=self.kernel_shapes2[i], 
                        stride=self.strides2[i], 
                        padding=paddings2[i],
                    ),
                    nn.GroupNorm(1, out_channels2[i]),
                    nn.ReLU(),
                )
                for i in range(len(out_channels2))
            ]
        )

    def forward(self, x):
        """
        x: (b, h, w, i, j)
        """
        b = x.shape[0]
        out1 = rearrange(x, "b h w i j -> b (i j) h w") # hp, wp, hq, wq
        out2 = rearrange(x, "b h w i j -> b (h w) i j")

        # with CUDATimer(f"linear"):
        out1 = self.linear_trans1(out1)
        for i in range(len(self.out_channels)):
            # with CUDATimer(f"conv {i}"):
            out1 = self.conv1[i](out1)
            # print(f"conv {i} out shape: {out.shape}")
        out1 = torch.mean(out1, dim=(2, 3)) # (b, out_channels[-1])

        out2 = self.linear_trans2(out2)
        for i in range(len(self.out_channels2)):
            out2 = self.conv2[i](out2)
        out2 = torch.mean(out2, dim=(2, 3)) # (b, out_channels[-1])

        out = torch.cat([out1, out2], dim=-1) # (b, 2*out_channels[-1])
        return out


def get_cmdtop_params_from_diameter(x, postfix=""):
    if x == 9:
        # size will be 9 -> 5 -> 3 -> 1
        cmdtop_params = {
            "in_channel": 64,
            "out_channels": (64, 128, 128),
            "kernel_shapes": (3, 3, 3),
            "strides": (2, 2, 2),
            "paddings": (1, 1, 0),
        }
    elif x == 7:
        # size will be 7 -> 4 -> 2 -> 1
        cmdtop_params = {
            "in_channel": 64,
            "out_channels": (64, 128, 128),
            "kernel_shapes": (3, 3, 2),
            "strides": (2, 2, 2),
            "paddings": (1, 1, 0),
        }
    elif x == 3:
        # size will be 3 -> 3 -> 3 -> 1
        # cmdtop_params = {
        #     "in_channel": 64,
        #     "out_channels": (64, 128, 128),
        #     "kernel_shapes": (3, 3, 3),
        #     "strides": (1, 1, 1),
        #     "paddings": (1, 1, 0),
        # }
        cmdtop_params = {
            "in_channel": 64,
            "out_channels": (128, ),
            "kernel_shapes": (3, ),
            "strides": (1, ),
            "paddings": (0, ),
        }
    elif x == 1:
        # size will be 1 -> 1 -> 1 -> 1
        cmdtop_params = {
            "in_channel": 64,
            "out_channels": (64, 128, 128),
            "kernel_shapes": (1, 1, 1),
            "strides": (1, 1, 1),
            "paddings": (0, 0, 0),
        }
    
    elif x == 15:
        # size will be 15 -> 7 -> 3 -> 1
        cmdtop_params = {
            "in_channel": 64,
            "out_channels": (64, 128, 128),
            "kernel_shapes": (3, 3, 3),
            "strides": (2, 2, 1),
            "paddings": (0, 0, 0),
        }
    
    elif x == 25:
        # size will be 25 -> 12 -> 6 -> 1
        cmdtop_params = {
            "in_channel": 2,
            "out_channels": (8, 16, 128),
            "kernel_shapes": (3, 3, 6),
            "strides": (2, 2, 1),
            "paddings": (0, 1, 0),
        }
    
    elif x == 29:
        # size will be 29 -> 14 -> 7 -> 1
        cmdtop_params = {
            "in_channel": 2,
            "out_channels": (8, 16, 128),
            "kernel_shapes": (3, 3, 7),
            "strides": (2, 2, 1),
            "paddings": (0, 1, 0),
        }
    
    elif x == 49:
        # size will be 49 -> 24 -> 11 -> 1
        cmdtop_params = {
            "in_channel": 2,
            "out_channels": (8, 16, 128),
            "kernel_shapes": (3, 3, 11),
            "strides": (2, 2, 1),
            "paddings": (0, 0, 0),
        }
    else:
        raise

    new_d = {}
    for k, v in cmdtop_params.items():
        new_d[f"{k}{postfix}"] = v
    return new_d