"""Zero-DCE++ (Li et al., 2021) -- reimplemented from the official reference
implementation at github.com/Li-Chongyi/Zero-DCE_extension (Zero-DCE++/model.py).
Weights: Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth (10,561 parameters).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CSDN_Tem(nn.Module):
    """Depth-wise separable conv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1,
                                    padding=1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class ZerodcePP(nn.Module):
    """enhance_net_nopool, official scale_factor = 12."""

    def __init__(self, scale_factor: int = 12, number_f: int = 32):
        super().__init__()
        self.scale_factor = scale_factor
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, number_f)
        self.e_conv4 = CSDN_Tem(number_f, number_f)
        self.e_conv5 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv6 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv7 = CSDN_Tem(number_f * 2, 3)

    def _curve(self, x, x_r):
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        out = x + x_r * (torch.pow(x, 2) - x)
        x = out + x_r * (torch.pow(out, 2) - out)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        return x + x_r * (torch.pow(x, 2) - x)

    def forward(self, x):
        x_down = x if self.scale_factor == 1 else F.interpolate(
            x, scale_factor=1 / self.scale_factor, mode="bilinear")
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor != 1:
            x_r = self.upsample(x_r)
        if x_r.shape[-2:] != x.shape[-2:]:
            # keep official behaviour at exact multiples, but adapt to any size
            x_r = F.interpolate(x_r, size=x.shape[-2:], mode="bilinear", align_corners=True)
        return self._curve(x, x_r)
