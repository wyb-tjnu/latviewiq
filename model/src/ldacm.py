import torch
import torch.nn as nn
import math


class LDACM(nn.Module):

    def __init__(self, in_channels=3):
        super(LDACM, self).__init__()

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial + latitude attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels + 1, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        B, C, H, W = x.shape
        device = x.device

        lat = torch.linspace(-0.5 * math.pi, 0.5 * math.pi, H, device=device)
        lat_weight = torch.cos(lat).view(1, 1, H, 1).expand(B, 1, H, W)

        x_c = x * self.channel_att(x)

        spatial_input = torch.cat([x_c, lat_weight], dim=1)
        spatial_weight = self.spatial_att(spatial_input)

        out = x + x * spatial_weight * lat_weight
        return out

