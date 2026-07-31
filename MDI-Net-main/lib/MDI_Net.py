import torch
import torch.nn as nn
import torch.nn.functional as F
from InceptionNext import inceptionnext_tiny

up_kwargs = {'mode': 'bilinear', 'align_corners': False}

# 仅保留基础小模块，MLFA 完全不封装
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ChannelSELayer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.gp_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduction_ratio = 8
        num_reduced = num_channels // self.reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_reduced)
        self.fc2 = nn.Linear(num_reduced, num_channels)
        self.act = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(num_channels)
    def forward(self, inp):
        bs, C, H, W = inp.shape
        y = self.gp_avg_pool(inp).view(bs, C)
        y = self.act(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        out = inp * y.view(bs, C, 1, 1)
        return self.act(self.bn(out))

class Conv2d_batchnorm(nn.Module):
    def __init__(self, nin, nout, ks):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, ks, padding='same', groups=8)
        self.bn = nn.BatchNorm2d(nout)
        self.act = nn.LeakyReLU()
        self.se = ChannelSELayer(nout)
        self.drop = nn.Dropout(0.3)
        self.nin, self.nout = nin, nout
    def forward(self, x):
        g = torch.gcd(self.nin, self.nout)
        x = x.view(x.size(0), g, -1, x.size(2), x.size(3)).transpose(1,2).contiguous().view(x.size())
        x = self.conv(x)
        return self.drop(self.act(self.bn(x)))

class LKFE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.split_indexes = (dim//3, 2*dim//3)
        self.conv0 = nn.Conv2d(2*dim//3, 2*dim//3, 5, padding=2, groups=2*dim//3)
        self.conv_spatial = nn.Conv2d(2*dim//3, 2*dim//3, 7, padding=9, dilation=3, groups=2*dim//3)
        self.conv1 = nn.Conv2d(2*dim//3, dim//3, 1)
        self.conv2 = nn.Conv2d(2*dim//3, dim//3, 1)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(2*dim//3, 2*dim//3, 1, bias=False),
            nn.BatchNorm2d(2*dim//3), nn.ReLU()
        )
        self.norm = nn.BatchNorm2d(dim)
    def forward(self, x):
        x_id, x_k = torch.split(x, self.split_indexes, dim=1)
        a1 = self.conv0(x_k)
        a2 = self.conv_spatial(a1)
        a1 = self.conv1(a1)
        a2 = self.conv2(a2)
        x_id = self.conv1x1(x_id)
        out = torch.cat([x_id, a1, a2], dim=1)
        out = out.view(out.size(0), 2, -1, out.size(2), out.size(3)).transpose(1,2).contiguous().view(out.size())
        return self.norm(out)

class DGIA(nn.Module):
    def __init__(self, l_dim, g_dim, out_dim):
        super().__init__()
        self.extra_l = LKFE(l_dim)
        self.up = nn.Upsample(scale_factor=2, bilinear=True)
        self.conv = BasicConv2d(g_dim, out_dim, 3, padding=1)
        self.avg = nn.AdaptiveAvgPool2d(1)
    def forward(self, l, g):
        l = self.extra_l(l)
        g = self.conv(self.up(g))
        return l * self.avg(g) + g



class MDI_Net(nn.Module):
    def __init__(self, out_planes=1):
        super().__init__()
        self.backbone = inceptionnext_tiny()
        c1,c2,c3,c4 = 96,192,384,768

        self.mlfa1_fusion1 = Conv2d_batchnorm(c1+c2+c3+c4, c1, (1,1))
        self.mlfa1_fusion2 = Conv2d_batchnorm(c2+c3+c4, c2, (1,1))
        self.mlfa1_fusion3 = Conv2d_batchnorm(c3+c4, c3, (1,1))
        self.mlfa1_se1 = ChannelSELayer(c1)
        self.mlfa1_se2 = ChannelSELayer(c2)
        self.mlfa1_se3 = ChannelSELayer(c3)

        self.mlfa2_fusion1 = Conv2d_batchnorm(c1+c2+c3+c4, c1, (1,1))
        self.mlfa2_fusion2 = Conv2d_batchnorm(c2+c3+c4, c2, (1,1))
        self.mlfa2_fusion3 = Conv2d_batchnorm(c3+c4, c3, (1,1))
        self.mlfa2_se1 = ChannelSELayer(c1)
        self.mlfa2_se2 = ChannelSELayer(c2)
        self.mlfa2_se3 = ChannelSELayer(c3)

        self.mlfa3_fusion1 = Conv2d_batchnorm(c1+c2+c3+c4, c1, (1,1))
        self.mlfa3_fusion2 = Conv2d_batchnorm(c2+c3+c4, c2, (1,1))
        self.mlfa3_fusion3 = Conv2d_batchnorm(c3+c4, c3, (1,1))
        self.mlfa3_se1 = ChannelSELayer(c1)
        self.mlfa3_se2 = ChannelSELayer(c2)
        self.mlfa3_se3 = ChannelSELayer(c3)

        self.up = nn.Upsample(scale_factor=2)
        self.down = nn.AvgPool2d(2)

        # DGIA & decoder
        self.fu1 = DGIA(c1,c2,c1)
        self.fu2 = DGIA(c2,c3,c2)
        self.fu3 = DGIA(c3,c4,c3)

        self.dec1 = nn.Sequential(nn.Conv2d(c1,64,3,1,1), nn.ReLU(), nn.Conv2d(64,out_planes,1))
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x1,x2,x3,x4 = self.backbone(x)

        # mlfa1 fusion part
        cat1 = torch.cat([x1, self.up(x2), self.up(self.up(x3)), self.up(self.up(self.up(x4)))], 1)
        f1 = self.mlfa1_fusion1(cat1)
        cat2 = torch.cat([x2, self.up(x3), self.up(self.up(x4))], 1)
        f2 = self.mlfa1_fusion2(cat2)
        cat3 = torch.cat([x3, self.up(x4)], 1)
        f3 = self.mlfa1_fusion3(cat3)

        x1 = self.mlfa1_se1(f1 * x1 + x1)
        x2 = self.mlfa1_se2(f2 * x2 + x2)
        x3 = self.mlfa1_se3(f3 * x3 + x3)

        cat1 = torch.cat([x1, self.up(x2), self.up(self.up(x3)), self.up(self.up(self.up(x4)))], 1)
        f1 = self.mlfa2_fusion1(cat1)
        cat2 = torch.cat([x2, self.up(x3), self.up(self.up(x4))], 1)
        f2 = self.mlfa2_fusion2(cat2)
        cat3 = torch.cat([x3, self.up(x4)], 1)
        f3 = self.mlfa2_fusion3(cat3)

        # x1 = self.mlfa2_se1(f1 * x1 + x1)
        # x2 = self.mlfa2_se2(f2 * x2 + x2)
        # x3 = self.mlfa2_se3(f3 * x3 + x3)

        cat1 = torch.cat([x1, self.up(x2), self.up(self.up(x3)), self.up(self.up(self.up(x4)))], 1)
        f1 = self.mlfa3_fusion1(cat1)
        cat2 = torch.cat([x2, self.up(x3), self.up(self.up(x4))], 1)
        f2 = self.mlfa3_fusion2(cat2)
        cat3 = torch.cat([x3, self.up(x4)], 1)
        f3 = self.mlfa3_fusion3(cat3)

        # x1 = self.mlfa3_se1(f1 * x1 + x1)
        # x2 = self.mlfa3_se2(f2 * x2 + x2)
        # x3 = self.mlfa3_se3(f3 * x3 + x3)

        # DGIA
        x_f3 = self.fu3(x3, x4)
        x_f2 = self.fu2(x2, x_f3)
        x_f1 = self.fu1(x1, x_f2)

        out = F.interpolate(self.drop(self.dec1(x_f1)), scale_factor=4, mode='bilinear')
        return out
