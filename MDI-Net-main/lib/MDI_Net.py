import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from InceptionNext import inceptionnext_tiny
up_kwargs = {'mode': 'bilinear', 'align_corners': False}

from torchsummary import summary


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    return param_size


class MDI_Net(nn.Module):
    def __init__(self, out_planes=1, encoder='inceptionnext_tiny'):
        super(MDI_Net, self).__init__()
        self.encoder = encoder
        if self.encoder == 'inceptionnext_tiny':
            mutil_channel = [96, 192, 384, 768]
            self.backbone = inceptionnext_tiny()

        self.dropout = torch.nn.Dropout(0.3)  
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.mlfa1 = MLFA(mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3], lenn=1)
        self.mlfa2 = MLFA(mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3], lenn=1)
        self.mlfa3 = MLFA(mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3], lenn=1)


        self.decoder4 = BasicConv2d(mutil_channel[3], mutil_channel[2], 3, padding=1)
        self.decoder3 = BasicConv2d(mutil_channel[2], mutil_channel[1], 3, padding=1)
        self.decoder2 = BasicConv2d(mutil_channel[1], mutil_channel[0], 3, padding=1)
        self.decoder1 = nn.Sequential(nn.Conv2d(mutil_channel[0], 64, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(64, out_planes, kernel_size=1, stride=1))

        self.fu1 = DGIA(96, 192,  96)
        self.fu2 = DGIA(192, 384, 192)
        self.fu3 = DGIA(384, 768,  384)
    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        '''
        x1 : (2,96,56,56)
        x2 : (2,192,28,28)
        x3 : (2,384,14,14)
        x4 : (2,768,7,7)
        '''

        x1, x2, x3, x4 = self.mlfa1(x1, x2, x3, x4)
        x1, x2, x3, x4 = self.mlfa2(x1, x2, x3, x4)
        x1, x2, x3, x4 = self.mlfa3(x1, x2, x3, x4)

        x_f_3 = self.fu3(x3, x4)
        x_f_2 = self.fu2(x2, x_f_3)
        x_f_1 = self.fu1(x1, x_f_2)

        d1 = self.decoder1(x_f_1)
        d1 = self.dropout(d1)  
        d1 = F.interpolate(d1, scale_factor=4, mode='bilinear')  # (1,1,224,224)
        return d1


class DGIA(nn.Module):
    def __init__(self, l_dim, g_dim, out_dim):
        super(DGIA,self).__init__()
        self.extra_l = LKFE(l_dim)
        self.bn = nn.BatchNorm2d(out_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3x3 = BasicConv2d(g_dim, out_dim, 3, padding=1)
        self.selection = nn.Conv2d(out_dim, 1, 1)
        self.proj = nn.Sequential(
            nn.Conv2d(2, 1, 1, 1),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self,l,g):
        l = self.extra_l(l)
        g = self.conv3x3(self.upsample(g))
        weight = self.avg_pool(g)  # (1,384,14,14)
        output = l * weight + g
        return output


class LKFE(nn.Module):
    def __init__(self, dim):
        super(LKFE, self).__init__()
        self.conv0 = nn.Conv2d(2*dim//3, 2*dim//3, 5, padding=2, groups=2*dim//3)
        self.conv_spatial = nn.Conv2d(2*dim//3, 2*dim//3, 7, stride=1, padding=9, groups=2*dim//3, dilation=3)
        self.conv1 = nn.Conv2d(2*dim//3, dim // 3, 1)
        self.conv2 = nn.Conv2d(2*dim//3, dim // 3, 1)

        self.split_indexes = (dim // 3, 2*dim//3)
        self.branch1 = nn.Sequential()
        self.conv1x1 = nn.Sequential(

            nn.Conv2d(in_channels=2*dim//3, out_channels=2*dim//3, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2*dim//3),  
            nn.ReLU(inplace=True))
        self.norm =nn.BatchNorm2d(dim)
    def forward(self, x):
        x_id, x_k= torch.split(x, self.split_indexes, dim=1)
        attn1 = self.conv0(x_k)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        x_id = self.branch1(x_id)
        attn = torch.cat((x_id, attn1, attn2), dim=1)
        out = channel_shuffle(attn, 2)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # (1,768,14,14)
        x = self.conv(x) # (1,384,14,14)
        x = self.bn(x)
        return x



def channel_shuffle(x, groups): 
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)  
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

class Conv2d_batchnorm(torch.nn.Module):
    """
    2D Convolutional layers
    """

    def __init__(
            self,
            num_in_filters,  
            num_out_filters, 
            kernel_size,
            stride=(1, 1), 
            activation="LeakyReLU",
    ):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            groups=8
        )
        self.num_in_filters =num_in_filters
        self.num_out_filters = num_out_filters
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):  # (2,1792,56,56)  1920->1792
        x=channel_shuffle(x,gcd(self.num_in_filters,self.num_out_filters))
        x = self.conv1(x)  # (2,128,56,56)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.sqe(x)


class Conv2d_channel(torch.nn.Module):
    """
    2D pointwise Convolutional layers
    """

    def __init__(self, num_in_filters, num_out_filters):

        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=(1, 1),
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        return self.sqe(self.activation(x))


class ChannelSELayer(torch.nn.Module):
    def __init__(self, num_channels):
        """
        Initialization

        Args:
            num_channels (int): No of input channels
        """

        super(ChannelSELayer, self).__init__()

        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1) 

        self.reduction_ratio = 8  # default reduction ratio

        num_channels_reduced = num_channels // self.reduction_ratio
        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()

        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)

    def forward(self, inp): 

        batch_size, num_channels, H, W = inp.size()
        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))
        out = self.bn(out)
        out = self.act(out)

        return out


class MLFA(torch.nn.Module):
    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, lenn=1):
        super().__init__()
        self.in_filters1 = in_filters1
        self.in_filters2 = in_filters2
        self.in_filters3 = in_filters3
        self.in_filters4 = in_filters4
        self.in_filters = (
                in_filters1 + in_filters2 + in_filters3 + in_filters4
        )  # total number of channels
        self.in_filters3_4 = (
                in_filters3 + in_filters4
        )
        self.in_filters2_3_4 = (
                in_filters2 + in_filters3 + in_filters4
        )

        self.no_param_up = torch.nn.Upsample(scale_factor=2)  # used for upsampling

        self.no_param_down = torch.nn.AvgPool2d(2)  # used for downsampling

        self.cnv_blks1 = torch.nn.ModuleList([])
        self.cnv_blks2 = torch.nn.ModuleList([])
        self.cnv_blks3 = torch.nn.ModuleList([])
        self.cnv_blks4 = torch.nn.ModuleList([])

        self.cnv_mrg1 = torch.nn.ModuleList([])
        self.cnv_mrg2 = torch.nn.ModuleList([])
        self.cnv_mrg3 = torch.nn.ModuleList([])
        self.cnv_mrg4 = torch.nn.ModuleList([])

        self.bns1 = torch.nn.ModuleList([])
        self.bns2 = torch.nn.ModuleList([])
        self.bns3 = torch.nn.ModuleList([])
        self.bns4 = torch.nn.ModuleList([])

        self.bns_mrg1 = torch.nn.ModuleList([])
        self.bns_mrg2 = torch.nn.ModuleList([])
        self.bns_mrg3 = torch.nn.ModuleList([])
        self.bns_mrg4 = torch.nn.ModuleList([])

        for i in range(lenn):
            self.cnv_blks1.append(
                Conv2d_batchnorm(self.in_filters, in_filters1, (1, 1))
            )
            self.cnv_mrg1.append(Conv2d_batchnorm(in_filters1, in_filters1, (1, 1)))
            self.bns1.append(torch.nn.BatchNorm2d(in_filters1))
            self.bns_mrg1.append(torch.nn.BatchNorm2d(in_filters1))

            self.cnv_blks2.append(
                Conv2d_batchnorm(self.in_filters2_3_4, in_filters2, (1, 1))
            )
            self.cnv_mrg2.append(Conv2d_batchnorm(2 * in_filters2, in_filters2, (1, 1)))
            self.bns2.append(torch.nn.BatchNorm2d(in_filters2))
            self.bns_mrg2.append(torch.nn.BatchNorm2d(in_filters2))

            self.cnv_blks3.append(
                Conv2d_batchnorm(self.in_filters3_4, in_filters3, (1, 1))
            )
            self.cnv_mrg3.append(Conv2d_batchnorm(2 * in_filters3, in_filters3, (1, 1)))
            self.bns3.append(torch.nn.BatchNorm2d(in_filters3))
            self.bns_mrg3.append(torch.nn.BatchNorm2d(in_filters3))

        self.act = torch.nn.LeakyReLU()

        self.sqe1 = ChannelSELayer(in_filters1)
        self.sqe2 = ChannelSELayer(in_filters2)
        self.sqe3 = ChannelSELayer(in_filters3)
        self.sqe4 = ChannelSELayer(in_filters4)

    def forward(self, x1, x2, x3, x4):

        batch_size, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape
        _, _, h3, w3 = x3.shape

        for i in range(len(self.cnv_blks1)):
            x_c1 = self.act(
                self.bns1[i](
                    self.cnv_blks1[i](
                        torch.cat(
                            [
                                x1,
                                self.no_param_up(x2),
                                self.no_param_up(self.no_param_up(x3)),
                                self.no_param_up(self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,  
                        )
                    )
                )
            )  
            x_c2 = self.act(
                self.bns2[i](
                    self.cnv_blks2[i](
                        torch.cat(
                            [
                                x2,
                                (self.no_param_up(x3)),
                                (self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c3 = self.act(
                self.bns3[i](
                    self.cnv_blks3[i](
                        torch.cat(
                            [
                                x3,
                                (self.no_param_up(x4)),
                            ],
                            dim=1,
                        )
                    )
                )
            )

            x_c1 = self.act(
                self.bns_mrg1[i](
                    torch.mul(x_c1, x1).view(batch_size, self.in_filters1, h1, w1) + x1
                )
            )
            x_c2 = self.act(
                self.bns_mrg2[i](
                    torch.mul(x_c2, x2).view(batch_size, self.in_filters2, h2, w2) + x2
                )
            )
            x_c3 = self.act(
                self.bns_mrg3[i](
                    torch.mul(x_c3, x3).view(batch_size, self.in_filters3, h3, w3) + x3
                )
            )

        x1 = self.sqe1(x_c1)
        x2 = self.sqe2(x_c2)
        x3 = self.sqe3(x_c3)

        return x1, x2, x3, x4



