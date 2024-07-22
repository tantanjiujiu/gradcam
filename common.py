import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class CA(nn.Module):
    # Coordinate Attention for Efficient Mobile Network Design
    '''
        Recent studies on mobile network design have demonstrated the remarkable effectiveness of channel attention (e.g., the Squeeze-and-Excitation attention) for lifting
    model performance, but they generally neglect the positional information, which is important for generating spatially selective attention maps. In this paper, we propose a
    novel attention mechanism for mobile iscyy networks by embedding positional information into channel attention, which
    we call “coordinate attention”. Unlike channel attention
    that transforms a feature tensor to a single feature vector iscyy via 2D global pooling, the coordinate attention factorizes channel attention into two 1D feature encoding
    processes that aggregate features along the two spatial directions, respectively
    '''

    def __init__(self, inp, oup, reduction=32):
        super(CA, self).__init__()

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        pool_h = nn.AdaptiveAvgPool2d((h, 1))
        pool_w = nn.AdaptiveAvgPool2d((1, w))
        x_h = pool_h(x)
        x_w = pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
class AAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AAM, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.Softmax(dim=1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, input_high, input_low):
        mid_high = self.global_pooling(input_high)
        weight_high = self.conv1(mid_high)

        mid_low = self.global_pooling(input_low)
        weight_low = self.conv2(mid_low)

        weight = self.conv3(weight_low + weight_high)
        low = self.conv4(input_low)
        return input_high + low.mul(weight)


class slim_conv_3x3(nn.Module):

    def __init__(self, in_planes, stride, groups, dilation):
        super(slim_conv_3x3, self).__init__()
        self.stride = stride

        reduce_1 = 2
        reduce_2 = 4

        self.conv2_2 = nn.Sequential(nn.Conv2d(in_planes // reduce_1, in_planes // reduce_2, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(in_planes // reduce_2),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_planes // reduce_2, in_planes // reduce_2, kernel_size=3,
                                               stride=stride, groups=groups, padding=dilation, bias=False,
                                               dilation=dilation),
                                     nn.BatchNorm2d(in_planes // reduce_2))

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_planes // reduce_1, in_planes // reduce_1, kernel_size=3, stride=stride, groups=groups,
                      padding=dilation, bias=False, dilation=dilation),
            nn.BatchNorm2d(in_planes // reduce_1))

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 32, kernel_size=1, bias=False),
                                nn.BatchNorm2d(in_planes // 32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_planes // 32, in_planes, kernel_size=1),
                                nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = x
        b, c, h, _ = out.size()

        w = self.pool(out)
        w = self.fc(w)
        w_f = torch.flip(w, [1])

        out1 = w * out
        out2 = w_f * out
        fs1 = torch.split(out1, c // 2, 1)
        fs2 = torch.split(out2, c // 2, 1)

        ft1 = fs1[0] + fs1[1]
        ft2 = fs2[0] + fs2[1]

        out2_1 = self.conv2_1(ft1)
        out2_2 = self.conv2_2(ft2)

        out = torch.cat((out2_1, out2_2), 1)
        return out
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


