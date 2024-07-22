from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch

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



# ResNet-50/101/152 残差结构 Bottleneck
class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,   # out_channels是第一、二层的卷积核个数
                               kernel_size=1, stride=1, bias=False)  # squeeze channels  高和宽不变
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, bias=False, padding=1)   # 实线stride为1，虚线stride为2
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,    # 卷积核个数为4倍
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    # 正向传播过程
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class Netv2(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        filters = [64, 128, 256, 512,1024]

        self.firstconv = nn.Conv2d(1, 64,kernel_size=1)
        self.firstbn = nn.BatchNorm2d(64)
        self.firstrelu = nn.ReLU(False)
        self.encoder1 = self._make_layer(block, 64,128, layers[0])
        self.encoder2 = self._make_layer(block, 128,256, layers[1])
        self.encoder3 = self._make_layer(block, 256, 512,layers[2])
        self.encoder4 = self._make_layer(block, 512,1024, layers[3])
        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[4], filters[3])
        self.decoder3 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder2 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder1 = DecoderBlockLinkNet(filters[1], filters[0])
        self.gau3 = AAM(filters[3], filters[3])  # RAUNet
        self.gau2 = AAM(filters[2], filters[2])
        self.gau1 = AAM(filters[1], filters[1])

        # Final Classifier
        self.finalconv1 = nn.Conv2d(64, 64, 1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(64, num_classes, 1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d3 = self.decoder4(e4)
        b3 = self.gau3(d3, e3)
        d2 = self.decoder3(b3)
        b2 = self.gau2(d2, e2)
        d1 = self.decoder2(b2)
        b1 = self.gau1(d1, e1)
        d0 = self.decoder1(b1)
        #
        # Final Classification
        f1 = self.finalconv1(d0)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)


        # if self.num_classes > 1:
        #     x_out = F.log_softmax(f5, dim=1)
        # else:
        x_out = f3
        return x_out
    def _make_layer(self, block,in_channel,channel, blocks):
        downsample = nn.Conv2d(in_channel, channel, stride=2, kernel_size=1, bias=False)
        layers = []
        for _ in range(1, blocks):
            layers.append(block(in_channel, in_channel))
        layers.append(downsample)
        return nn.Sequential(*layers)


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x
if __name__ == '__main__':   #测试模块
# #     # input_low = torch.randn(2,64, 512, 512)
# #     # input_high = torch.randn(2,128, 224, 224)
# #     # # AAMmodel = AAM(32,64)
# #     # # # resnet101=ResNet101(1000)
# #     # # # resnet152=ResNet152(1000)
# #     # # out = AAMmodel(input_high,input_low)
# #     # convmodel1 = BottleNeck(64,32,downsample = nn.Conv2d(64, 32 * 4, stride=2, kernel_size=1,
# #     #                                     bias=False),stride=2)
# #     # convmodel2 = BottleNeck(128, 32)
# #     # # # resnet101=ResNet101(1000)
# #     # # # resnet152=ResNet152(1000)
# #     # out1 = convmodel1(input_low)
# #     # print(out1.shape)
# #     # out2 = convmodel2(out1)
# #     # print(out2.shape)
# #
    input = torch.randn(4, 1, 512, 512)
    # net = BottleNeck(64,64)
    net = Netv2(BottleNeck,[3,4,4,3])
    out = net(input)
    print("s",out.shape)

