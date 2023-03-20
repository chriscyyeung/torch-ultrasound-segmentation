import torch
import torch.nn as nn
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_stages=1, kernel_size=3, stride=1, padding=1):
        super().__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                ops.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            else:
                ops.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
            ops.append(nn.BatchNorm2d(out_channels))
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, first_channels, n_classes):
        super().__init__()

        # Encoder
        self.conv1 = ConvBlock(in_channels, first_channels, n_stages=2)
        self.down1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(first_channels, first_channels * 2, n_stages=2)
        self.down2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(first_channels * 2, first_channels * 4, n_stages=2)
        self.down3 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv4 = ConvBlock(first_channels * 4, first_channels * 8, n_stages=2)

        # Decoder (output_padding=1 for odd-sized inputs)
        self.up1 = nn.ConvTranspose2d(first_channels * 8, first_channels * 4, 2, stride=2, output_padding=1)
        self.conv5 = ConvBlock(first_channels * 8, first_channels * 4, n_stages=2)
        self.up2 = nn.ConvTranspose2d(first_channels * 4, first_channels * 2, 2, stride=2)
        self.conv6 = ConvBlock(first_channels * 4, first_channels * 2, n_stages=2)
        self.up3 = nn.ConvTranspose2d(first_channels * 2, first_channels, 2, stride=2)
        self.conv7 = ConvBlock(first_channels * 2, first_channels, n_stages=2)

        # Final 1x1 convolution and activation
        self.final_conv = nn.Conv2d(first_channels, n_classes, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        d1 = self.down1(x1)
        x2 = self.conv2(d1)
        d2 = self.down2(x2)
        x3 = self.conv3(d2)
        d3 = self.down3(x3)

        x = self.conv4(d3)

        x = self.up1(x)
        x = self.conv5(torch.cat([x3, x], 1))
        x = self.up2(x)
        x = self.conv6(torch.cat([x2, x], 1))
        x = self.up3(x)
        x = self.conv7(torch.cat([x1, x], 1))

        x = self.final_conv(x)
        return x


if __name__ == '__main__':
    model = UNet(1, 64, 1)
    summary(model, (1, 300, 300))
