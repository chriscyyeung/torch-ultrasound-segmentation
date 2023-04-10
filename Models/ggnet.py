# Adapted from https://github.com/xorangecheng/GlobalGuidance-Net

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights


class GGNet(nn.Module):
    def __init__(self, num_classes=1, freeze_bn=False):
        super().__init__()

        self.backbone = Encoder()
        self.aspp = ASPP()
        self.decoder = Decoder(num_classes)
        self.predict5 = nn.Conv2d(256, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, layer1, low_level_feat, pre1, pre2, pre3, pre4 = self.backbone(input)
        
        x = self.aspp(x)
        pre5 = self.predict5(F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True))
        x = self.decoder(x, low_level_feat, layer1)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        smoothed_4 = self.pool(x)
        edge4 = torch.abs(x - smoothed_4)
        o0_e = x + edge4

        return o0_e, pre1, pre2, pre3, pre4, pre5

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


# Backbone
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnext = ResNext101()

        self.down0 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU()
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )

        self.predict0 = nn.Conv2d(128, 1, kernel_size=1)
        self.predict3 = nn.Conv2d(128, 1, kernel_size=1)
        self.predict2 = nn.Conv2d(128, 1, kernel_size=1)
        self.predict1 = nn.Conv2d(128, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(3,stride=1,padding=1)

    def forward(self, x):
        layer0 = self.resnext.layer0(x)
        layer1 = self.resnext.layer1(layer0)
        layer2 = self.resnext.layer2(layer1)
        layer3 = self.resnext.layer3(layer2)

        down0 = F.interpolate(self.down0(layer0), size=layer1.size()[2:], mode='bilinear',align_corners=True)
        down3 = F.interpolate(self.down3(layer3), size=layer1.size()[2:], mode='bilinear',align_corners=True)
        down2 = F.interpolate(self.down2(layer2), size=layer1.size()[2:], mode='bilinear',align_corners=True)
        down1 = self.down1(layer1)

        predict0 = self.predict0(down0)
        predict3 = self.predict3(down3)
        predict2 = self.predict2(down2)
        predict1 = self.predict1(down1)

        fuse1 = self.fuse1(torch.cat((down0, down3, down2, down1), 1))

        o1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        o2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        o3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        o0 = F.interpolate(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)

        p4_edge = o0
        smoothed_4 = self.pool(p4_edge)
        edge4 = torch.abs(p4_edge - smoothed_4)
        o0_e = o0 + edge4

        p3_edge = o3
        smoothed_3 = self.pool(p3_edge)
        edge3 = torch.abs(p3_edge - smoothed_3)
        o3_e = o3 + edge3

        p2_edge = o2
        smoothed_2 = self.pool(p2_edge)
        edge2 = torch.abs(p2_edge - smoothed_2)
        o2_e = o2 + edge2

        p1_edge = o1
        smoothed_1 = self.pool(p1_edge)
        edge1 = torch.abs(p1_edge - smoothed_1)
        o1_e = o1 + edge1

        return layer3, layer1, fuse1, o1_e, o2_e, o3_e, o0_e


# Pretrained ResNext101
class ResNext101(nn.Module):
    def __init__(self):
        super().__init__()

        net = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3:5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]
    
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# Atrous spatial pyramid pooling
class ASPP(nn.Module):
    def __init__(self):
        super().__init__()

        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(1024, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(1024, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(1024, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(1024, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(1024, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super().__init__()

        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                        stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        low_level_inplanes = 256
        self.conv1 = nn.Conv2d(256, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        self.dg = DGNLB(256, 256)
   
        self._init_weight()

    def forward(self, x, low_level_feat, layer1):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        layer1 = self.conv2(layer1)
        layer1 = self.bn2(layer1)
        layer1 = self.relu(layer1)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = self.dg(x, low_level_feat)
        x = torch.cat((x, layer1), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DGNLB(nn.Module):
    def __init__(self, in_channels, in_channels_guide):
        super().__init__()

        self.pconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.BatchNorm2d(in_channels), nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.PReLU()
            )
        self.cconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.BatchNorm2d(in_channels), nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.PReLU()
            )
        self.fconv = nn.Sequential(
            nn.Conv2d(in_channels,in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.PReLU()
        )
        self.PAM = G_PAM_Module(in_channels, in_channels_guide)
        self.CAM = G_CAM_Module(in_channels, in_channels_guide)
        self.predict = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x, g):
        pam = self.PAM(x, g)
        pam2 = self.pconv(pam)
        cam = self.CAM(pam2, g)
        cam2 = self.cconv(cam)
        final = self.fconv(cam2)
        return final


# Spatial-wise guidance block
class G_PAM_Module(nn.Module):
    #Ref from SAGAN
    def __init__(self, in_dim, in_dim_guide):
        super().__init__()
        
        self.query_guide = nn.Conv2d(in_channels=in_dim_guide, out_channels=in_dim_guide, kernel_size=1)
        self.key_guide = nn.Conv2d(in_channels=in_dim_guide, out_channels=in_dim_guide, kernel_size=1)
        # self.value_guide=Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x,g):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        gm_batchsize, gC, gheight, gwidth = g.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        proj_query_guide = self.query_guide(g).view(gm_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key_guide = self.key_guide(g).view(gm_batchsize, -1, width * height)
        energy_guide = torch.bmm(proj_query_guide, proj_key_guide)
        attention = self.softmax(energy)
        attention_guide = self.softmax(energy_guide)
        guide_energy = torch.bmm(attention, attention_guide)
        guide_attention = self.softmax(guide_energy)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, guide_attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


# Channel-wise guidance block
class G_CAM_Module(nn.Module):
    def __init__(self, in_channels, in_channels_guide):
        super().__init__()

        self.z = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.seconv = nn.Conv2d(in_channels_guide, in_channels, kernel_size=1)
        self.W = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x,g):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()

        proj_query = x.view(m_batchsize, C, -1)

        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy_new = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy_new)

        proj_query_guide = g.view(m_batchsize, C, -1)
        proj_key_guide = g.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy_guide_new = torch.bmm(proj_query_guide,proj_key_guide)
        attention_guide = self.softmax(energy_guide_new)
        guide_energy = torch.bmm(attention,attention_guide)
        guide_energy_new = torch.max(guide_energy, -1, keepdim=True)[0].expand_as(guide_energy) - guide_energy
        guide_attention = self.softmax(guide_energy_new)

        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(guide_attention,proj_value).contiguous().view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


if __name__ == "__main__":
    net = GGNet()
    # print(net)
    x = torch.randn(32, 3, 256, 256)
    net(x)
