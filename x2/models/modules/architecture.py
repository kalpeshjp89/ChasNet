import math
import torch
import torch.nn as nn
import torchvision
from . import block as B
from . import spectral_norm as SN

####################
# Generator
####################
class OurGen(nn.Module):
    def __init__(self, in_nc, nf):
        super(OurGen, self).__init__()
        
        self.model = B.VKGen(in_nc, nf)

    def forward(self,x):
        x = self.model(x)
        return x

class OurGen2(nn.Module):
    def __init__(self, in_nc, nf):
        super(OurGen2, self).__init__()
        
        self.model = B.VKGen2(in_nc, nf)

    def forward(self,x):
        x = self.model(x)
        return x

class Patch_Discriminator(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Patch_Discriminator, self).__init__()
        # 192, 64 (12,512)
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=4, stride=2, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, 2*base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 96, 64 (6,64)
        conv2 = B.conv_block(2*base_nf, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*4, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 128 (3,128)
        conv4 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*8, 1, kernel_size=4, norm_type=None, \
            act_type='sigm', mode=mode)
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5)

    def forward(self, x):
        x = self.features(x)
        return x

class DiscGAN(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(DiscGAN, self).__init__()
        # 192, 64 (12,512)
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=3, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 96, 64 (6,64)
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=3, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 128 (3,128)
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=3, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 256 (2,256)
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 512 (1,512)
        
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7)

        # classifier
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(base_nf*8, 64), nn.LeakyReLU(0.2, True), nn.Linear(64, 1))

    def forward(self, x):
        x = self.gap(self.features(x))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x