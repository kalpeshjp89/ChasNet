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
            act_type='sigmoid', mode=mode)
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5)

    def forward(self, x):
        x = self.features(x)
        return x