from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.ops as ops
####################
# Basic blocks
####################
class PA(nn.Module):
    def __init__(self,in_nc):
        super(PA, self).__init__()
        self.c1 = conv_block(in_nc, in_nc, kernel_size=1, norm_type=None, act_type='leakyrelu')

    def forward(self, x):
        x1 = self.c1(x)
        out = x.mul(x1) + x
        return out
class CA(nn.Module):
    def __init__(self,in_nc):
        super(CA, self).__init__()
        self.g = nn.AdaptiveAvgPool2d((1,1))
        self.c1 = conv_block(in_nc, 16, kernel_size=1, norm_type=None, act_type='prelu')
        self.c2 = conv_block(16, in_nc, kernel_size=1, norm_type=None, act_type='sigm')
        self.w1 = nn.Parameter(torch.FloatTensor([1]))

    def forward(self, x):
        g1 = self.g(x)
        b=torch.std(x.view(x.size(0),x.size(1),-1),2)
        soc=b.view(b.size(0),b.size(1),1,1)
        x1 = self.w1*g1 + (1-self.w1)*soc
        atn = self.c2(self.c1(x1))
        out = x.mul(atn)
        return out

class Residual(nn.Module):

    def __init__(self, nf):
        super(Residual, self).__init__()
        self.conv1 = conv_block(nf, nf, kernel_size=3, norm_type='batch', act_type='leakyrelu')
        self.conv2 = conv_block(nf, nf, kernel_size=3, act_type='leakyrelu',dilation=2)
        self.pa = PA(nf)
        self.ca = CA(nf)

    def forward(self, x):
        x1 = self.ca(self.pa(self.conv2(self.conv1(x))))
        #out = x.mul(self.w) + x1
        out = x + x1
        return out

def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'sigm':
        layer = nn.Sigmoid()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding



def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)



####################
# Upsampler
####################
class OurUpSample(nn.Module):
    def __init__(self,in_nc, nc, kernel_size=3, stride=1, bias=True, pad_type='zero', \
            act_type=None, mode='CNA',upscale_factor=2):
        super(OurUpSample, self).__init__()
        self.U1 = pixelshuffle_block(in_nc, nc, upscale_factor=upscale_factor, kernel_size=3, norm_type = 'batch')
        self.co1 = conv_block(nc, 16, kernel_size=1, norm_type=None, act_type='prelu', mode='CNA')
        self.co2 = conv_block(16, 3, kernel_size=3, norm_type=None, act_type='prelu', mode='CNA')

    def forward(self, x):
        out1 = self.U1(x)
        return self.co2(self.co1(out1))

def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)


class CSB(nn.Module):
    def __init__(self,in_nc, nc, stride=1, bias=True, kernel_size=3, pad_type='zero', \
            norm_type=None, act_type=None, mode='CNA'):
        super(CSB, self).__init__()
        self.nf = int(0.5*nc)
        self.conv0 = conv_block(in_nc, nc, kernel_size=1, norm_type='batch', act_type=act_type)
        self.conv1 = conv_block(self.nf, self.nf, kernel_size=kernel_size, norm_type=None, act_type=act_type)
        self.conv2 = conv_block(2*self.nf, self.nf, kernel_size=kernel_size, norm_type=None, act_type=act_type)
        self.conv3 = conv_block(3*self.nf, self.nf, kernel_size=kernel_size, norm_type=None, act_type=act_type)
        self.res1 = Residual(self.nf)
        self.res2 = Residual(self.nf)

    def forward(self, x):
        x0 = self.conv0(x)
        x11 = x0[:,0:self.nf,:,:]
        x12 = x0[:,self.nf:,:,:]
        x1 = self.conv1(x11)
        x2 = self.conv2(torch.cat((x1,x11),1))
        x3 = self.conv3(torch.cat((x2,x1,x11),1))
        x4 = self.res2(self.res1(x12))
        out = torch.cat((x3,x4),1)
        return out

class BB(nn.Module):
    def __init__(self,nf):
        super(BB, self).__init__()
        self.b11=CSB(nf,nf,kernel_size=3, act_type='leakyrelu')
        self.b12=CSB(2*nf,nf,kernel_size=3, act_type='leakyrelu')
        self.b13=CSB(3*nf,nf,kernel_size=3, act_type='leakyrelu')
        self.c1 = conv_block(4*nf, 2*nf, kernel_size=1, act_type='leakyrelu')

    def forward(self, x):
        xa1 = self.b11(x)
        xa2 = self.b12(torch.cat((x,xa1),1))
        xa3 = self.b13(torch.cat((x,xa1,xa2),1))
        out = self.c1(torch.cat((x,xa1,xa2,xa3),1))
        return out


#Remove spliting but use 1x1 to reduce channel instead of CS
class VKGen(nn.Module):
    def __init__(self,in_nc, nf):
        super(VKGen, self).__init__()
        self.nf=nf
        self.cn1= conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
        self.cn2= conv_block(nf, 2*nf, kernel_size=5, norm_type=None, act_type=None, mode='CNA')

        self.b1=BB(nf)
        self.b2=BB(nf)
        self.b3=BB(nf)
        self.b4=BB(nf)
        self.b5=BB(nf)
        self.b6=BB(nf)
        self.b7=BB(nf)
        self.b8=BB(nf)
        self.c1=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c2=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c3=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c4=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c5=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c6=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c7=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c8=conv_block(3*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')

        self.grl= nn.Upsample(scale_factor=2, mode='bicubic')
        
        self.ups1= OurUpSample(nf,nf, kernel_size=3, act_type='leakyrelu',upscale_factor=2)
        #self.ups2= OurUpSample(nf*4,nf, kernel_size=3, act_type='leakyrelu',upscale_factor=4)

    def forward(self, x):
        x1 = self.cn2(self.cn1(x))
        x11 = x1[:,0:self.nf,:,:]
        x12 = x1[:,self.nf:,:,:]
        
        xa1 = self.b1(x11)
        xa11 = xa1[:,0:self.nf,:,:]
        xa12 = xa1[:,self.nf:,:,:]

        xb1 = self.b2(xa11)
        xb11 = xb1[:,0:self.nf,:,:]
        xb12 = xb1[:,self.nf:,:,:]

        xc1 = self.b3(xb11)
        xc11 = xc1[:,0:self.nf,:,:]
        xc12 = xc1[:,self.nf:,:,:]

        xd1 = self.b4(xc11)
        xd11 = xd1[:,0:self.nf,:,:]
        xd12 = xd1[:,self.nf:,:,:]

        xe1 = self.b5(xd11)
        xe11 = xe1[:,0:self.nf,:,:]
        xe12 = xe1[:,self.nf:,:,:]

        xf1 = self.b6(xe11)
        xf11 = xf1[:,0:self.nf,:,:]
        xf12 = xf1[:,self.nf:,:,:]

        xg1 = self.b7(xf11)
        xg11 = xg1[:,0:self.nf,:,:]
        xg12 = xg1[:,self.nf:,:,:]

        xh1 = self.b8(xg11)
        xh11 = xh1[:,0:self.nf,:,:]
        xh12 = xh1[:,self.nf:,:,:]

        xr1 = self.c1(torch.cat((xa12,xb12),1))
        xr2 = self.c2(torch.cat((xr1,xc12),1))
        xr3 = self.c3(torch.cat((xr2,xd12),1))
        xr4 = self.c4(torch.cat((xr3,xe12),1))
        xr5 = self.c5(torch.cat((xr4,xf12),1))
        xr6 = self.c6(torch.cat((xr5,xg12),1))
        xr7 = self.c7(torch.cat((xr6,xh12),1))
        xr = self.c8(torch.cat((xh11,xr7,x12),1))

        u1 = self.ups1(xr)
        u2 = self.grl(x)
        
        out = u1+u2
        return out

class VKGen2(nn.Module):
    def __init__(self,in_nc, nf):
        super(VKGen2, self).__init__()
        self.nf=nf
        self.cn1= conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
        self.cn2= conv_block(nf, 2*nf, kernel_size=5, norm_type=None, act_type=None, mode='CNA')

        self.b1=BB(nf)
        self.b2=BB(nf)
        self.b3=BB(nf)
        self.b4=BB(nf)
        self.b5=BB(nf)
        self.b6=BB(nf)
        self.b7=BB(nf)
        self.b8=BB(nf)
        self.c1=conv_block(3*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        #self.c2=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        #self.c3=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        #self.c4=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        #self.c5=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        #self.c6=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        #self.c7=conv_block(2*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        #self.c8=conv_block(3*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')

        self.c9=conv_block(3*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c10=conv_block(3*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c11=conv_block(3*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')
        self.c12=conv_block(3*nf, nf, kernel_size=1, norm_type=None, act_type='leakyrelu')

        self.grl= nn.Upsample(scale_factor=2, mode='bicubic')
        self.ds= nn.Upsample(scale_factor=0.5, mode='bicubic')
        
        self.ups1= OurUpSample(nf,nf, kernel_size=3, act_type='leakyrelu',upscale_factor=2)
        #self.ups2= OurUpSample(nf*4,nf, kernel_size=3, act_type='leakyrelu',upscale_factor=4)

    def forward(self, x):
        x1 = self.cn2(self.cn1(x))
        x11 = x1[:,0:self.nf,:,:]
        x12 = x1[:,self.nf:,:,:]
        
        xa1 = self.b1(x11)
        xa11 = xa1[:,0:self.nf,:,:]
        xa12 = xa1[:,self.nf:,:,:]

        xb1 = self.b2(self.ds(xa11))
        xb11 = xb1[:,0:self.nf,:,:]
        xb12 = xb1[:,self.nf:,:,:]

        xc1 = self.b3(self.ds(xb11))
        xc11 = xc1[:,0:self.nf,:,:]
        xc12 = xc1[:,self.nf:,:,:]

        xd1 = self.b4(xc11)

        
        xe1 = self.b5(self.c9(torch.cat((xd1,xc12),1)))

        xf1 = self.b6(self.c10(torch.cat((xb12,self.grl(xe1)),1)))

        xg1 = self.b7(self.c11(torch.cat((xa12,self.grl(xf1)),1)))
        xg11 = xg1[:,0:self.nf,:,:]
        xg12 = xg1[:,self.nf:,:,:]

        xh1 = self.b8(xg11)

        xr1 = self.c1(torch.cat((xg12,xh1),1))

        u1 = self.ups1(xr1)
        #u2 = self.grl(x)
        
        out = u1
        return out