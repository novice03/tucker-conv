import torch.nn as nn
from timm.models.efficientnet_blocks import make_divisible
from timm.models.layers import create_conv2d

class TuckerConv(nn.Module):
    def __init__(self, in_chs, out_chs, in_comp_ratio = 0.25, out_comp_ratio = 0.75, act_layer = nn.ReLU6, 
                     norm_layer = nn.BatchNorm2d, comp_kernel_size = 1, reg_kernel_size = 3, pad_type = '', residual = True):
        super(TuckerConv, self).__init__()
        self.residual = residual
        comp_chs = make_divisible(in_comp_ratio * in_chs)
        reg_chs = make_divisible(out_comp_ratio * out_chs)
        
        # Point - wise compression
        self.conv_pw = create_conv2d(in_chs, comp_chs, comp_kernel_size, padding = pad_type)
        self.bn1 = norm_layer(comp_chs)
        self.act1 = act_layer(inplace = True)
        
        # Regular convolution
        self.conv_reg = create_conv2d(comp_chs, reg_chs, reg_kernel_size, padding = pad_type)
        self.bn2 = norm_layer(reg_chs)
        self.act2 = act_layer(inplace = True)
        
        # Point - wise linear projection
        self.conv_pwl = create_conv2d(reg_chs, out_chs, comp_kernel_size, padding = pad_type)
        self.bn3 = norm_layer(out_chs)
        
    def forward(self, x):
        shortcut = x
        
        # Point - wise compression
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Regular convolution
        x = self.conv_reg(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        # Point - wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)
        
        if self.residual:
            x = x + shortcut
        
        return x