# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath
from model.BIF.biFormer import Block as BLK


class BiFBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, top_k=4,kv_downsample_mode='ada_avgpool'):
        """ Biformer and Conv Block
        """
        super(BiFBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.topk = top_k
        self.kv_downsample_mode = kv_downsample_mode

        self.trans_block = BLK(dim = self.trans_dim,n_win=8,topk=self.topk,kv_downsample_mode=self.kv_downsample_mode)
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res

        return x

class BDNet_control(nn.Module):
    def __init__(self, in_nc=1, config=[1, 2, 3, 3, 3, 2, 1], dim=64):
        super(BDNet_control, self).__init__()
        self.config = config
        self.dim = dim
        self.kv_downsample_mode = 'ada_avgpool'

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]
        begin = 0
        self.m_down1 = [BiFBlock(dim // 2, dim // 2,   1, self.kv_downsample_mode)
                        for i in range(config[0])] + \
                       [nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [BiFBlock(dim, dim,  4 , self.kv_downsample_mode)
                        for i in range(config[1])] + \
                       [nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [BiFBlock(2 * dim, 2 * dim,  16, self.kv_downsample_mode)
                        for i in range(config[2])] + \
                       [nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [BiFBlock(4 * dim, 4 * dim, 8 ,self.kv_downsample_mode)
                       for i in range(config[3])]

        begin += config[3]
        self.m_up3 = [nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False), ] + \
                     [BiFBlock(2 * dim, 2 * dim,  16,self.kv_downsample_mode)
                      for i in range(config[4])]

        begin += config[4]
        self.m_up2 = [nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False), ] + \
                     [BiFBlock(dim, dim,  4,self.kv_downsample_mode)
                      for i in range(config[5])]

        begin += config[5]
        self.m_up1 = [nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False), ] + \
                     [BiFBlock(dim // 2, dim // 2,  1,self.kv_downsample_mode)
                      for i in range(config[6])]

        self.m_tail = [nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)
        # self.apply(self._init_weights)

    def forward(self, x0, control=None):
        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        with torch.no_grad():
            x1 = self.m_head(x0)
            x2 = self.m_down1(x1)
            x3 = self.m_down2(x2)
            x4 = self.m_down3(x3)
            x = self.m_body(x4)

        if control is None:
            raise
        x += control.pop()
        x = self.m_up3(x + x4 + control.pop())
        x = self.m_up2(x + x3 + control.pop())
        x = self.m_up1(x + x2 + control.pop())
        x = self.m_tail(x + x1 + control.pop())
        x = x[..., :h, :w]

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class InceptionModule(nn.Module):
    def __init__(self,ch):
        super(InceptionModule,self).__init__()
        self.conv1x1 = nn.Sequential(nn.Conv2d(ch,ch,1))
        self.conv3x3 = nn.Sequential(nn.Conv2d(ch,ch//2,3,padding=1))
        self.conv5x5 = nn.Sequential(nn.Conv2d(ch,ch//2,5,padding=2))
        self.merge = nn.Sequential(zero_module(nn.Conv2d(2*ch,ch,1)))
       
    def forward(self,x):
        out1 = self.conv1x1(x)
        out3 = self.conv3x3(x)
        out5 = self.conv5x5(x)

        out = torch.cat([out1,out3,out5],dim=1)
        out = self.merge(out)
        return out

class ControlNet(nn.Module):
    def __init__(self, in_nc=1, config=[1, 2, 3, 3, 3, 2, 1], dim=64):
        super(ControlNet, self).__init__()
        self.config = config
        self.dim = dim
        self.kv_downsample_mode = 'ada_avgpool'
        self.hint_channels = 1
        hint_channels = self.hint_channels
        ch = 64

        self.input_hint_block = nn.Sequential(
            nn.Conv2d(hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=1),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=1),
            nn.SiLU(),
            zero_module(nn.Conv2d(256, ch, 3, padding=1))
        )

        self.zero_convs1=InceptionModule(ch)
        self.zero_convs2=InceptionModule(128)
        self.zero_convs3=InceptionModule(256)
        self.zero_convs4=InceptionModule(512)
        self.zero_convs5=InceptionModule(512)

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [BiFBlock(dim // 2, dim // 2, 1, self.kv_downsample_mode)
                        for i in range(config[0])] + \
                       [nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [BiFBlock(dim, dim, 4 , self.kv_downsample_mode)
                        for i in range(config[1])] + \
                       [nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [BiFBlock(2 * dim, 2 * dim, 16, self.kv_downsample_mode)
                        for i in range(config[2])] + \
                       [nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [BiFBlock(4 * dim, 4 * dim, 8 ,self.kv_downsample_mode)
                       for i in range(config[3])]

        begin += config[3]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.apply(self._init_weights)

    def forward(self, x0, hint=None):
        outs = []
        if hint is not None:
            guided_hint = self.input_hint_block(hint)

        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)
        x1 = self.m_head(x0)
        x1 += guided_hint
        outs.append(self.zero_convs1(x1))
        x2 = self.m_down1(x1)
        outs.append(self.zero_convs2(x2))
        x3 = self.m_down2(x2)
        outs.append(self.zero_convs3(x3))
        x4 = self.m_down3(x3)
        outs.append(self.zero_convs4(x4))
        x = self.m_body(x4)
        outs.append(self.zero_convs5(x))

        return outs

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class BDCNet(nn.Module):
    def __init__(self):
        super(BDCNet, self).__init__()
        self.module = BDNet_control()
        self.control_model = ControlNet()
    def forward(self,x ,hint):
        control = self.control_model(x,hint)
        out = self.module(x,control)
        return out

if __name__ == '__main__':
    def network_parameters(nets):
        num_params = sum(param.numel() for param in nets.parameters())
        return num_params

    net = BDCNet()
    x = torch.randn(1,1,256,256)
    hint = torch.randn(1,1,256,256)
    print(net(x,hint).shape)
    p_number = network_parameters(net)
    print(p_number)