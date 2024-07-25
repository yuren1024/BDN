# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from model.BIF.biFormer import Block as BLK


class BiFBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim,top_k=4,kv_downsample_mode='ada_avgpool'):
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

class BDNet(nn.Module):
    def __init__(self, in_nc=1, config=[1, 2, 3, 3, 3, 2, 1], dim=64):
        super(BDNet, self).__init__()
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
        self.apply(self._init_weights)

    def forward(self, x0):
        h, w = x0.size()[-2:]
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
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