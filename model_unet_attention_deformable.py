"""
@author: zjf
@create time: 2021/12/28 17:19
@desc:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.deform_conv_v2 import get_deform_conv



class Attention_pure(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        C = int(C // 3)
        qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(x)
        return x


class MixBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm, downsample=2):
        super().__init__()

        self.dim = dim
        self.norm1 = nn.BatchNorm2d(dim)

        self.conv2 = nn.Conv2d(dim, dim, (1, 1))
        self.conv = nn.Conv2d(dim // 2, dim // 2, (3, 3), padding=(1, 1), groups=dim // 2)
        self.channel_up = nn.Conv2d(dim // 2, 3 * dim // 2, (1, 1))
        self.attn = Attention_pure(
            dim // 2,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.downsample = downsample
        mlp_hidden_dim = int(dim * mlp_ratio)

    def forward(self, x):
        B, _, H, W = x.shape
        residual = x

        qkv = x[:, :(self.dim // 2), :, :]
        conv = x[:, (self.dim // 2):, :, :]
        conv = self.conv(conv)

        sa = nn.functional.interpolate(qkv, size=(H // self.downsample, W // self.downsample), mode='bilinear',
                                       align_corners=True)
        sa = self.channel_up(sa)
        B, _, H_down, W_down = sa.shape
        sa = sa.flatten(2).transpose(1, 2)

        sa = self.attn(sa)
        sa = sa.reshape(B, H_down, W_down, -1).permute(0, 3, 1, 2).contiguous()
        sa = nn.functional.interpolate(sa, size=(H, W), mode='bilinear', align_corners=True)
        x = torch.cat([conv, sa], dim=1)
        x = self.conv2(self.norm2(x))
        return x


class model(nn.Module):
    def __init__(self, num_classes=10):
        super(model, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(100, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            # nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.LeakyReLU()
        )
        self.deform_conv1 = get_deform_conv(128, 128, modulation=True)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            # nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.LeakyReLU(),
        )

        self.deform_conv2 = get_deform_conv(256, 256, modulation=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.LeakyReLU(),
        )
        self.deform_conv3 = get_deform_conv(512, 512, modulation=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.trans1 = MixBlock(dim=512, num_heads=8)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.up3 = nn.ConvTranspose2d(512, 512, (2, 2), stride=(2, 2))

        self.upconv3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
        )

        self.up2 = nn.ConvTranspose2d(256, 256, (2, 2), stride=(2, 2))
        self.upconv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
        )

        self.up1 = nn.ConvTranspose2d(128, 128, (2, 2), stride=(2, 2))
        self.upconv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU()
        )

    def forward(self, x):
        # 100*256*256
        conv1 = self.conv1(x)
        # print(conv1.shape)
        conv1 = self.deform_conv1(conv1)
        # 128*256*256
        conv1_pool1 = self.pool1(conv1)
        # 128*128*128
        conv2 = self.conv2(conv1_pool1)
        conv2 = self.deform_conv2(conv2)
        # 256*128*128
        conv2_pool2 = self.pool2(conv2)
        # 256*64*64
        conv3 = self.conv3(conv2_pool2)
        conv3 = self.deform_conv3(conv3)
        # 512*64*64
        conv3_pool3 = self.pool3(conv3)
        # 512*32*32

        conv4_1 = self.conv4_1(conv3_pool3)
        trans1 = self.trans1(conv4_1)
        # 512*32*32
        conv4_2 = self.conv4_2(trans1)
        # 512*32*32
        upconv3_ConvTranspose2d = self.up3(conv4_2)
        # 512*64*64
        cat3 = torch.cat([upconv3_ConvTranspose2d, conv3], 1)
        # 1024*64*64
        upconv3 = self.upconv3(cat3)
        # 256*64*64
        upconv2_ConvTranspose2d = self.up2(upconv3)
        # 256*128*128
        cat2 = torch.cat([upconv2_ConvTranspose2d, conv2], 1)
        # 512*128*128
        upconv2 = self.upconv2(cat2)
        # 128*128*128

        upconv1_ConvTranspose2d = self.up1(upconv2)
        # 128*256*256
        cat1 = torch.cat([upconv1_ConvTranspose2d, conv1], 1)
        # 256*256*256
        out = self.upconv1(cat1)
        # 1*256*256
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


def get_net():
    return model()
