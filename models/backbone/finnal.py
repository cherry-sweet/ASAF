"""
based on https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:  # ['avg','max']
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))) # (1,128,7,7)-->(1,128,1,1)
                channel_att_raw = self.mlp( avg_pool )  # (1,128,1,1)-->(1,128)
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale + x # (16,1024,12,12)


def logsumexp_2d(tensor):

    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):

    def __init__(self,upsample=None):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.upsample_size = upsample

    def forward(self, x):
        x_upsampled = F.interpolate(x, size=self.upsample_size, mode='bilinear', align_corners=True) # (16,1024,12,12)-->(16,1024,96,96)
        x_compress = self.compress(x_upsampled)  # (16,1024,96,96)-->(16,2,96,96)
        scale = self.spatial(x_compress)  # (16,2,96,96)-->(16,1,96,96)
        return scale  # (16,1,96,96)


class CBAM(nn.Module):

    def __init__(self, gate_channels,upsample=None, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(upsample)
        # self.fc = nn.Conv2d(gate_channels, part_channels, kernel_size=1, bias=False)

    def forward(self, x, decision_mask):
        x_out = self.ChannelGate(x)  # (16,1024,12,12)-->(16,1024,12,12)
        x_out = self.SpatialGate(x_out)  # (16,1,96,96)
        # x_t, parts_mask = x_out  # (1,128,7,7)  (1,1,7,7)  x_t已经是与mask乘过的了
        # x_t = self.fc(x_t)  # (1,128,7,7)-->(1,16,7,7)
        return x_out # (16,1,96,96)  归一化后

class atten_fused(nn.Module):

    def __init__(self,parts_dim=1024, num_stages=3, fuse=True):
        super(atten_fused, self).__init__()
        self.num_stages = num_stages
        self.upsample_size = [96,48,24]
        self.parts_dim = parts_dim
        cbam_list = [CBAM(gate_channels=self.parts_dim, upsample=self.upsample_size[i], reduction_ratio=16) for i in range(self.num_stages)]
        self.cbam_list = nn.ModuleList(cbam_list)
        self.fuse = fuse
        if self.fuse is not True:
            self.gate = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x, decision_mask):
        B, N, C = x[-1].shape  # (16, 144, 1024)
        H = W = int(N ** 0.5)
        x_last = x[-1].view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        attn_list_out = []
        for mask, cbam in zip(decision_mask[:3], self.cbam_list):  # self.cbam_list 8个cbam
            mask = mask.permute(0, 3, 1, 2).contiguous()
            fuse_attn = cbam(x_last, mask) #
            if self.fuse == True:
                # scale = fuse_attn * mask  # (16,1,96,96)  scale:全局特征获得到全局注意力  decision_mask是局部注意力
                scale = 0.5*fuse_attn + 0.5*mask
                fuse_attn = torch.sigmoid(scale)  # broadcasting  归一化
                attn = fuse_attn.permute(0, 2, 3, 1)
            else:
                concat_input = torch.cat([fuse_attn, mask], dim=1)  # (B, 2, 96, 96)
                attn = self.gate(concat_input)  # 输出新的 attention
                attn = torch.sigmoid(attn)  # 归一化
                attn = attn.permute(0, 2, 3, 1)


            attn_list_out.append(attn)  # list3:(16,96,96,1) (16,48,48,1) (16,24,24,1)
        return attn_list_out


