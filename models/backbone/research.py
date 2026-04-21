import torch
import math
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
        B, N, C = x.shape  # (b, h*w, c)
        H = W = int(N ** 0.5)  # 如果 h*w 是平方数，比如 49=7*7, 196=14*14
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        for pool_type in self.pool_types:
            if pool_type=='avg':  # (16,128,96,96)-->(16,128,1,1)
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool ) # (16,128)
            elif pool_type=='max':  # (16,128,96,96)-->(16,128,1,1)
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
                channel_att_sum = channel_att_sum + channel_att_raw  # 平均和最大相加

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)  # (16,2,96,96)
        x_out = self.spatial(x_compress)  # (16,1,96,96)
        scale = torch.sigmoid(x_out) # (16,1,96,96) broadcasting
        scale = scale.permute(0, 2, 3, 1).contiguous()
        # return x * scale
        return scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)  # (16,128,96,96)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class new_attn(nn.Module):

    def __init__(self, parts_dim=[128,256,512], fuse=True):
        super(new_attn, self).__init__()
        self.parts_dim = parts_dim
        cbam_list = [CBAM(gate_channels=dim, reduction_ratio=16) for dim in
                     self.parts_dim]
        self.cbam_list = nn.ModuleList(cbam_list)
        self.fuse = fuse
        if self.fuse is not True:
            self.gate = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.v = nn.ModuleList([nn.Linear(dim, dim, bias=True) for dim in [128,256,512]])
        self.proj = nn.ModuleList([nn.Linear(dim, dim, bias=True) for dim in [128,256,512]])
    def forward(self, x, decision_mask): # 前面是attn  后面是fea
        outs =[]  # x:list4:(16,9216,128)  (16,2304,256)  (16,576,512) (16,144,1024) decision:(16,96,96,1) ...
        attn_new_list = []
        for i in range(3):
            out_v = self.v[i](x[i]) # (16,9216,128)
            attn_g = self.cbam_list[i](x[i])  # (16,96,96,1)
            attn_new = attn_g*decision_mask[i] # (16,96,96,1)
            attn_new_list.append(attn_new)
            # att: (B, H, W, 1)
            attn_new = attn_new.reshape(attn_new.shape[0], -1, 1)  # -> (B, H*W, 1)
            out_fea = out_v*attn_new  # (16,9216,128)
            out = self.proj[i](out_fea)
            outs.append(out)

        return outs,attn_new_list
