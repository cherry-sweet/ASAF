import copy
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from typing import Union




class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same',
                 bias=False, bn=True, relu=False):
        super(Conv2d, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.initialize()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None


class AttentionRefine(nn.Module):

    def __init__(self, in_channel,out_channel,dim1,dim2,dim):
        super(AttentionRefine, self).__init__()
        self.dim = in_channel
        # self.ca = SE(dim = self.dim)
        # self.ct = Conv2d(self.dim, self.embed_dim,1)
        self.attn = nn.ModuleList([
            CrossAttention(dim1=dim1[i], dim2=dim2[i], dim=dim[i])
            for i in range(3)
        ])
        # self.proj = nn.Sequential(
        #     Conv2d(self.embed_dim,self.embed_dim,3),
        #     Conv2d(self.embed_dim,out_channel,1)
        # )

    def forward(self, list):
        outputs = []
        for attn_layer, (attn, out) in zip(self.attn, list):
            attn = attn.reshape(attn.shape[0],-1,1)
            out = attn_layer(attn, out)
            outputs.append(out)
        return outputs



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, r, c = x.shape
        rev = int(r ** 0.5)
        x = x.view(b, rev, rev, c).permute(0, 3, 1, 2)
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, H, W)
        x = torch.cat([avg_out, max_out], dim=1)       # (B, 2, H, W)
        attn = self.conv(x)                            # (B, 1, H, W)
        return self.sigmoid(attn)

class SpatialAttentionRefiner2(nn.Module):
    def __init__(self, channels=[128, 128, 128]):
        super().__init__()
        self.attn = nn.ModuleList([
            SpatialAttention() for _ in channels
        ])

    def forward(self, attn_maps, feats):
        """
        attn_maps: list of (B, H, W, 1), e.g., [(16,96,96,1), (16,48,48,1), (16,24,24,1)]
        feats:     list of (B, L, C),       e.g., [(16,9216,128), ..., (16,144,1024)]
        return:     list of refined attention maps
        """
        refined_attn = []
        for i in range(len(feats)-1):
            sa_weight = self.attn[i](feats[i])             # (B,1,H,W)
            attn = attn_maps[i].permute(0, 3, 1, 2) * sa_weight        # 加权
            attn = attn / (attn.amax(dim=(2, 3), keepdim=True) + 1e-6)   # 归一化
            attn = attn.permute(0, 2, 3, 1)
            refined_attn.append(attn)
        return refined_attn


class CrossLayerSpatialAttentionRefiner(nn.Module):
    def __init__(self, channels=[128, 256, 512], high_dim=1024):
        super().__init__()
        self.attn = nn.ModuleList([
            SpatialAttention() for _ in channels
        ])
        self.project = nn.ModuleList([
            nn.Conv2d(c + high_dim, c, kernel_size=1) for c in channels
        ])

    def forward(self, attn_maps, feats):
        """
        attn_maps: list of (B, H, W, 1), e.g., [(16,96,96,1), (16,48,48,1), (16,24,24,1)]
        feats:     list of (B, L, C),       e.g., [(16,9216,128), ..., (16,144,1024)]
        """
        high_feat = feats[-1]  # (B, 144, 1024)
        B, L_high, C_high = high_feat.shape
        H_high = W_high = int(L_high ** 0.5)  # 12x12
        high_feat = high_feat.transpose(1, 2).reshape(B, C_high, H_high, W_high) # (B, 1024, 12, 12)

        refined_attn = []
        for i in range(3):  # 前三个浅层特征做 refinement
            low_feat = feats[i]  # (B, L, C)
            B, L, C = low_feat.shape
            H = W = int(L ** 0.5)
            low_feat = low_feat.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)

            # 上采样高层语义特征
            up_high = F.interpolate(high_feat, size=(H, W), mode='bilinear', align_corners=False)

            # 拼接并投影
            fused = torch.cat([low_feat, up_high], dim=1)  # (B, C+high_dim, H, W)
            fused = self.project[i](fused)  # (B, C, H, W)

            # 计算空间注意力
            sa_weight = self.attn[i](fused)  # (B, 1, H, W)

            # 原始注意力图加权
            attn = attn_maps[i].permute(0, 3, 1, 2)  # (B, 1, H, W)
            attn = attn * sa_weight
            attn = attn / (attn.amax(dim=(2, 3), keepdim=True) + 1e-6)
            refined_attn.append(attn.permute(0, 2, 3, 1))  # (B, H, W, 1)

        return refined_attn





class HighAttnGuidedRefiner(nn.Module):
    def __init__(self, channels=[128, 256, 512], high_dim=1024):
        super().__init__()
        self.attn = nn.ModuleList([
            SpatialAttention() for _ in channels
        ])
        self.project = nn.ModuleList([
            nn.Conv2d(c + high_dim, c, kernel_size=1) for c in channels
        ])

    def forward(self, attn_maps, feats):
        """
        attn_maps: list of (B, H, W, 1), e.g., [(16,96,96,1), ..., (16,12,12,1)]
        feats:     list of (B, L, C),       e.g., [(16,9216,128), ..., (16,144,1024)]
        return:    refined attention maps (前3个)
        """
        high_feat = feats[-1]                      # (B, 144, 1024)
        high_attn = attn_maps[-1]                  # (B, 12, 12, 1)

        B, L_high, C_high = high_feat.shape
        H_high = W_high = int(L_high ** 0.5)
        high_feat = high_feat.transpose(1, 2).reshape(B, C_high, H_high, W_high)

        refined_attn = []
        for i in range(3):
            # 当前浅层特征
            low_feat = feats[i]
            B, L, C = low_feat.shape
            H = W = int(L ** 0.5)
            low_feat = low_feat.transpose(1, 2).reshape(B, C, H, W)

            # 上采样高层语义特征 + 注意力
            up_high_feat = F.interpolate(high_feat, size=(H, W), mode='bilinear', align_corners=False)
            up_high_attn = F.interpolate(high_attn.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=False)  # (B,1,H,W)

            # 拼接特征 → 降通道
            fused = torch.cat([low_feat, up_high_feat], dim=1)   # (B, C + high_dim, H, W)
            fused = self.project[i](fused)                       # (B, C, H, W)

            # 空间注意力（基于 fused 特征）
            sa_weight = self.attn[i](fused)                      # (B,1,H,W)

            # attention map 融合高层语义引导
            attn = attn_maps[i].permute(0, 3, 1, 2)              # (B,1,H,W)
            attn = attn + up_high_attn                           # 加权引导
            attn = attn * sa_weight
            attn = attn / (attn.amax(dim=(2, 3), keepdim=True) + 1e-6)
            refined_attn.append(attn.permute(0, 2, 3, 1))        # (B,H,W,1)

        return refined_attn



class SpatialAttentionRefiner3(nn.Module):
    def __init__(self, channels=[128, 128, 128]):
        super().__init__()
        self.attn_block = SpatialAttention()
        self.conv_list = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, padding=1),  # 对应 24×24
            nn.Conv2d(1, 1, kernel_size=3, padding=1),  # 对应 48×48
            nn.Conv2d(1, 1, kernel_size=3, padding=1)  # 对应 96×96
        ])
        self.gate = nn.Conv2d(1, 1, kernel_size=1)



    def forward(self, attn_maps, feats):
        """
        attn_maps: list of (B, H, W, 1), e.g., [(16,96,96,1), (16,48,48,1), (16,24,24,1)]
        feats:     list of (B, L, C),       e.g., [(16,9216,128), ..., (16,144,1024)]
        return:     list of refined attention maps
        """
        refined_attn = []
        sa_weight = self.attn_block(feats[-1])
        for i in range(len(feats)-1):
            attn2 = attn_maps[i]
            B, H, W, C = attn2.shape
            attn2 = attn2.permute(0, 3, 1, 2)
            sa_weight_upsampled = F.interpolate(sa_weight, size=(H, W), mode='bilinear',align_corners=True)  # (B, 1, H, W)
            con_sa_weight_upsampled = self.conv_list[i](sa_weight_upsampled)

            gate = torch.sigmoid(self.gate(con_sa_weight_upsampled))  # (B, 1, 96, 96)
            attn = attn2 * gate + attn2

            attn = attn / (attn.amax(dim=(2, 3), keepdim=True) + 1e-6)   # 归一化
            attn = attn.permute(0, 2, 3, 1)
            refined_attn.append(attn)
        return refined_attn


class SpatialAttentionRefiner(nn.Module):
    def __init__(self, channels=[128, 128, 128]):
        super().__init__()
        self.attn_block = SpatialAttention()
        self.gate = nn.Conv2d(2, 1, kernel_size=3, padding=1)



    def forward(self, attn_maps, feats):
        """
        attn_maps: list of (B, H, W, 1), e.g., [(16,96,96,1), (16,48,48,1), (16,24,24,1)]
        feats:     list of (B, L, C),       e.g., [(16,9216,128), ..., (16,144,1024)]
        return:     list of refined attention maps
        """
        refined_attn = []
        sa_weight = self.attn_block(feats[-1])
        for i in range(len(feats)-1):
            attn2 = attn_maps[i]
            B, H, W, C = attn2.shape
            attn2 = attn2.permute(0, 3, 1, 2)
            sa_weight_upsampled = F.interpolate(sa_weight, size=(H, W), mode='bilinear',align_corners=True)  # (B, 1, H, W)

            concat_input = torch.cat([attn2, sa_weight_upsampled], dim=1)  # (B, 2, 96, 96)
            attn = self.gate(concat_input)  # 输出新的 attention


            attn = attn / (attn.amax(dim=(2, 3), keepdim=True) + 1e-6)   # 归一化
            attn = attn.permute(0, 2, 3, 1)
            refined_attn.append(attn)
        return refined_attn

#
# class ImagePyramid(nn.Module):
#     # --------------------------------------------------------
#     # InSPyReNet
#     # Copyright (c) 2021 Taehun Kim
#     # Licensed under The MIT License
#     # https://github.com/plemeri/InSPyReNet
#     # --------------------------------------------------------
#     def __init__(self, ksize=7, sigma=1, channels=1):
#         super().__init__()
#         self.ksize = ksize
#         self.sigma = sigma
#         self.channels = channels
#         self.uthreshold = 0.5
#         k = cv2.getGaussianKernel(ksize, sigma)
#         k = np.outer(k, k)
#         k = torch.tensor(k).float()
#         self.kernel = k.repeat(channels, 1, 1, 1)
#         self.temperature = nn.Parameter(torch.tensor(2.0))
#
#     def get_uncertain(self, fea, smap, epoch,visual=True):
#         B,_,_= fea.shape  # fea:(16,144,1024)  samp:(16,12,12,1)
#         smap = smap.permute(0, 3, 1, 2) #  (16,1,12,12)
#         smap = torch.sigmoid(smap)  # 显著图归一化  (16,1,12,12)
#         p = smap - self.uthreshold  # (8,1,96,96)  self.uthreshold 是不确定性阈值
#         cg = self.uthreshold - torch.abs(p)  #
#         cg = F.pad(cg, (self.ksize // 2,) * 4, 'constant', 0)  # 四周补0操作  (16,1,18,18)
#         self.kernel = self.kernel.to(dtype=cg.dtype, device=cg.device)
#         cg = F.conv2d(cg, self.kernel * 4, groups=1)  # self.kernel:(1,1,7,7)  cg:(16,1,12,12)
#         cg = cg / (cg.max() + 1e-6)
#         # _u = torch.where(cg > 0.1, 0.0, 1.0)
#         _u = torch.sigmoid((cg - 0.5) * self.temperature)
#         ##试试残差增强  或者前30轮不增强  后面增强  前面就好好学习就行
#         # fea = fea * _u + fea.detach() * (1 - _u) * 0.1
#         _u = _u.view(B, 1, -1).permute(0, 2, 1)  # _u:(16,144,1)
#         if visual is True:
#             os.makedirs(f'../visualize/attn4/', exist_ok=True)
#             torch.save(_u, '../visualize/attn4/u.pt')
#         # fea = fea*_u   或者用其他的残差方式
#         weight = max(0.0, 1 - epoch / 50)
#         # fea = fea * _u + fea.detach() * (1 - _u) * weight
#         fea = fea * (1-_u) + fea * _u * weight
#         # fea = fea + fea*_u
#         return fea, _u
# #

class ImagePyramid(nn.Module):
    def __init__(self, ksize=7, sigma=1, channels=1):
        super().__init__()
        self.ksize = ksize
        self.sigma = sigma
        self.channels = channels
        self.uthreshold = 0.5
        k = cv2.getGaussianKernel(ksize, sigma)
        k = np.outer(k, k)
        k = torch.tensor(k).float()
        self.kernel = k.repeat(channels, 1, 1, 1)
        self.temperature = nn.Parameter(torch.tensor(2.0))

    def get_uncertain(self, fea, smap, epoch, mid_width=0.1, strength=0.5, visual=True):
        B,_,_ = fea.shape
        smap = smap.permute(0, 3, 1, 2)
        smap = torch.sigmoid(smap)

        # 保留原有可微分 _u 生成方式
        p = smap - self.uthreshold
        cg = self.uthreshold - torch.abs(p)
        cg = F.pad(cg, (self.ksize//2,)*4, 'constant', 0)
        self.kernel = self.kernel.to(dtype=cg.dtype, device=cg.device)
        cg = F.conv2d(cg, self.kernel * 4, groups=1)
        cg = cg / (cg.max() + 1e-6)
        _u = torch.sigmoid((cg - 0.5) * self.temperature)

        # 可微分的“中值抑制”：用 sigmoid 做 soft mask
        mid_threshold = _u.median(dim=-1)[0].median(dim=-1)[0].view(B,1,1,1)
        mid_mask = torch.sigmoid(10 * (mid_width - torch.abs(_u - mid_threshold)))
        _u = _u * (1 - mid_mask) + _u * mid_mask * strength

        _u = _u.view(B, 1, -1).permute(0,2,1)
        weight = max(0.0, 1 - epoch / 50)
        fea = fea * (1 - _u) + fea * _u * weight

        if visual:
            os.makedirs(f'../visualize/attn4/', exist_ok=True)
            torch.save(_u, '../visualize/attn4/u.pt')

        return fea, _u




class GCNCombiner(nn.Module):

    def __init__(self,
                 total_num_selects: int,
                 num_classes: int,
                 proj_size: Union[int, None] = None):
        """
        If building backbone without FPN, set fpn_size to None and MUST give
        'inputs' and 'proj_size', the reason of these setting is to constrain the
        dimension of graph convolutional network input.
        """
        super(GCNCombiner, self).__init__()

        self.proj_size = proj_size
        ### build one layer structure (with adaptive module)
        num_joints = total_num_selects // 256

        self.param_pool0 = nn.Linear(total_num_selects, num_joints)

        A = torch.eye(num_joints) / 100 + 1 / 100
        self.adj1 = nn.Parameter(copy.deepcopy(A))
        self.conv1 = nn.Conv1d(self.proj_size, self.proj_size, 1)
        self.batch_norm1 = nn.BatchNorm1d(self.proj_size)

        self.conv_q1 = nn.Conv1d(self.proj_size, self.proj_size // 4, 1)
        self.conv_k1 = nn.Conv1d(self.proj_size, self.proj_size // 4, 1)
        self.alpha1 = nn.Parameter(torch.zeros(1))

        ### merge information
        self.param_pool1 = nn.Linear(num_joints, 1)

        #### class predict
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.proj_size, num_classes)

        self.tanh = nn.Tanh()

        self.embed_adj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1024)
            ) for dim in [128, 256, 512]
        ])

    def forward(self, x):
        """
        在特征上进行处理的  """
        hs = []
        for i in range(len(x)-1):
            fea = self.embed_adj[i](x[i])  # (16,200)
            hs.append(fea)
        hs.append(x[-1])
        # list4:(8,256,1536) (8,128,1536) (8,64,1536) (8,32,1536)
        hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous()  # (8,1536,480)  B, S', C --> B, C, S
        # print(hs.size(), names)
        hs = self.param_pool0(hs)  # 通过一个线性层映射(16,1024,47)  7 就是结点个数
        ### adaptive adjacency  构建自适应邻接矩阵
        q1 = self.conv_q1(hs).mean(1)  # 经过卷积层
        k1 = self.conv_k1(hs).mean(1)
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))  # (8,7,7) 反映不同节点之间的关联性
        A1 = self.adj1 + A1 * self.alpha1  # self.adj1 固定初始化   (16,47,47)
        ### graph convolution
        hs = self.conv1(hs)
        hs = torch.matmul(hs, A1)
        hs = self.batch_norm1(hs)
        ### predict
        hs = self.param_pool1(hs)
        hs = self.dropout(hs)
        hs = hs.flatten(1)
        hs = self.classifier(hs)

        return hs  # (16,200)

