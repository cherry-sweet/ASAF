# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import sys

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
# from .MIT import AttentionRefine,SpatialAttentionRefiner, ImagePyramid,GCNCombiner
# from finnal import atten_fused
from .research import new_attn

class AttentionMaskPredict(nn.Module):
	def __init__(self, dim=1, input_dim=1):
		super().__init__()
		self.norm = nn.LayerNorm(input_dim)
		self.linear1 = nn.Linear(dim, 64)
		self.linear2 = nn.Linear(64, 32)
		self.linear3 = nn.Linear(32, dim)
		self.final_linear = nn.Linear(dim, 2)
		self.log_softmax = nn.LogSoftmax(dim=-1)
		self.gelu = nn.GELU()

	def forward(self, x):
		# x: (B, N, 1)
		x = x.transpose(1, 2)  # → (B, 1, N) for LayerNorm across N
		x = self.norm(x)
		x = x.transpose(1, 2)  # → (B, N, 1)

		x = self.gelu(self.linear1(x))  # (B, N, 64)
		x = self.gelu(self.linear2(x))  # (B, N, 32)
		x = self.gelu(self.linear3(x))  # (B, N, 1)
		x = self.final_linear(x)        # (B, N, 1)
		x = self.log_softmax(x)         # (B, N, 1)
		return x

class Mlp(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x


def window_partition(x, window_size):
	"""
	Args:
		x: (B, H, W, C)
		window_size (int): window size

	Returns:
		windows: (num_windows*B, window_size, window_size, C)
	"""
	B, H, W, C = x.shape
	x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
	windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
	return windows


def window_reverse(windows, window_size, H, W):
	"""
	Args:
		windows: (num_windows*B, window_size, window_size, C)
		window_size (int): Window size
		H (int): Height of image
		W (int): Width of image

	Returns:
		x: (B, H, W, C)
	"""
	B = int(windows.shape[0] / (H * W / window_size / window_size))
	x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
	x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
	return x



class WindowAttention(nn.Module):
	r""" Window based multi-head self attention (W-MSA) module with relative position bias.
	It supports both of shifted and non-shifted window.

	Args:
		dim (int): Number of input channels.
		window_size (tuple[int]): The height and width of the window.
		num_heads (int): Number of attention heads.
		qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
		qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
		attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
		proj_drop (float, optional): Dropout ratio of output. Default: 0.0
	"""

	def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.,
	             proj_drop=0.,assess=True):

		super().__init__()
		self.dim = dim
		self.window_size = window_size  # Wh, Ww
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = qk_scale or head_dim ** -0.5

		# define a parameter table of relative position bias
		self.relative_position_bias_table = nn.Parameter(
			torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

		# get pair-wise relative position index for each token inside the window
		coords_h = torch.arange(self.window_size[0])
		coords_w = torch.arange(self.window_size[1])
		coords = torch.stack(torch.meshgrid([coords_h, coords_w],indexing="ij"))  # 2, Wh, Ww
		# coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
		coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
		relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
		relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
		relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
		relative_coords[:, :, 1] += self.window_size[1] - 1
		relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
		relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
		self.register_buffer("relative_position_index", relative_position_index)

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

		self.assess=assess

		trunc_normal_(self.relative_position_bias_table, std=.02)
		self.softmax = nn.Softmax(dim=-1)



	def forward(self, x, mask=None):
		"""
		Args:
			x: input features with shape of (num_windows*B, N, C)
			mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
		"""
		B_, N, C = x.shape
		qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

		q = q * self.scale
		attn = (q @ k.transpose(-2, -1))
		#  相对位置编码
		relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
			self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
		relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
		attn = attn + relative_position_bias.unsqueeze(0)

		if mask is not None:
			nW = mask.shape[0]
			attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
			attn = attn.view(-1, self.num_heads, N, N)
			attn = self.softmax(attn)
		else:
			attn = self.softmax(attn)

		# if self.assess:
			# os.makedirs(f'visualize/backbone_attention/', exist_ok=True)
			# torch.save(attn, 'visualize/backbone_attention/attn.pt')

		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		if self.assess:
			return x,attn
		else:
			return x, None

	def extra_repr(self) -> str:
		return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

	def flops(self, N):
		# calculate flops for 1 window with token length of N
		flops = 0
		# qkv = self.qkv(x)
		flops += N * self.dim * 3 * self.dim
		# attn = (q @ k.transpose(-2, -1))
		flops += self.num_heads * N * (self.dim // self.num_heads) * N
		#  x = (attn @ v)
		flops += self.num_heads * N * N * (self.dim // self.num_heads)
		# x = self.proj(x)
		flops += N * self.dim * self.dim
		return flops


class SwinTransformerBlock(nn.Module):
	r""" Swin Transformer Block.

	Args:
		dim (int): Number of input channels.
		input_resolution (tuple[int]): Input resulotion.
		num_heads (int): Number of attention heads.
		window_size (int): Window size.
		shift_size (int): Shift size for SW-MSA.
		mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
		qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
		qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
		drop (float, optional): Dropout rate. Default: 0.0
		attn_drop (float, optional): Attention dropout rate. Default: 0.0
		drop_path (float, optional): Stochastic depth rate. Default: 0.0
		act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
		norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
		fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
	"""

	def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
	             mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
	             act_layer=nn.GELU, norm_layer=nn.LayerNorm):
		super().__init__()
		self.dim = dim
		self.input_resolution = input_resolution
		self.num_heads = num_heads
		self.window_size = window_size
		self.shift_size = shift_size
		self.mlp_ratio = mlp_ratio
		if min(self.input_resolution) <= self.window_size:
			# if window size is larger than input resolution, we don't partition windows
			self.shift_size = 0
			self.window_size = min(self.input_resolution)
		assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

		self.norm1 = norm_layer(dim)
		self.attn = WindowAttention(
			dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
			qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


		if self.shift_size > 0:
			# calculate attention mask for SW-MSA
			H, W = self.input_resolution
			img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
			h_slices = (slice(0, -self.window_size),
			            slice(-self.window_size, -self.shift_size),
			            slice(-self.shift_size, None))
			w_slices = (slice(0, -self.window_size),
			            slice(-self.window_size, -self.shift_size),
			            slice(-self.shift_size, None))
			cnt = 0
			for h in h_slices:
				for w in w_slices:
					img_mask[:, h, w, :] = cnt
					cnt += 1

			mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
			mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
			attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
			attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
		else:
			attn_mask = None

		self.register_buffer("attn_mask", attn_mask)

	def forward(self, x):
		H, W = self.input_resolution
		B, L, C = x.shape
		# print(H,W,L,C)
		assert L == H * W, "input feature has wrong size"

		shortcut = x
		x = self.norm1(x)
		x = x.view(B, H, W, C)

		# cyclic shift
		if self.shift_size > 0:
			shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
				# partition windows
			x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
		else:
			shifted_x = x
			# partition windows
			x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

		x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

		# W-MSA/SW-MSA
		attn_windows, attn_weight = self.attn(x_windows, mask=self.attn_mask)  # (1024,4,144,144) nW*B, window_size*window_size, C

		# merge windows
		attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

		# reverse cyclic shift
		if self.shift_size > 0:
			shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
			x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
		else:
			shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
			x = shifted_x

		# attn_weight
		win_num = int(self.input_resolution[0]/self.window_size)*int(self.input_resolution[1]/self.window_size)
		attn_weight = attn_weight.view(B,win_num,*attn_weight.shape[1:])
		attn_weight = attn_weight.mean(dim=-2)  # (101,64,4,144,144)   4个头
		attn_weight = attn_weight[:,:,0,:] # 看好维度  取得平均   取第一个头？？？  cub:0 nabird:
		attn_weight = attn_weight.view(-1, self.window_size, self.window_size, 1)
		# # reverse cyclic shift
		if self.shift_size > 0:
			shifted_x = window_reverse(attn_weight, self.window_size, H, W)  # B H' W' C
			attn_weight = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
		else:
			shifted_x = window_reverse(attn_weight, self.window_size, H, W)  # B H' W' C
			attn_weight = shifted_x
		attn = attn_weight  # (16,96,96,1)
		# attn = attn / (torch.amax(attn, dim=(1, 2), keepdim=True) + 1e-6)
		#
		# # attn_weight mask predict
		# attn_weight  = attn_weight.view(B, -1, 1)
		# attn_weight = self.attn_mask_predict(attn_weight)
		# attn_weight = F.gumbel_softmax(attn_weight, tau=1.0, hard=True)
		# binary_mask = attn_weight[:, :, 1]

		# reshape 回原始形状 (16, 96, 96, 1)
		# binary_mask = binary_mask.view(B, H, W, 1)
		# 将 mask 作用在 feature 上（广播乘法）   看加在什么位置好
		#   # (16, 96, 96, 128)
		# x = x * binary_mask


		x = x.view(B, H * W, C)
		x = shortcut + self.drop_path(x)


		# FFN
		x = x + self.drop_path(self.mlp(self.norm2(x)))

		return x,attn

	def extra_repr(self) -> str:
		return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
		       f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

	def flops(self):
		flops = 0
		H, W = self.input_resolution
		# norm1
		flops += self.dim * H * W
		# W-MSA/SW-MSA
		nW = H * W / self.window_size / self.window_size
		flops += nW * self.attn.flops(self.window_size * self.window_size)
		# mlp
		flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
		# norm2
		flops += self.dim * H * W
		return flops


class PatchMerging(nn.Module):
	r""" Patch Merging Layer.

	Args:
		input_resolution (tuple[int]): Resolution of input feature.
		dim (int): Number of input channels.
		norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
	"""

	def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
		super().__init__()
		self.input_resolution = input_resolution
		self.dim = dim
		self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
		self.norm = norm_layer(4 * dim)

	def forward(self, x):
		"""
		x: B, H*W, C
		"""
		H, W = self.input_resolution
		B, L, C = x.shape
		# print(H,W,L,C)
		assert L == H * W, "input feature has wrong size"
		assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

		x = x.view(B, H, W, C)

		x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
		x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
		x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
		x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
		x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
		x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

		x = self.norm(x)
		x = self.reduction(x)

		return x

	def extra_repr(self) -> str:
		return f"input_resolution={self.input_resolution}, dim={self.dim}"

	def flops(self):
		H, W = self.input_resolution
		flops = H * W * self.dim
		flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
		return flops


class BasicLayer(nn.Module):
	""" A basic Swin Transformer layer for one stage.

	Args:
		dim (int): Number of input channels.
		input_resolution (tuple[int]): Input resolution.
		depth (int): Number of blocks.
		num_heads (int): Number of attention heads.
		window_size (int): Local window size.
		mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
		qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
		qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
		drop (float, optional): Dropout rate. Default: 0.0
		attn_drop (float, optional): Attention dropout rate. Default: 0.0
		drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
		norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
		downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
		use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
		fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
	"""

	def __init__(self, dim, input_resolution, depth, num_heads, window_size,
	             mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
	             drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

		super().__init__()
		self.dim = dim
		self.input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2) if downsample else input_resolution
		self.depth = depth
		self.use_checkpoint = use_checkpoint

		# build blocks
		self.blocks = nn.ModuleList([
			SwinTransformerBlock(dim=dim, input_resolution=self.input_resolution,
			                     num_heads=num_heads, window_size=window_size,
			                     shift_size=0 if (i % 2 == 0) else window_size // 2,
			                     mlp_ratio=mlp_ratio,
			                     qkv_bias=qkv_bias, qk_scale=qk_scale,
			                     drop=drop, attn_drop=attn_drop,
			                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
			                     norm_layer=norm_layer)
			for i in range(depth)])

		# patch merging layer
		if downsample is not None:
			self.downsample = downsample(input_resolution, dim=dim // 2, norm_layer=norm_layer)
		else:
			self.downsample = None

	def forward(self, x):
		attns = []
		if self.downsample is not None:
			x = self.downsample(x)
		for blk in self.blocks:
			if self.use_checkpoint:
				x = checkpoint.checkpoint(blk, x)
			else:
				x, attn = blk(x)
				attns.append(attn)
		return x, attns

	def extra_repr(self) -> str:
		return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

	def flops(self):
		flops = 0
		for blk in self.blocks:
			flops += blk.flops()
		if self.downsample is not None:
			flops += self.downsample.flops()
		return flops


class PatchEmbed(nn.Module):
	r""" Image to Patch Embedding

	Args:
		img_size (int): Image size.  Default: 224.
		patch_size (int): Patch token size. Default: 4.
		in_chans (int): Number of input image channels. Default: 3.
		embed_dim (int): Number of linear projection output channels. Default: 96.
		norm_layer (nn.Module, optional): Normalization layer. Default: None
	"""

	def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
		super().__init__()
		img_size = to_2tuple(img_size)
		patch_size = to_2tuple(patch_size)
		patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
		self.img_size = img_size
		self.patch_size = patch_size
		self.patches_resolution = patches_resolution
		self.num_patches = patches_resolution[0] * patches_resolution[1]

		self.in_chans = in_chans
		self.embed_dim = embed_dim

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
		if norm_layer is not None:
			self.norm = norm_layer(embed_dim)
		else:
			self.norm = None

	def forward(self, x):
		B, C, H, W = x.shape
		# FIXME look at relaxing size constraints
		assert H == self.img_size[0] and W == self.img_size[1], \
			f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
		x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
		if self.norm is not None:
			x = self.norm(x)
		return x

	def flops(self):
		Ho, Wo = self.patches_resolution
		flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
		if self.norm is not None:
			flops += Ho * Wo * self.embed_dim
		return flops


class SwinTransformer(nn.Module):
	r""" Swin Transformer
		A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
		  https://arxiv.org/pdf/2103.14030

	Args:
		img_size (int | tuple(int)): Input image size. Default 224
		patch_size (int | tuple(int)): Patch size. Default: 4
		in_chans (int): Number of input image channels. Default: 3
		num_classes (int): Number of classes for classification head. Default: 1000
		embed_dim (int): Patch embedding dimension. Default: 96
		depths (tuple(int)): Depth of each Swin Transformer layer.
		num_heads (tuple(int)): Number of attention heads in different layers.
		window_size (int): Window size. Default: 7
		mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
		qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
		qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
		drop_rate (float): Dropout rate. Default: 0
		attn_drop_rate (float): Attention dropout rate. Default: 0
		drop_path_rate (float): Stochastic depth rate. Default: 0.1
		norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
		ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
		patch_norm (bool): If True, add normalization after patch embedding. Default: True
		use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
		fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
	"""

	def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
	             embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
	             window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
	             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
	             norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
	             use_checkpoint=False, cross_layer=False,
	             **kwargs):
		super().__init__()
		self.num_classes = num_classes
		self.num_layers = len(depths)
		self.embed_dim = embed_dim
		self.ape = ape
		self.patch_norm = patch_norm
		self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
		self.mlp_ratio = mlp_ratio
		self.cross_layer = cross_layer
		self.save_feature = None
		self.attn_globle = None
		self.attn_new = None

		# split image into non-overlapping patches
		self.patch_embed = PatchEmbed(
			img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
			norm_layer=norm_layer if self.patch_norm else None)
		num_patches = self.patch_embed.num_patches
		patches_resolution = self.patch_embed.patches_resolution
		self.patches_resolution = patches_resolution

		# absolute position embedding
		if self.ape:
			self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
			trunc_normal_(self.absolute_pos_embed, std=.02)

		self.pos_drop = nn.Dropout(p=drop_rate)

		# stochastic depth
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

		# build layers

		# Add
		# self.num_layers -= 1
		# Add end

		self.layers = nn.ModuleList()
		for i_layer in range(self.num_layers):
			input_resolution = (patches_resolution[0], patches_resolution[1]) if i_layer == 0 else \
				(patches_resolution[0] // (2 ** (i_layer - 1)),
				 patches_resolution[1] // (2 ** (i_layer - 1)))
			layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
			                   input_resolution=input_resolution,
			                   depth=depths[i_layer],
			                   num_heads=num_heads[i_layer],
			                   window_size=window_size,
			                   mlp_ratio=self.mlp_ratio,
			                   qkv_bias=qkv_bias, qk_scale=qk_scale,
			                   drop=drop_rate, attn_drop=attn_drop_rate,
			                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
			                   norm_layer=norm_layer,
			                   # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
			                   downsample=PatchMerging if (i_layer > 0) else None,
			                   use_checkpoint=use_checkpoint)
			self.layers.append(layer)

		# Modified
		self.norm = norm_layer(self.num_features)
		if not cross_layer:
			self.avgpool = nn.AdaptiveAvgPool1d(1)
			self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
		# Modified end

		self.apply(self._init_weights)
		self.use_mit = True
		self.con = False
		self.only_class = False
		self.uncertain_map = False
		self.fea_fuse = False
		if self.use_mit:
			if not self.only_class:
				# self.attn_refiner = SpatialAttentionRefiner(channels=[128, 128, 128])
				# self.attn_fuse = atten_fused()  上一版
				self.attn_fuse = new_attn()

			self.head_local = nn.ModuleList([
				nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
				for dim in [128, 256, 512]
			])
			if self.con:
				self.project = nn.Sequential(
					nn.Linear(self.num_features, 512), nn.ReLU(), nn.Linear(512, num_classes)
			)
		if self.uncertain_map:
			self.image_pyramid = ImagePyramid(7, 1)
		if self.fea_fuse == True:
			self.combiner = GCNCombiner(12240, num_classes, 1024)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			nn.init.kaiming_normal_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	@torch.jit.ignore
	def no_weight_decay(self):
		return {'absolute_pos_embed'}

	@torch.jit.ignore
	def no_weight_decay_keywords(self):
		return {'relative_position_bias_table'}

	def forward_features(self, x):
		x = self.patch_embed(x)
		if self.ape:
			x = x + self.absolute_pos_embed
		x = self.pos_drop(x)
		layer_x = []
		attns = []
		outputs = []
		for layer in self.layers:  # 4层  从这里面输出最后一层的把
			x, attn = layer(x)
			attns.append(attn[-1])  # 这个stage里有很多层 取最后一层的attn
			outputs.append(x)
			if self.cross_layer:
				layer_x.append(x)
		self.attn_globle = attns

		if self.cross_layer:
			layer_x[-1] = self.norm(layer_x[-1])
			# print(self.cross_layer)
			return layer_x

		if self.use_mit:
			outputs[-1] = self.norm(outputs[-1])
			out = [attns,outputs]
			return out

		else:
			x = self.norm(x)  # B L C
			return x



	def forward(self, x, epoch=50):
		x = self.forward_features(x)   # (16,144,1024)
		adj_fea = []
		if self.use_mit:
			four_x = []
			if self.only_class:  # 直接进行分类  走这个
				for i in range(len(x[-1])-1):
					local_x = x[-1][i]
					local_x = self.avgpool(local_x.transpose(1, 2))  # B C 1 平均池化
					local_x = torch.flatten(local_x, 1)  # 展开
					local_x_ce = self.head_local[i](local_x)  # (16,200)  全连接层映射
					four_x.append(local_x_ce)

			else:  # x:list2:(16,96,96,1)(16,48,48,1)(16,24,24,1)(16,12,12,1)  (16,9216,128)(16,2304,256)(16,576,512)(16,144,1024)
				# local_fea = self.attn_refiner(x[0],x[1])
				local_fea,attn_new_list = self.attn_fuse(x[1],x[0])  # list3:(16,1,96,96) (16,1,48,48) (16,1,24,24)
				self.attn_new = attn_new_list
				for i in range(len(local_fea)):  # (16,9216,128)  (16,2304,256)  (16,576,512)
					# B,_,_,_= local_fea[i].shape
					# attn = local_fea[i].view(B, -1, 1)
					# fused_fea = x[1][i] * attn + x[1][i] # fea×attn
					# adj_fea.append(fused_fea)
					# fused_fea = self.avgpool(fused_fea.transpose(1, 2))  # B C 1  在patch维度进行吗
					# fused_fea = torch.flatten(fused_fea, 1)
					# fused_fea = self.head_local[i](fused_fea)
					# four_x.append(fused_fea)
					fused_fea = self.avgpool(local_fea[i].transpose(1, 2))  # B C 1  (16,128,9216)-->(16,128,1)
					fused_fea = torch.flatten(fused_fea, 1)  # (16,128)
					fused_fea = self.head_local[i](fused_fea)  # 线性映射
					four_x.append(fused_fea)

				# 全局特征
			golab_x = x[-1][-1]  # (16,144,1024)
			global_atten = x[0][-1]  # ()
			# self.attn_globle = global_atten
			if self.uncertain_map:
				golab_x,u = self.image_pyramid.get_uncertain(golab_x, global_atten, epoch)
				self.attn_new = u
			self.save_feature = golab_x.mean(1)
			adj_fea.append(golab_x)
			golab_x = self.avgpool(golab_x.transpose(1, 2))  # B C 1em
			golab_x = torch.flatten(golab_x, 1)
			golab_x_ce = self.head(golab_x)  # (16,200)
			four_x.append(golab_x_ce)
			if self.con:
				golab_x_con = self.project (golab_x)
				four_x.append(golab_x_con)
			if self.fea_fuse == True:
				out = self.combiner(adj_fea)
				four_x.append(out)
			return four_x
		else:
			self.save_feature = x[-1].mean(1)  # (144)
			if not self.cross_layer:
				x = self.avgpool(x.transpose(1, 2))  # B C 1
				x = torch.flatten(x, 1)
				x = self.head(x)
		return x

	def flops(self, input_size=(1, 3, 224, 224)):
		"""
        计算 SwinTransformer + new_attn + multi-exit 的 FLOPs
        input_size: tuple (B,C,H,W)
        """
		B, C, H, W = input_size
		flops = 0

		# 1. 原始 Swin FLOPs
		flops += self.patch_embed.flops()
		for i, layer in enumerate(self.layers):
			flops += layer.flops()
		# norm
		flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
		# global head
		flops += self.num_features * self.num_classes

		# 2. new_attn FLOPs
		if hasattr(self, 'attn_fuse') and self.attn_fuse is not None:
			parts_dim = self.attn_fuse.parts_dim  # [128, 256, 512]
			for i, dim in enumerate(parts_dim):
				# 假设每个 stage 的 feature map size 随层数缩小
				h_stage = H // (2 ** i)
				w_stage = W // (2 ** i)
				N = h_stage * w_stage

				# v 投影 Linear(dim, dim)
				flops += 2 * B * N * dim * dim

				# CBAM ChannelGate
				r = 16
				pool_types = 2  # avg + max
				flops += pool_types * 2 * dim * (dim // r)

				# CBAM SpatialGate
				flops += B * dim * h_stage * w_stage * 7 * 7 * 2  # 近似 7x7 卷积，乘以 batch

				# attention 加权
				flops += B * N * dim

				# proj 投影 Linear(dim, dim)
				flops += 2 * B * N * dim * dim

		# 3. multi-exit heads FLOPs
		if hasattr(self, 'head_local') and self.head_local is not None:
			for i, dim in enumerate(self.attn_fuse.parts_dim):
				flops += 2 * B * dim * self.num_classes  # Linear(dim -> num_classes)

		return flops


def swin_backbone(**kwargs):
	""" Swin-B @ 384x384, trained ImageNet-22k
	"""
	model_kwargs = dict(
		patch_size=4, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
	return SwinTransformer(**model_kwargs)


def swin_backbone_large(**kwargs):
	""" Swin-L @ 384x384, trained ImageNet-22k
	"""
	model_kwargs = dict(
		patch_size=4, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
	return SwinTransformer(**model_kwargs)

def swin_backbone_tiny( **kwargs):
	""" Swin-T @ 224x224, trained ImageNet-1k
	"""
	model_kwargs = dict(
	    patch_size=4,  embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
	return SwinTransformer( **model_kwargs)

if __name__ == '__main__':
	x = torch.rand(1, 3, 448, 448)
	# model = swin_backbone_large(window_size=12, img_size=384, num_classes=200, cross_layer=False)
	model = swin_backbone(num_classes=200, drop_path_rate=0,img_size=448, window_size=14)
	# print(model)
	from thop import profile

	flops, params = profile(model, inputs=(x,))
	flops = model.flops()
	print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
	print('Params = ' + str(params / 1000 ** 2) + 'M')
	y = model(x)
	for i in y:
		print(i.shape)
