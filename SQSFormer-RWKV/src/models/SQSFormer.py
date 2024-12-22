from logging import config
from random import random
from random import randint

import PIL
import time, json
import math
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
import torch.nn.init as init
from einops import rearrange, repeat
import collections
import torch.nn as nn
from utils import device


def _weights_init(m):
    classname = m.__class__.__name_
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

def RWKV_Init(module):
    if not isinstance(module, nn.Module):
        raise TypeError(f"Expected an instance of nn.Module, but got {type(module)}")
    for m in module.modules():  # 遍历 nn.Module 的所有子模块
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            name = '[unknown weight]'
            for name, parameter in m.named_parameters():  # 查找参数名称
                if id(m.weight) == id(parameter):
                    break

            shape = m.weight.data.shape
            gain = 1.0 # 正向增益用于正交初始化，负向用于正态分布初始化
            scale = 1.0  # 额外的缩放因子
            print(f"Weight shape: {shape}")
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))

            if hasattr(m, 'scale_init'):
                scale = m.scale_init

            gain *= scale
            if gain == 0:
                nn.init.zeros_(m.weight)
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.normal_(m.weight, mean=0, std=-gain)

class Residual(nn.Module): #残差网络
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # 如果 fn 函数改变了维度，则通过线性层调整 x 的维度
        if x.shape[-1] != self.fn(x, **kwargs).shape[-1]:
            x = nn.Linear(x.shape[-1], self.fn(x, **kwargs).shape[-1])(x)
            print(f"x shape: {x}")
            print(f"self.fn(x, **kwargs) + x: {self.fn(x, **kwargs) + x}")


        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):   #dim = 64
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 确保 dim 是输入的最后一个维度
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class RWKV_TimeMix(nn.Module):
    def __init__(self, layer_id, params):
        super().__init__()
        self.params = params
        net_params = params['net']
        self.layer_id = layer_id  # 层的编号，用于标识不同的网络层。
        self.n_head = net_params.get("n_head")  # 注意力头的数量，设置为4。
        self.ctx_len = net_params.get("ctx_len")  # 上下文长度，设置为169。
        self.head_size = net_params.get("head_size")  # 每个注意力头的尺寸，设置为12。
        # 使用 nn.ZeroPad2d 实现的时间偏移操作，对输入序列进行时间步的移动。
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        # 可学习的时间权重参数，形状为 (n_head, ctx_len)，用于调整不同时间步的影响。
        self.time_w = nn.Parameter(torch.ones(self.n_head, self.ctx_len))  # 形状为(4, 169)。
        print("初始化 time_w.shape:", self.time_w.shape)

        self.key = nn.Linear(net_params.get('n_attn'), self.head_size)  # 输入512，输出12。
        self.value = nn.Linear(net_params.get('n_attn'), self.head_size)  # 输入512，输出12。
        self.receptance = nn.Linear(net_params.get('n_attn'), self.head_size)  # 输入512，输出12。

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, net_params.get('ctx_len')))  # 形状为(4, 1, 169)。
        print("初始化 time_alpha.shape:", self.time_alpha.shape)
        self.time_beta = nn.Parameter(torch.ones(self.n_head, net_params.get('ctx_len'), 1))  # 形状为(4, 169, 1)。
        print("初始化 time_beta.shape:", self.time_beta.shape)
        self.time_gamma = nn.Parameter(torch.ones(net_params.get('ctx_len'), 1))  # 形状为(169, 1)。
        print("初始化 time_gamma.shape:", self.time_gamma.shape)

        self.output = nn.Linear(16, 64)  # 将注意力机制的输出映射到所需维度的线性层。
        print("RWKV_TimeMix 初始化完成")

    # forward 方法处理
    def forward(self, x, params):  # x的形状 (batch_size, seq_len, dim)
        print("\n输入 x.shape:", x.shape)
        batch_size, seq_len, dim = x.shape  # batch_size = 65, seq_len = 169, dim = 512
        print("batch_size:", batch_size, "seq_len:", seq_len, "dim:", dim)

        if dim != 512:
            x = nn.Linear(dim, 512)(x)
            print("调整后的 x.shape:", x.shape)

        B, T_x, C = x.size()
        print("B:", B, "T_x:", T_x, "C:", C)
        TT = self.ctx_len  # TT = 256
        print("TT (ctx_len):", TT)

        # 构造 w
        w = F.pad(self.time_w, (0, TT))
        print("填充后的 w.shape:", w.shape)

        w = torch.tile(w, (1, TT))
        print("平铺后的 w.shape:", w.shape)

        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        print("重塑后的 w.shape:", w.shape)

        w = w[:, :, TT - 1:]
        print("切片后的 w.shape:", w.shape)

        T = T_x  # 更新 T 为 x 的序列长度
        print("更新后的 T:", T)

        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]
        print("权重计算后的 w.shape:", w.shape)

        # 之后的代码保持不变
        x_shifted = self.time_shift(x[:, :, :C // 2])
        print("时间偏移后的 x_shifted.shape:", x_shifted.shape)
        x = torch.cat([x_shifted, x[:, :, C // 2:]], dim=-1)
        print("拼接后的 x.shape:", x.shape)

        k = self.key(x)
        print("计算后的 k.shape:", k.shape)
        v = self.value(x)
        print("计算后的 v.shape:", v.shape)
        r = self.receptance(x)
        print("计算后的 r.shape:", r.shape)  #计算后的 r.shape: torch.Size([65, 144, 12])

        k = torch.clamp(k, min=-60, max=30)
        print("截断后的 k.shape:", k.shape)
        k = torch.exp(k)
        print("指数运算后的 k.shape:", k.shape)

        sum_k = torch.cumsum(k, dim=1)
        print("累加后的 sum_k.shape:", sum_k.shape)

        kv = k * v
        print("w.shape:", w.shape)
        print("kv.shape:", kv.shape)

        # 调整 kv 的形状以匹配 w
        kv = kv.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        print("重塑并转置后的 kv.shape:", kv.shape)   #([65, 1, 144, 12])

        # 进行 einsum 操作
        wkv = torch.einsum('htu,bthd->bhud', w, kv)  #([65, 144, 144, 12])
        print("计算后的 wkv.shape:", wkv.shape)
        wkv = wkv.mean(dim=1)
        # wkv = wkv.view(B, T, -1)  #([65, 144, 1728])
        # print("重塑后的 wkv.shape:", wkv.shape)

        # r_expanded = torch.sigmoid(r).unsqueeze(1)  # 变为 [65, 1, 144, 12]
        rwkv = r * wkv / (sum_k + 1e-8)

        # rwkv = torch.sigmoid(r) * wkv / (sum_k + 1e-8)
        # print("计算后的 rwkv.shape:", rwkv.shape)

        rwkv = self.output(rwkv)
        print("输出层后的 rwkv.shape:", rwkv.shape)

        rwkv = rwkv[:, :T, :]
        print("切片后的 rwkv.shape:", rwkv.shape)

        rwkv = rwkv * self.time_gamma[:T, :]
        print("最终的 rwkv.shape:", rwkv.shape)

        return rwkv

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        dropout = 0.1
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1. / 2)
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return self.weight * x_normed

class RWKV_ChannelMix(nn.Module):  #通道混合模块
    def __init__(self,  layer_id ,params):
        super(RWKV_ChannelMix, self).__init__()
        self.params = params
        net_params = params['net']
        self.n_ffn = net_params.get('n_ffn')
        self.layer_id = 0
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 使用 nn.ZeroPad2d((0, 0, 1, -1))，在时间维度上实现上移一位。
        hidden_sz = 5 * net_params.get('n_embd') // 4  # 通过 5/2 比例调整后的隐藏层大小，通常用于减少计算成本。320
        self.key = nn.Linear(net_params.get('n_embd1'), hidden_sz)  # 线性层(512, 320)
        self.value = nn.Linear(net_params.get('n_embd1'), hidden_sz)  # 线性层(512, 320)
        self.receptance = nn.Linear(net_params.get('n_embd1'), net_params.get('n_embd'))  # 线性层(512, 128)
        self.weight = nn.Linear(hidden_sz, net_params.get('n_embd'))  # (320, 128)

        # self.rmsnorm = RMSNorm(512)  # 初始化 RMSNorm，输入维度为 512

        self.output_adjust = nn.Linear(128, 64)
        self.receptance.scale_init = 0
        self.weight.scale_init = 0
        print("RWKV_ChannelMix __init__")
    def forward(self, x, params):
        self.params = params
        net_params = params['net']
        B, T, C = x.size()   #B = 65  T = 169 C = 64  B是批次大小，T是时间步长，C是通道数。
        # # 使用 RMSNorm 归一化
        # x = self.rmsnorm(x)
        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=2)
        #两个x [65, 169, 32]  拼接以后还是[65, 169, 64]
        if x.shape[-1] != net_params.get('n_attn'):
            x = nn.Linear(x.shape[-1], net_params.get('n_attn')).to(device)(x)  #x[65, 169, 128]

        k = self.key(x)   #[65, 169, 320]
        v = self.value(x) #[65, 169, 320]
        r = self.receptance(x) #[65, 169, 128]


        wkv = self.weight(F.mish(k) * v)  #使用 mish 激活函数计算，最终通过 weight 层映射到原始维度。 F.mish(k)[65, 169, 320]  wkv[65, 169, 128]
        rwkv = torch.sigmoid(r) * wkv     #rwkv size: torch.Size([65, 169, 128])将 receptance 的 sigmoid 输出与 wkv 相乘进行混合。
        rwkv = self.output_adjust(rwkv)   #rwkv size: torch.Size([65, 169, 64])通过线性层对输出维度进行调整，使最终输出的维度为 64。
        # print("RWKV_ChannelMix forward")
        return rwkv

class RWKV_TinyAttn(nn.Module):
    def __init__(self,  params):
        super().__init__()
        self.params = params
        net_params = params['net']
        self.d_attn = net_params.get('rwkv_tiny_attn')  # TinyAttn 的维度64
        self.n_head = net_params.get('rwkv_tiny_head')  # 注意力头的数量 4
        self.head_size = self.d_attn // self.n_head   #16
        self.qkv = nn.Linear(net_params.get('n_embd') // 2, self.d_attn * 3)  #192
        self.out = nn.Linear(self.d_attn, net_params.get('n_embd'))  # 512
        self.adjust_dim = nn.Linear(net_params.get('n_embd'), 64)  # 输出维度为 64
        self.out1 = nn.Linear(self.d_attn, 64)
        self.dropout = nn.Dropout(p=0.1)  # 添加 dropout 层
    def forward(self, x, mask):   #mask：形状 [B, T]，进行掩码操作
        B, T, C = x.size()  # 输入 x：[65, 169, 192]
        qkv = self.qkv(x)  # 计算 qkv：[65, 169, 64]
        q, k, v = qkv.chunk(3, dim=-1)

        if self.n_head > 1:   #重塑并转置 q、k、v
            q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)   # q.shape = [65, 4, 169, 16]
            k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)   # k.shape = [65, 4, 169, 16]
            v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)    # v.shape = [65, 4, 169, 16]
            # 应用 dropout 在 qk 之后

        qk = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))  # qk.shape = [65, 4, 169, 169]
        qk = self.dropout(qk)
        qk = qk.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))   #掩码形状扩展为 [65, 1, 1, 169]
        qk = F.softmax(qk, dim=-1)   #


        qkv = qk @ v  ## qkv.shape = [65, 4, 169, 16]

        if self.n_head > 1:
            qkv = qkv.transpose(1, 2).contiguous().view(B, T, -1)
            # qkv.transpose(1, 2)：交换维度，形状变为 [65, 169, 4, 16]，重塑回之前形状 qkv.shape = [65, 169, 64]

        out = self.out(qkv)         # out.shape = [65, 169, 512]
        # out = self.out1(qkv)

        out = self.adjust_dim(out)  # out.shape = [65, 169, 64]

        # print("RWKV_TinyAttn forward")
        return out

class RWKV(nn.Module):    #Transformer
    def __init__(self, dim, depth, layer_id, params , mlp_dim, dropout):
        super(RWKV, self).__init__()
        self.layers = nn.ModuleList([])
        self.tiny_attn =  RWKV_TinyAttn(params)
        print("RWKV init")
        for _ in range(depth):    #循环三次
            # self.layers.append(nn.ModuleList([
            #     Residual(LayerNormalize(dim, RWKV_ChannelMix( layer_id=layer_id, params=params))),
            #     Residual(LayerNormalize(dim, RWKV_TimeMix(layer_id=layer_id, params=params)))
            # ]))
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim,  RWKV_ChannelMix(layer_id=layer_id, params=params))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))
            # self.layers.append(nn.ModuleList([
            #     Residual(LayerNormalize(dim, RWKV_TimeMix(layer_id=layer_id, params=params))),
            #     Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            # ]))
    def forward(self, x, params, mask=None):  # 添加 mask 参数
        x_center_attention = []  # 记录经过 RWKV_ChannelMix 后的中心位置
        x_center_mlp = []  # 记录经过 RWKV_TimeMix 后的中心位置
        for channel, mlp in self.layers:
            # x = channel(x, params=params) # 传递 params 参数
            index = int(x.shape[1] // 2)
            # x = mlp(x)
            x_center_attention.append(x[:, index, :])
            x_center_mlp.append(x[:, index, :])
            mask = torch.ones_like(x[:, :, 0])  # 创建掩码
            x = self.tiny_attn(x, mask)  #
            # print("RWKV_TinyAttn forward")

        return x, x_center_attention, x_center_mlp

class SE(nn.Module):
    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)
        print("SE init")
    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        print("SE forward")
        return F.sigmoid(out)

def layer_id(args):
    pass

class SQSFormer(nn.Module):

    def __init__(self, params):
        super(SQSFormer, self).__init__()
        self.params = params
        net_params = params['net']
        data_params = params['data']
        self.model_type = net_params.get("model_type", 0)

        num_classes = data_params.get("num_classes", 16)
        self.patch_size = data_params.get("patch_size", 13)
        self.spectral_size = data_params.get("spectral_size", 200)
        self.serve_patch_size = self.patch_size


        mlp_dim = net_params.get("mlp_dim")
        depth = net_params.get("depth")
        kernal = net_params.get('kernal')
        padding = net_params.get('padding')
        dropout = net_params.get("dropout")
        conv2d_out = 64
        dim = net_params.get('dim')
        dim_heads = dim

        self.use_mask = net_params.get('use_mask')
        # mask_size_list = [i for i in range(1, self.patch_size + 1, 2)]
        mask_size_list = [1,3,5] #########
        self.mask_list = self.generate_mask(self.patch_size ** 2, mask_size_list)
        print("[check], use_mask=%s, mask_list_size=%s" % (self.use_mask, len(self.mask_list)), mask_size_list)



        image_size = self.patch_size * self.patch_size

        self.pixel_patch_embedding = nn.Linear(conv2d_out, dim)

        self.local_trans_pixel = RWKV(dim, depth, layer_id, params, mlp_dim, dropout)
        self.new_image_size = image_size

        self.pixel_pos_embedding = nn.Parameter(torch.randn(self.new_image_size, dim))
        self.pixel_pos_embedding_relative = nn.Parameter(torch.randn(self.new_image_size, dim))
        self.pixel_pos_scale = nn.Parameter(torch.ones(1) * 0.01)
        self.center_weight = nn.Parameter(torch.ones(depth, 1, 1) * 0.01)
        # self.center_weight = nn.Parameter(torch.ones(depth, 1, 1) * 0.001)
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.spectral_size, out_channels=conv2d_out, kernel_size=(kernal, kernal), padding=(padding,padding)),
            nn.BatchNorm2d(conv2d_out),
            nn.ReLU(),
            # nn.Conv2d(in_channels=conv2d_out,out_channels=dim,kernel_size=3,padding=1),
            # nn.BatchNorm2d(dim),
            # nn.ReLU()
        )

        self.senet = SE(conv2d_out, 5)

        self.cls_token_pixel = nn.Parameter(torch.randn(1, 1, dim))
        self.to_latent_pixel = nn.Identity()

        self.mlp_head =nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.mlp_head.weight)
        torch.nn.init.normal_(self.mlp_head.bias, std=1e-6)
        self.dropout = nn.Dropout(0.1)

        linear_dim = dim * 3 ###################
        self.classifier_mlp = nn.Sequential(
            nn.Linear(dim, linear_dim),
            nn.BatchNorm1d(linear_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(linear_dim, num_classes),
        )

    def centerlize(self, x):      #中心旋转****************RIPE  Rotation-Invariant Position Embedding Module
        x = rearrange(x, 'b s h w-> b h w s')
        b, h, w, s = x.shape
        center_w = w // 2
        center_h = h // 2
        center_pixel = x[:,center_h, center_w, :]
        center_pixel = torch.unsqueeze(center_pixel, 1)
        center_pixel = torch.unsqueeze(center_pixel, 1)
        x_pixel = x +  center_pixel
        x_pixel = rearrange(x_pixel, 'b h w s-> b s h w')
        return x_pixel

    def get_position_embedding(self, x, center_index, cls_token=False):    #随机中心位置嵌入************RIPE
        center_h, center_w = center_index
        b, s, h, w = x.shape
        pos_index = []
        for i in range(h):
            temp_index = []
            for j in range(w):
                temp_index.append(max(abs(i-center_h), abs(j-center_w)))
            pos_index.append(temp_index[:])
        pos_index = np.asarray(pos_index)
        pos_index = pos_index.reshape(-1)
        if cls_token:
            pos_index = np.asarray([-1] + list(pos_index))
        pos_emb = self.pixel_pos_embedding_relative[pos_index, :]
        return pos_emb

    def generate_mask(self, seq_num, mask_size_list):  #生成mask
        # seq int
        # return [1, seq_num, seq_num]
        res_mask = []

        center_index = seq_num // 2 #序列长度
        for mask_size in mask_size_list:
            gap_index = (mask_size ** 2) // 1
            #mask的范围
            start_index = center_index - gap_index
            end_index = center_index + gap_index
            ll = np.asarray([0] * seq_num)
            ll[start_index:end_index + 1] = 1  #生成一个长度为 seq_num 的全零数组 ll，在 start_index 到 end_index 之间的值设置为 1，以此表示被遮住的部分
            ll = ll.reshape([1, -1])
            res_mask.append(torch.from_numpy(ll))   #mask数量
        return res_mask

    def random_mask(self, patch_size):
        p_center = (randint(0, 100) < 50)  #一半中心mask，一半非中心mask
        if p_center:
            # 中心mask
            return self.mask_list[randint(0, len(self.mask_list) - 1)]  #从列表随机取一个
        else:
            # 非中心mask
            max_left = patch_size // 2 - 1  #左侧的最大索引
            real_left = randint(0, max_left)  #左边界随机位置
            min_right = patch_size // 2 + 1  #右侧的最小索引
            real_right = randint(min_right, patch_size - 1) #右边界随机位置
            patch = np.zeros((patch_size, patch_size))
            patch[real_left:real_right + 1, real_left:real_right + 1] = 1
            # print(real_left, real_right)
            # print(patch)
            patch = torch.from_numpy(patch.reshape([1, -1]))
            return patch

    def get_serving_mask(self):  #生成一个以矩阵中心为基准的方形mask 中心区域的值为1，其他区域为0
        patch = np.zeros((self.patch_size, self.patch_size))
        center = self.patch_size // 2  #计算中心点
        center_l = center - self.serve_patch_size // 2  #计算左边界
        center_r = center + self.serve_patch_size // 2  #计算右边界
        patch[center_l:center_r + 1, center_l:center_r + 1] = 1   #mask标记的设成1，没标记保持0
        # print(real_left, real_right)
        # print(patch)
        patch = torch.from_numpy(patch.reshape([1, -1]))
        return patch

    def encoder_block(self, x):
        x_pixel = x
        b, s, w, h = x_pixel.shape
        x_pixel = self.conv2d_features(x_pixel)
        pos_emb = self.get_position_embedding(x_pixel, (h // 1, w // 1), cls_token=False)
        x_pixel = rearrange(x_pixel, 'b s w h -> b (w h) s')  # (batch, w*h, s)
        x_pixel = x_pixel + torch.unsqueeze(pos_emb, 0) * self.pixel_pos_scale
        x_pixel = self.dropout(x_pixel)

        x_pixel, x_center_list, x_center_mlp = self.local_trans_pixel(x_pixel, self.params)
        x_center_tensor = torch.stack(x_center_list, dim=0)  # [depth, batch, dim]
        logit_pixel = torch.sum(x_center_tensor * self.center_weight, dim=0)
        logit_x = logit_pixel
        # logit_x = x_center_list[-1]
        reduce_x = torch.mean(x_pixel, dim=1)

        cur_mask_2 = self.random_mask(self.patch_size)

        x_pixel, x_center_list, _ = self.local_trans_pixel(x_pixel, mask=cur_mask_2, params=self.params )
        #
        cur_mask_3 = self.random_mask(self.patch_size)

        x_pixel, x_center_list, _ = self.local_trans_pixel(x_pixel, mask=cur_mask_3, params=self.params)
        #
        cur_mask_4 = self.random_mask(self.patch_size)

        x_pixel, x_center_list, _ = self.local_trans_pixel(x_pixel, mask=cur_mask_4, params=self.params)

        cur_mask_5 = self.random_mask(self.patch_size)

        x_pixel, x_center_list, _ = self.local_trans_pixel(x_pixel, mask=cur_mask_5, params=self.params)

        # shape [batch, seq, dim]

        # print(f"logit_x shape: {logit_x.shape}")
        return logit_x, reduce_x

        # return logit_x, reduce_x

    def forward(self, x,left=None,right=None):
        '''
        x: (batch, s, w, h), s=spectral, w=weigth, h=height

        '''

        logit_x, _ = self.encoder_block(x)
        mean_left, mean_right = None, None
        if left is not None and right is not None:
            _, mean_left = self.encoder_block(left)
            _, mean_right = self.encoder_block(right)

        return  self.classifier_mlp(logit_x), mean_left, mean_right




