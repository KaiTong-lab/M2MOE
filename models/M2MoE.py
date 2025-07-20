import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch import einsum
import numpy as np
from torch import Tensor
from typing import List, Optional, Tuple, Union
from models.common import RevIN
from models.Embed import PatchEmbedding
from einops import rearrange
from torch.distributions.normal import Normal

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, downsample=False):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1, stride) if downsample or in_channels != out_channels else None

    def forward(self, x):  # [B, C, T]
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class FPN(nn.Module):
    def __init__(self, patch_num, d_model, layers: List[int]):
        super(FPN, self).__init__()
        self.d_model = d_model
        self.patch_num = patch_num

        self.stage1 = self._make_layer(d_model, 256, layers[0])
        self.stage2 = self._make_layer(256, 256, layers[1], stride=2)
        self.stage3 = self._make_layer(256, 256, layers[2], stride=2)
        self.stage4 = self._make_layer(256, 256, layers[3], stride=2)

        # 尺度投影
        self.linearp2 = nn.Linear(256 * ((patch_num - 1) // 1 + 1), patch_num * d_model)
        self.linearp3 = nn.Linear(256 * ((patch_num - 1) // 2 + 1), patch_num * d_model)
        self.linearp4 = nn.Linear(256 * ((patch_num - 1) // 4 + 1), patch_num * d_model)
        self.linearp5 = nn.Linear(256 * ((patch_num - 1) // 8 + 1), patch_num * d_model)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(TCNBlock(in_channels, out_channels, stride=stride, downsample=True))
        for _ in range(1, blocks):
            layers.append(TCNBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):  # x: [B*N, d_model, patch_num]
        p2 = self.stage1(x)     # [B, 256, L/2]
        p3 = self.stage2(p2)    # [B, 256, L/4]
        p4 = self.stage3(p3)    # [B, 256, L/8]
        p5 = self.stage4(p4)    # [B, 256, L/16]

        # 拉平 -> 映射为统一维度
        p2 = self.linearp2(p2.flatten(1)).reshape(x.size(0), self.patch_num, self.d_model)
        p3 = self.linearp3(p3.flatten(1)).reshape(x.size(0), self.patch_num, self.d_model)
        p4 = self.linearp4(p4.flatten(1)).reshape(x.size(0), self.patch_num, self.d_model)
        p5 = self.linearp5(p5.flatten(1)).reshape(x.size(0), self.patch_num, self.d_model)

        return p2, p3, p4, p5


class DualAttention(nn.Module):
    """
    双重注意力机制：
    1）Patch内注意力：对每个patch内部的d_model维度做注意力
    2）Patch间注意力：对patch序列维度做注意力
    """
    def __init__(self, d_model, patch_num, num_heads=4):
        super(DualAttention, self).__init__()
        self.patch_num = patch_num
        self.d_model = d_model
        self.num_heads = num_heads
        self.scale = (d_model // num_heads) ** -0.5
        
        # qkv 线性层（统一处理）
        self.qkv_patch = nn.Linear(d_model, d_model * 3)    # patch内attention
        self.qkv_seq = nn.Linear(d_model, d_model * 3)      # patch间attention
        
        self.out_patch = nn.Linear(d_model, d_model)
        self.out_seq = nn.Linear(d_model, d_model)
        self.a = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        """
        x: [batch_size, patch_num, d_model]
        """
        B, N, C = x.shape # [batch*nvars, patch_num, d_model]
        H = self.num_heads # 4
        
        # === Patch内注意力 ===
        # 对每个patch向量的维度d_model做注意力，视为序列长度为C
        # 先转置成 [B, N, C] -> [B*N, C]，然后形变成多头
        x_patch = x.reshape(B*N, C)  # 合并batch和patch
        qkv = self.qkv_patch(x_patch).reshape(B*N, 3, H, C//H).permute(1,2,0,3)  # 3 x H x (B*N) x d_head
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [H, B*N, d_head]
        attn_patch = (q @ k.transpose(-2, -1)) * self.scale  # [H, B*N, B*N]，这里k,q都为(B*N, d_head)，会导致计算量大，通常需要简化
        attn_patch = attn_patch.softmax(dim=-1)
        out_patch = (attn_patch @ v)  # [H, B*N, d_head]
        out_patch = out_patch.transpose(0,1).reshape(B*N, C)
        out_patch = self.out_patch(out_patch)
        out_patch = out_patch.reshape(B, N, C)
        
        # === Patch间注意力 ===
        # 对patch序列维度做注意力
        qkv_seq = self.qkv_seq(x).reshape(B, N, 3, H, C//H).permute(2,0,3,1,4) # 3 x B x H x N x d_head
        q, k, v = qkv_seq[0], qkv_seq[1], qkv_seq[2]  # each [B, H, N, d_head]
        attn_seq = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn_seq = attn_seq.softmax(dim=-1)
        out_seq = (attn_seq @ v)  # [B, H, N, d_head]
        out_seq = out_seq.transpose(1,2).reshape(B, N, C)  # B, N, C
        out_seq = self.out_seq(out_seq)

        # out = self.a * out_patch + (1 - self.a) * out_seq
        
        # 融合两个注意力输出
        out = out_patch + out_seq  # 简单加和，也可加权融合
        return out


class SEFusion(nn.Module):
    """
    基于SE的融合模块，用于融合不同尺度特征
    输入多尺度张量列表，输出加权融合结果
    """
    def __init__(self, d_model, num_scales=4, reduction=16):
        super(SEFusion, self).__init__()
        self.num_scales = num_scales
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 对patch_num维度做池化
        self.fc1 = nn.Linear(d_model, d_model // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model // reduction, num_scales)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, features):
        """
        features: list of tensors, 每个tensor形状 [B, patch_num, d_model]
        """
        # 先对每个特征做全局平均池化，得到 [B*nvars, d_model]
        pooled_feats = [self.avgpool(f.permute(0, 2, 1)).squeeze(-1) for f in features]
        # 堆叠成 [B*nvars, num_scales, d_model]
        stacked = torch.stack(pooled_feats, dim=1)
        
        # SE机制计算权重
        se = self.fc1(stacked)      # [B*nvars, num_scales, d_model//reduction]
        se = self.relu(se)
        se = self.fc2(se)           # [B*nvars, num_scales, num_scales]

        # 这里对num_scales维度做softmax归一化权重
        # 取对角线元素作为每个尺度权重
        weights = torch.diagonal(se, dim1=1, dim2=2)  # [B*nvars, num_scales]
        weights = self.softmax(weights)                              # [B*nvars, num_scales]
        
        # 加权融合
        fused = 0
        for i in range(self.num_scales):
            w = weights[:, i].unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
            fused += features[i] * w
        return fused


class MultiScaleModule(nn.Module):
    """
    结合FPN输出多尺度特征，经过双重注意力和SE融合模块
    """
    def __init__(self, patch_num, d_model, num_heads=4):
        super(MultiScaleModule, self).__init__()
        self.d_model = d_model
        self.patch_num = patch_num
        self.num_heads = num_heads
        self.num_scales = 4
        
        self.dual_attentions = nn.ModuleList([DualAttention(d_model, patch_num, num_heads) for _ in range(self.num_scales)])
        self.se_fusion = SEFusion(d_model, self.num_scales)
        # self.se_fusion = GatedMLPFusion(d_model, self.num_scales)

        
    def forward(self, p2, p3, p4, p5):
        # p_i: [B, patch_num, d_model]
        attn_feats = []
        for i, feat in enumerate([p2, p3, p4, p5]):
            attn_feat = self.dual_attentions[i](feat)
            attn_feats.append(attn_feat)
        fused_output = self.se_fusion(attn_feats)
        return fused_output  # [B, patch_num, d_model]
    
class FlattenHead(nn.Module):
    def __init__(self,  nf, target_window, head_dropout=0):
        super().__init__()
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """
        gates: Tensor of shape [batch_size, num_experts]
        """
        self._gates = gates  # [B, E]
        self._num_experts = num_experts

        # 获取非零门控位置 (batch_idx, expert_idx)
        nonzero_indices = torch.nonzero(gates, as_tuple=False)
        # 排序以保证一致性
        sorted_experts, index_sorted = nonzero_indices.sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)  # 每个 token 分配给哪个 expert
        self._batch_index = nonzero_indices[index_sorted[:, 1], 0]  # 每个 token 来自哪个 batch

        self._part_sizes = (gates > 0).sum(0).tolist()  # 每个 expert 接收多少 token

        gates_exp = gates[self._batch_index]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)  # 每个 token 的 gate 值

    def dispatch(self, inp):
        """
        inp: Tensor of shape [B, ..., D]
        返回一个 list，每个元素是分配给某个 expert 的输入
        """
        dispatched = inp[self._batch_index]  # [sum(N_i), ...]
        return torch.split(dispatched, self._part_sizes, dim=0)

    def combine(self, expert_outs, multiply_by_gates=True):
        """
        expert_outs: list of expert outputs, each of shape [N_i, ..., D]
        返回合并后的输出: Tensor of shape [B, ..., D]
        """
        stitched = torch.cat(expert_outs, dim=0)  # [sum(N_i), ...]

        if multiply_by_gates:
            while self._nonzero_gates.dim() < stitched.dim():
                self._nonzero_gates = self._nonzero_gates.unsqueeze(-1)
            stitched = stitched * self._nonzero_gates  # 广播相乘

        output_shape = [self._gates.size(0)] + list(stitched.shape[1:])
        zeros = torch.zeros(*output_shape, device=stitched.device, dtype=stitched.dtype)
        combined = zeros.index_add(0, self._batch_index, stitched)

        return combined

    def expert_to_gates(self):
        """
        返回每个 expert 对应的门控值 list
        """
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class M2MoE(nn.Module):
    def __init__(self, input_shape, seq_len, pred_len, patch_configs, moe_k, loss_coef, dropout, norm=True, layernorm=True):
        super(M2MoE, self).__init__()

        self.moe_k = moe_k
        self.patch_configs = patch_configs
        self.noisy_gating = True
        self.loss_coef = loss_coef
        self.models = nn.ModuleList()
        self.k = moe_k
        self.seq_len = seq_len
        self.input_dim = input_shape[-1]

        for patch_len, stride in patch_configs:
            model = M2M(input_shape=input_shape,
                        seq_len=seq_len,
                        pred_len=pred_len,
                        dropout=dropout,               
                        norm=norm,
                        layernorm=layernorm,
                        patch_len=patch_len,
                        stride=stride)
            self.models.append(model)

        num_experts = len(self.models)
        self.w_gate = nn.Parameter(torch.zeros(self.input_dim, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(self.input_dim, num_experts))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def forward(self, x):
        B, T, D = x.shape# [B, seq_len, input_dim]
        x_summary = x.mean(dim=1)  # [B, D]
        gates, load = self.noisy_top_k_gating(x_summary, self.training) # [B, n_experts]

        dispatcher = SparseDispatcher(len(self.models), gates)
        expert_inputs = dispatcher.dispatch(x)

        expert_outputs, expert_losses = [], []
        for i, model in enumerate(self.models):
            if len(expert_inputs[i]) == 0:
                expert_outputs.append(torch.zeros(0, self.models[0].pred_len, D, device=x.device))
                expert_losses.append(torch.tensor(0.0, device=x.device))
            else:
                out, loss = model(expert_inputs[i])
                expert_outputs.append(out)
                expert_losses.append(loss)

        y = dispatcher.combine(expert_outputs)

        # balance loss
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= self.loss_coef

        return y, balance_loss

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate  # [B, n_experts]

        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.w_gate.shape[1]), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.w_gate.shape[1] and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = (gates > 0).sum(0).float()
        return gates, load

    def _prob_in_top_k(self, clean, noisy, noise_std, top_logits):
        B, M = clean.shape
        topk = top_logits.shape[1]
        flat = top_logits.flatten()
        pos_if_in = torch.arange(B, device=clean.device) * topk + self.k
        thresh_in = flat.gather(0, pos_if_in).unsqueeze(1)
        is_in = noisy > thresh_in

        pos_if_out = pos_if_in - 1
        thresh_out = flat.gather(0, pos_if_out).unsqueeze(1)

        normal = Normal(self.mean, self.std)
        prob_in = normal.cdf((clean - thresh_in) / noise_std)
        prob_out = normal.cdf((clean - thresh_out) / noise_std)

        return torch.where(is_in, prob_in, prob_out)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0.], device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

class M2M(nn.Module):


    def __init__(self, input_shape,seq_len, pred_len,  dropout, norm=True, layernorm=True,patch_len=16, stride=16):
        super(M2M, self).__init__()

        self.norm = norm
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = 128


        self.patch_len = patch_len
        self.stride = stride
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, dropout)
        if self.norm:
            self.rev_norm = RevIN(input_shape[-1])
        self.patch_num = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.fpn = FPN(self.patch_num, self.d_model, layers=[1, 1, 1, 1])
        self.multi_scale_module = MultiScaleModule(self.patch_num, self.d_model, num_heads=4)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.d_model)
        self.activation = F.relu
        self.conv1 = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model*4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.d_model*4, out_channels=self.d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.head_nf = self.d_model * self.patch_num
        self.head = FlattenHead(self.head_nf, self.pred_len,
                                head_dropout=dropout)
        self.nvars = input_shape[-1]
        self.heads = nn.ModuleList([
            FlattenHead(self.d_model * self.patch_num, self.pred_len, head_dropout=dropout)
            for _ in range(self.nvars)
        ])

    def forward(self, x): # [batch_size, seq_len, dim]
        # [batch_size, seq_len, feature_num]
        # layer norm
        if self.norm:
            x = self.rev_norm(x, 'norm')
        # [batch_size, seq_len, feature_num]

        # [batch_size, seq_len, feature_num]
        x = torch.transpose(x, 1, 2)
        # [batch_size, feature_num, seq_len]
        x_emb, dim = self.patch_embedding(x) # [b*nvars,p_n, d_model] 
        x_enc = x_emb.permute(0, 2, 1)  # [B, d_model, patch_num]
        p2, p3, p4, p5 = self.fpn(x_enc)
        out = self.multi_scale_module(p2, p3, p4, p5)
        out = x_emb + self.dropout(out)
        out = self.norm(out)
        y = self.dropout(self.activation(self.conv1(out.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        out = self.norm1(y + out)
        out = torch.reshape(
            out, (-1, dim, out.shape[-2], out.shape[-1])) # [b,nvars, p_n, d_model]
        dec_out = self.head(out)
        dec_out = dec_out.permute(0, 2, 1)# [batch_size, pred_len, feature_num]

        if self.norm:
            dec_out = self.rev_norm(dec_out, 'denorm', self.target_slice) # [batch_size, pred_len, feature_num]
        # [batch_size, pred_len, feature_num]

        if self.target_slice:
            dec_out = dec_out[:, :, self.target_slice] # [batch_size, pred_len, feature_num]
        moe_loss =0
        return dec_out, moe_loss # [batch_size, pred_len, feature_num]