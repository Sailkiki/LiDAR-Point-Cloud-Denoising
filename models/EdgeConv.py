import torch
from torch.nn import Module, Linear, ModuleList
from .utils import *
from .feature import *

class ChannelGate(Module):
    def __init__(self, channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super().__init__()
        
        self.channels = channels
        self.pool_types = pool_types
        
        # 多层感知机
        self.mlp = nn.Sequential(
            Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            Linear(channels // reduction_ratio, channels // reduction_ratio),
            nn.ReLU(),
            Linear(channels // reduction_ratio, channels)
        )
        
        # 空间注意力
        self.spatial_gate = nn.Sequential(
            Linear(2, 1),
            nn.Sigmoid()
        )
        
        # SE-style 通道注意力
        self.channel_gate = nn.Sequential(
            Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, n, k, c = x.size()  # batch, num_points, k_neighbors, channels
        
        # 1. 多池化策略
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool_feat = x.mean(dim=2)  # (B, N, C)
            elif pool_type == 'max':
                pool_feat = x.max(dim=2)[0]  # (B, N, C)
            
            if channel_att_sum is None:
                channel_att_sum = self.mlp(pool_feat)
            else:
                channel_att_sum += self.mlp(pool_feat)
                
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2)  # (B, N, 1, C)
        
        # 2. 空间注意力
        avg_pool = torch.mean(x, dim=-1, keepdim=True)  # (B, N, K, 1)
        max_pool = torch.max(x, dim=-1, keepdim=True)[0]  # (B, N, K, 1)
        spatial_feat = torch.cat([avg_pool, max_pool], dim=-1)  # (B, N, K, 2)
        spatial_att = self.spatial_gate(spatial_feat)  # (B, N, K, 1)
        
        # 3. SE-style 通道注意力
        se_feat = x.mean(dim=2)  # (B, N, C)
        channel_att = self.channel_gate(se_feat).unsqueeze(2)  # (B, N, 1, C)
        
        # 组合所有注意力机制
        att = scale * spatial_att * channel_att
        
        # 残差连接
        return x * att + x

class OffsetPredictor(Module):
    """预测每个点的位置偏移量"""
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            Linear(in_channels, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, 24),  # 输出三维偏移量 (Δx, Δy, Δz)
            nn.Tanh()               # 限制偏移量范围在[-1,1]之间
        )
        
    def forward(self, x):
        return self.mlp(x) * 0.1    # 缩放偏移量，防止过大扰动
    

class DenseEdgeConv(Module):

    def __init__(self, in_channels, num_fc_layers, growth_rate, knn=16, aggr='max',
                  activation='relu', relative_feat_only=False, use_deformable=False):
        super().__init__()
        
        assert num_fc_layers > 2

        self.in_channels = in_channels # 24
        self.knn = knn # 16
        self.num_fc_layers = num_fc_layers # 3
        self.growth_rate = growth_rate # 12
        self.relative_feat_only = relative_feat_only

        self.use_deformable = use_deformable
        if self.use_deformable:
            self.offset_predictor = OffsetPredictor(in_channels)
            self.adaptive_knn = int(knn * 1.5)  # 扩大候选邻域数量

        if self.use_deformable:
            self.edge_attention = nn.Sequential(
                Linear(in_channels, 12), # (24, 12)
                nn.ReLU(),
                Linear(12, 16)
            )

        self.channel_gate1 = ChannelGate(in_channels + growth_rate)
        self.channel_gate2 = ChannelGate(in_channels + 2 * growth_rate)
        self.channel_gate3 = ChannelGate(in_channels + 3 * growth_rate)
        self.layer_first = FCLayer_first(3*in_channels, 4*in_channels, growth_rate, bias=True)
        self.layer_last = FCLayer(in_channels + (num_fc_layers - 1) * growth_rate, growth_rate, bias=True, activation=None)
        self.layers = ModuleList()
        for i in range(1, num_fc_layers-1):
            self.layers.append(FCLayer(in_channels + i * growth_rate, growth_rate, bias=True, activation=activation))

        self.aggr = Aggregator(aggr)

    @property
    def out_channels(self):
        return self.in_channels + self.num_fc_layers * self.growth_rate

    def get_edge_feature(self, x, knn_idx):
        """
        :param  x:          (B, N, d)
        :param  knn_idx:    (B, N, K)
        :return (B, N, K, 2*d)
        """
        knn_feat = knn_group(x, knn_idx)   # B * N * K * d
        x_tiled = x.unsqueeze(-2).expand_as(knn_feat)

        edge_feat = torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=3)
        return edge_feat

    def forward(self, x, pos):
        """
        :param  x:  (B, N, d)
        :return (B, N, d+L*c)
        """
        if self.use_deformable:
            # 预测位置偏移量
            offsets = self.offset_predictor(x) 
            deformed_pos = pos + offsets
            # 基于变形后坐标计算扩展的KNN索引
            knn_idx = get_knn_idx(deformed_pos, deformed_pos, k=self.adaptive_knn, offset=1)
            # 重要性采样：选择最相关的k个邻居
            edge_feat = self.get_edge_feature(x, knn_idx)  # (B, N, K, 3d)
            attention = torch.sigmoid(self.edge_attention(edge_feat.mean(dim=-1)))  # (B, N, K)
            topk_idx = torch.topk(attention, self.knn, dim=-1)[1]  # (B, N, k)
            knn_idx = torch.gather(knn_idx, -1, topk_idx)  # 最终knn索引
        else:
            knn_idx = get_knn_idx(pos, pos, k=self.knn, offset=1)
        
        # First Layer
        edge_feat = self.get_edge_feature(x, knn_idx)
        y = torch.cat([
            self.layer_first(edge_feat),              # (B, N, K, c)
            x.unsqueeze(-2).repeat(1, 1, self.knn, 1) # (B, N, K, d)
        ], dim=-1)  # (B, N, K, d+c)


        # Intermediate Layers
        for layer in self.layers:
            y = torch.cat([
                layer(y),           # (B, N, K, c)
                y,                  # (B, N, K, c+d)
            ], dim=-1)  # (B, N, K, d+c+...)
        
        y = self.channel_gate2(y)
        # Last Layer
        y = torch.cat([
            self.layer_last(y), # (B, N, K, c)
            y                   # (B, N, K, d+(L-1)*c)
        ], dim=-1)  # (B, N, K, d+L*c)

        # Pooling
        y = self.aggr(y, dim=-2)

        return y