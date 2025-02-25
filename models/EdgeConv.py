import torch
from torch.nn import Module, Linear, ModuleList
from .utils import *
from .feature import *


class DenseEdgeConv(Module):

    def __init__(self, in_channels, num_fc_layers, growth_rate, knn=16, aggr='max', activation='relu', relative_feat_only=False):
        super().__init__()
        
        assert num_fc_layers > 2

        self.in_channels = in_channels # 24
        self.knn = knn # 16
        self.num_fc_layers = num_fc_layers # 3
        self.growth_rate = growth_rate # 12
        self.relative_feat_only = relative_feat_only

        self.layer_first = FCLayer_first(3*in_channels, 
                                         4*in_channels, 
                                         growth_rate, 
                                         bias=True)

        self.layer_last = FCLayer(in_channels + (num_fc_layers - 1) * growth_rate, 
                                  growth_rate, 
                                  bias=True, 
                                  activation=None)

        self.layers = ModuleList()

        for i in range(1, num_fc_layers-1):
            self.layers.append(MlpX(in_channels + i * growth_rate, 
                                  (in_channels + i * growth_rate) * 2, 
                                  growth_rate, 
                                  bias=True, 
                                  ))

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
        
        # Last Layer
        y = torch.cat([
            self.layer_last(y), # (B, N, K, c)
            y                   # (B, N, K, d+(L-1)*c)
        ], dim=-1)  # (B, N, K, d+L*c)

        # Pooling
        y = self.aggr(y, dim=-2)

        return y
