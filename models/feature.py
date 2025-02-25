import torch
from torch.nn import Module, Linear, ModuleList
from .utils import *
from .feature import *
from .EdgeConv import DenseEdgeConv


class FeatureExtraction(Module):

    def __init__(self, 
        in_channels=3, 
        dynamic_graph=True, 
        conv_channels=24, 
        num_convs=4, 
        conv_num_fc_layers=3, 
        conv_growth_rate=12, 
        conv_knn=16, 
        conv_aggr='max', 
        activation='relu'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dynamic_graph = dynamic_graph
        self.num_convs = num_convs

        # Edge Convolution Units
        self.transforms = ModuleList()
        self.convs = ModuleList()
        for i in range(num_convs):
            if i == 0:
                trans = FCLayer(in_channels, conv_channels, bias=True, activation=None)
                conv = DenseEdgeConv(
                    conv_channels, 
                    num_fc_layers=conv_num_fc_layers, 
                    growth_rate=conv_growth_rate, 
                    knn=conv_knn, 
                    aggr=conv_aggr, 
                    activation=activation,
                    relative_feat_only=True,
                )
            else:
                trans = FCLayer(in_channels, conv_channels, bias=True, activation=activation)
                conv = DenseEdgeConv(
                    conv_channels, 
                    num_fc_layers=conv_num_fc_layers, 
                    growth_rate=conv_growth_rate, 
                    knn=conv_knn, 
                    aggr=conv_aggr, 
                    activation=activation,
                    relative_feat_only=False,
                )
            self.transforms.append(trans)
            self.convs.append(conv)
            in_channels = conv.out_channels

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def dynamic_graph_forward(self, x):
        for i in range(self.num_convs):
            x = self.transforms[i](x)
            x = self.convs[i](x, x)
        return x

    def static_graph_forward(self, pos):
        x = pos
        for i in range(self.num_convs):
            x = self.transforms[i](x)
            x = self.convs[i](x, pos)
        return x 

    def forward(self, x):
        if self.dynamic_graph:
            return self.dynamic_graph_forward(x)
        else:
            return self.static_graph_forward(x)