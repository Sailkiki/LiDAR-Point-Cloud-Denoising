import torch
from torch.nn import Module, Linear, ModuleList
from .utils import *
from .feature import *
from .EdgeConv import DenseEdgeConv

class NoiseRegionTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, d_model=16, num_classes=60):
        super(NoiseRegionTransformer, self).__init__()
        
        # Transformer的配置
        self.d_model = d_model  # 特征维度
        self.num_heads = num_heads  # 注意力头数
        self.num_layers = num_layers  # Transformer层数
        
        # 位置编码
        self.position_encoding = nn.Parameter(torch.randn(1, 1024, d_model))  # 假设最大点数为1024

        # 自注意力层
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads) for _ in range(num_layers)
        ])
        
        # 线性层用于转换特征维度
        self.fc_in = nn.Linear(input_dim, d_model)
        
        # 多层感知机（Feed Forward）
        self.fc_out = nn.Linear(d_model, num_classes)
        
        # 噪声区域加权模块
        self.noise_attention = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, 1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        :param x: 输入点云特征，大小为 (B, N, D) 其中 B 是批大小，N 是点数，D 是特征维度
        :return: 噪声去除后的点云特征，大小为 (B, N, D)
        """
        # print(x.shape)
        # 获取批次大小和点云中的点数
        B, N, D = x.size()
        
        # 输入特征线性转换
        x = self.fc_in(x)  # 转换为 d_model 维度
        # 加上位置编码
        x = x + self.position_encoding[:, :N, :]
        # Transformer层的处理
        for layer in self.attention_layers:
            x = layer(x)
        
        # 噪声识别：使用自注意力机制计算噪声区域的权重
        attention_weights = self.noise_attention(x.transpose(1, 2))  # (B, N, 1)
        attention_weights = attention_weights.transpose(1, 2)  # 转置为 (B, 1, N)
        
        # 对输入点云加权
        x = x * attention_weights  # 根据注意力权重加权
        # 通过全连接层进行输出
        x = self.fc_out(x)
        return x


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
        activation='relu',
        transformer_heads=2,  # Transformer的头数
        transformer_layers=1, # Transformer的层数
        transformer_d_model=32, # Transformer的特征维度
        transformer_num_classes=24 # 输出类别数
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


            self.noise_transformer = NoiseRegionTransformer(
                input_dim=24,  # 使用卷积后的特征维度
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                d_model=transformer_d_model,
                num_classes=transformer_num_classes
            )

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def dynamic_graph_forward(self, x):
        for i in range(self.num_convs):
            x = self.transforms[i](x) 
            x = self.noise_transformer(x)
            # print(x.shape)
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