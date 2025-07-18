import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlockConv1d(nn.Module):
    """ 1D-Convolutional ResNet block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                norm_method='batch_norm', legacy=False):
        super().__init__()

        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if norm_method == 'batch_norm':
            norm = nn.BatchNorm1d
        elif norm_method == 'sync_batch_norm':
            norm = nn.SyncBatchNorm
        else:
             raise Exception("Invalid norm method: %s" % norm_method)

        self.bn_0 = norm(size_in)
        self.bn_1 = norm(size_h)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.fc_c = nn.Conv1d(c_dim, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

        

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        out = x_s + dx + self.fc_c(c)

        return out


class Enhanced(nn.Module):
    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm', dropout_rate=0.1):
        super().__init__()
        
        # 设置维度
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in
        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        norm = nn.BatchNorm1d if norm_method == 'batch_norm' else nn.SyncBatchNorm
        self.main_path = nn.Sequential(
            norm(size_in),
            nn.GELU(),  
            nn.Conv1d(size_in, size_h, 1),
            nn.Dropout(dropout_rate),
            norm(size_h),
            nn.GELU(),
            nn.Conv1d(size_h, size_out, 1),
        )
        self.attention = nn.Sequential(
            nn.Conv1d(size_out, size_out // 4, 1),
            nn.GELU(),
            nn.Conv1d(size_out // 4, size_out, 1),
            nn.Sigmoid()
        )
        self.condition_path = nn.Sequential(
            nn.Conv1d(c_dim, c_dim // 2, 1),
            nn.GELU(),
            nn.Conv1d(c_dim // 2, size_out, 1)
        )
        self.shortcut = None if size_in == size_out else \
            nn.Conv1d(size_in, size_out, 1, bias=False)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(size_out, size_out // 8, 1),
            nn.GELU(),
            nn.Conv1d(size_out // 8, size_out, 1),
            nn.Sigmoid()
        )
        
        self.output_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x, c):
        main = self.main_path(x)
        att = self.attention(main)
        main = main * att
        se_weight = self.se(main)
        main = main * se_weight
        cond = self.condition_path(c)
        skip = self.shortcut(x) if self.shortcut else x
        out = skip + main + cond
        return out * self.output_scale



class ScoreNet(nn.Module):

    def __init__(self, z_dim, dim, out_dim, hidden_size, num_blocks):
        """
        Args:
            z_dim:   Dimension of context vectors. 
            dim:     Point dimension.
            out_dim: Gradient dim.
            hidden_size:   Hidden states dim.
        """
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        # Input = Conditional = zdim (code) + dim (xyz)
        c_dim = z_dim + dim
        # print(c_dim, hidden_size) # 63, 128
        self.conv_p = nn.Conv1d(c_dim, hidden_size, 1)
        self.blocks = nn.ModuleList([
            ResnetBlockConv1d(c_dim, hidden_size) for _ in range(num_blocks)
        ])

        self.bn_out = nn.BatchNorm1d(hidden_size)
        self.conv_out = nn.Conv1d(hidden_size, out_dim, 1)
        self.actvn_out = nn.ReLU()

        # 添加特征融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_size*2, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

    def forward(self, x, c):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :param c: (bs, self.zdim) Shape latent code
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        p = x.transpose(1, 2)  # (bs, dim, n_points)
        batch_size, D, num_points = p.size()

        c_expand = c.unsqueeze(2).expand(-1, -1, num_points)
        c_xyz = torch.cat([p, c_expand], dim=1) # 4096, 63, 32
        net = self.conv_p(c_xyz)
        for block in self.blocks:
            net = block(net, c_xyz)
        out = self.conv_out(self.actvn_out(self.bn_out(net))).transpose(1, 2)
        return out