import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 8
        self.filters_f_g = self.inter_channels
        self.filters_h = in_channels

        # gamma is a learnable scalar initialized to 0
        self.gamma = nn.Parameter(torch.zeros(1))

        # f, g: 1x1 convs to reduce channels
        self.f = nn.Conv2d(in_channels, self.filters_f_g, kernel_size=1)
        self.g = nn.Conv2d(in_channels, self.filters_f_g, kernel_size=1)
        
        # h: 1x1 conv to original channels
        self.h = nn.Conv2d(in_channels, self.filters_h, kernel_size=1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        output: [B, C, H, W]
        """
        curr_batch_size, C, H, W = x.size()
        N = H * W

        # f: [B, C', H, W] -> [B, C', N] -> permute -> [B, N, C']
        f_x = self.f(x).view(curr_batch_size, self.filters_f_g, -1).permute(0, 2, 1)
        
        # g: [B, C', H, W] -> [B, C', N]
        g_x = self.g(x).view(curr_batch_size, self.filters_f_g, -1)
        
        f_x = self.f(x).view(curr_batch_size, self.filters_f_g, -1) # [B, C', N]
        g_x = self.g(x).view(curr_batch_size, self.filters_f_g, -1) # [B, C', N]
        h_x = self.h(x).view(curr_batch_size, self.filters_h, -1)   # [B, C, N]
        
        s = torch.bmm(g_x.permute(0, 2, 1), f_x) # [B, N, N]
        
        beta = F.softmax(s, dim=-1)
        
        o = torch.bmm(beta, h_x.permute(0, 2, 1)) # [B, N, C]
        o = o.permute(0, 2, 1).contiguous() # [B, C, N]
        o = o.view(curr_batch_size, self.filters_h, H, W)
        
        x = self.gamma * o + x
        
        return x


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    """
    def __init__(self, in_channels, reduction_ratio=0.125):
        super(ChannelAttention, self).__init__()
        reduced_channels = int(in_channels * reduction_ratio)
        if reduced_channels == 0:
            reduced_channels = 1
            
        # Shared MLP
        # GlobalPooling in PyTorch leaves (B, C, 1, 1).
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, in_channels),
            nn.ReLU() # Keras code has ReLU at the end too!
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        batch, channels, _, _ = x.size()
        
        # Global Max Pooling
        max_pool = F.max_pool2d(x, kernel_size=x.size()[2:]) # [B, C, 1, 1]
        
        # Global Avg Pooling
        avg_pool = F.avg_pool2d(x, kernel_size=x.size()[2:]) # [B, C, 1, 1]
        
        # MLP
        mlp_max = self.mlp(max_pool) # [B, C]
        mlp_avg = self.mlp(avg_pool) # [B, C]
        
        # Sum
        out = mlp_max + mlp_avg # [B, C]
        out = torch.sigmoid(out) # [B, C]
        
        # Reshape for broadcasting
        out = out.view(batch, channels, 1, 1)
        
        # Multiply
        return x * out

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        
        # Max along channel axis
        # torch.max returns (values, indices)
        max_pool, _ = torch.max(x, dim=1, keepdim=True) # [B, 1, H, W]
        
        # Mean along channel axis
        avg_pool = torch.mean(x, dim=1, keepdim=True) # [B, 1, H, W]
        
        # Concat
        concat = torch.cat([max_pool, avg_pool], dim=1) # [B, 2, H, W]
        
        # Conv
        out = self.conv(concat) # [B, 1, H, W]
        out = self.sigmoid(out)
        
        return out

class CSA(nn.Module):
    """
    Channel-Spatial Attention Module
    """
    def __init__(self, in_channels, reduction_ratio=0.5):
        super(CSA, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        # x: [B, C, H, W]
        
        # Channel Attention
        x_ch = self.ca(x)
        
        # Spatial Attention
        sa_map = self.sa(x_ch)
        
        # Refined feature = x_ch * sa_map
        refined = x_ch * sa_map
        
        # Residual
        return refined + x
