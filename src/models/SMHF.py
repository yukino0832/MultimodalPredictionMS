import torch
import torch.nn as nn
import torchvision.models as models
from .attention import SelfAttention, CSA

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=1, with_conv_shortcut=False):
        super(BottleneckBlock, self).__init__()
        f1, f2, f3 = filters
        
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(f3)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if with_conv_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, f3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(f3)
            )
        elif in_channels != f3:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, f3, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(f3)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class DownsamplingModel(nn.Module):
    """
    Input: (32, 32, 1024) -> Output: (16, 16, 2048)
    """
    def __init__(self, in_channels):
        super(DownsamplingModel, self).__init__()
        # filters=[512, 512, 2048]
        filters = [512, 512, 2048]
        
        self.block1 = BottleneckBlock(in_channels, filters, stride=2, with_conv_shortcut=True)
        self.block2 = BottleneckBlock(filters[2], filters, stride=1, with_conv_shortcut=False)
        self.block3 = BottleneckBlock(filters[2], filters, stride=1, with_conv_shortcut=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class MetaFusion(nn.Module):
    def __init__(self, img_dim, meta_dim):
        super(MetaFusion, self).__init__()
        self.meta_projector = nn.Linear(meta_dim, img_dim)
    
    def forward(self, img_feat, meta_feat):
        meta_gate = self.meta_projector(meta_feat)
        gate = torch.tanh(meta_gate)
        out = img_feat + (img_feat * gate)

        return out

class SMHF(nn.Module):
    def __init__(self, num_classes=4):
        super(SMHF, self).__init__()
        
        # Backbones
        resnet_mg = models.resnet50(pretrained=True)
        resnet_us = models.resnet50(pretrained=True)
        
        # Extract up to layer3 (conv4_x). Output channels: 1024.
        self.mg_stream = nn.Sequential(
            resnet_mg.conv1,
            resnet_mg.bn1,
            resnet_mg.relu,
            resnet_mg.maxpool,
            resnet_mg.layer1,
            resnet_mg.layer2,
            resnet_mg.layer3
        )
        
        self.us_stream = nn.Sequential(
            resnet_us.conv1,
            resnet_us.bn1,
            resnet_us.relu,
            resnet_us.maxpool,
            resnet_us.layer1,
            resnet_us.layer2,
            resnet_us.layer3
        )
        
        self.sa1 = SelfAttention(1024)
        
        # Downsampling Models
        self.ds_11 = DownsamplingModel(1024)
        self.ds_12 = DownsamplingModel(1024)
        
        # US Branch
        self.sa2 = SelfAttention(1024)
        self.ds_us = DownsamplingModel(1024)
        
        # Inter-modality Attention
        # Inputs: x11, x12, x2. Each (B, 2048, 16, 16).
        # PyTorch: concat dim 3 (W).
        self.sa3 = SelfAttention(2048)
        
        # CSA
        self.csa = CSA(in_channels=2048*3)
        
        # Classifier
        # Global Avg Pooling for each branch (x31, x32, x33).
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.cl = nn.Sequential(
            nn.Linear(8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.mf = MetaFusion(img_dim=2048*3, meta_dim=128)
        
        self.fc = nn.Sequential(
            nn.Linear(2048*3 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_mlo, x_cc, x_us, clinical):
        # Allow 1-channel input by repeating(for ResNet input)
        if x_mlo.size(1) == 1:
            x_mlo = x_mlo.repeat(1, 3, 1, 1)
        if x_cc.size(1) == 1:
            x_cc = x_cc.repeat(1, 3, 1, 1)
        if x_us.size(1) == 1:
            x_us = x_us.repeat(1, 3, 1, 1)

        # 1. Feature Extraction
        f_mlo = self.mg_stream(x_mlo) # [B, 1024, H/16, W/16]
        f_cc = self.mg_stream(x_cc)
        f_us = self.us_stream(x_us)
        
        # 2. MG Intra-modality Attention
        # Concat spatially along Width (dim 3)
        # [B, 1024, H, W] -> [B, 1024, H, 2W]
        c1 = torch.cat([f_mlo, f_cc], dim=3)
        
        a1 = self.sa1(c1)
        
        # Split
        w = f_mlo.size(3)
        f_mlo_att = a1[:, :, :, :w]
        f_cc_att = a1[:, :, :, w:]
        
        # Downsample
        x11 = self.ds_11(f_mlo_att) # [B, 2048, H/2, W/2]
        x12 = self.ds_12(f_cc_att)
        
        # 3. US Intra-modality Attention
        a2 = self.sa2(f_us)
        x2 = self.ds_us(a2) # [B, 2048, H/2, W/2]
        
        # 4. Inter-modality Attention
        # Concat spatially [x11, x12, x2] -> [B, 2048, H', 3W']
        ccc = torch.cat([x11, x12, x2], dim=3)
        
        a3 = self.sa3(ccc)
        
        # Split
        w2 = x11.size(3)
        x61 = a3[:, :, :, :w2]
        x62 = a3[:, :, :, w2:2*w2]
        x63 = a3[:, :, :, 2*w2:]
        
        # 5. Channel-Spatial Attention
        # Concat along Channel dimension [B, 6144, H', W']
        c6 = torch.cat([x61, x62, x63], dim=1)
        
        x3 = self.csa(c6)
        
        # x3: [B, 6144, H', W']
        x31 = x3[:, :2048, :, :]
        x32 = x3[:, 2048:4096, :, :]
        x33 = x3[:, 4096:, :, :]
        
        p1 = self.avgpool(x31).view(x31.size(0), -1)
        p2 = self.avgpool(x32).view(x32.size(0), -1)
        p3 = self.avgpool(x33).view(x33.size(0), -1)
        
        concat_img = torch.cat([p1, p2, p3], dim=1) # [B, 6144]
        
        # 6. Meta Fusion
        img_cl = self.mf(concat_img, self.cl(clinical))
        
        concat_final = torch.cat([img_cl, self.cl(clinical)], dim=1)
        
        # 7. Classification
        out = self.fc(concat_final)
        
        return out
