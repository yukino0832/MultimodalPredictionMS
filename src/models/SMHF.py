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
    Input: (H, W, C_in) -> Output: (H/2, W/2, C_out) C_out = C_in * 2
    """
    def __init__(self, in_channels):
        super(DownsamplingModel, self).__init__()
        
        f3 = in_channels * 2
        f1 = f3 // 4
        f2 = f3 // 4
        
        filters = [f1, f2, f3]
        
        # stride=2 in block1 does downsampling
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
    def __init__(self, num_classes=2, backbone='resnet18'):
        super(SMHF, self).__init__()
        
        self.backbone_name = backbone
        
        def get_resnet_layers(resnet_model):
            layers = [
                resnet_model.conv1,
                resnet_model.bn1,
                resnet_model.relu,
                resnet_model.maxpool,
                resnet_model.layer1,
                resnet_model.layer2,
                resnet_model.layer3
            ]
            return nn.Sequential(*layers)

        def get_densenet_layers(densenet_model):
            layers = []
            features = densenet_model.features
            for name, module in features.named_children():
                if name == 'transition3':
                    break
                layers.append(module)
            return nn.Sequential(*layers)

        def get_inception_layers(inception_model):
            layers = [
                inception_model.Conv2d_1a_3x3,
                inception_model.Conv2d_2a_3x3,
                inception_model.Conv2d_2b_3x3,
                inception_model.maxpool1,
                inception_model.Conv2d_3b_1x1,
                inception_model.Conv2d_4a_3x3,
                inception_model.maxpool2,
                inception_model.Mixed_5b,
                inception_model.Mixed_5c,
                inception_model.Mixed_5d,
                inception_model.Mixed_6a,
                inception_model.Mixed_6b,
                inception_model.Mixed_6c,
                inception_model.Mixed_6d,
                inception_model.Mixed_6e
            ]
            return nn.Sequential(*layers)

        # Initialize backbones
        if 'resnet' in backbone:
            if backbone == 'resnet18':
                resnet_mg = models.resnet18(pretrained=True)
                resnet_us = models.resnet18(pretrained=True)
            elif backbone == 'resnet34':
                resnet_mg = models.resnet34(pretrained=True)
                resnet_us = models.resnet34(pretrained=True)
            elif backbone == 'resnet50':
                resnet_mg = models.resnet50(pretrained=True)
                resnet_us = models.resnet50(pretrained=True)
            else:
                raise ValueError(f"Unsupported resnet backbone: {backbone}")
            
            self.mg_stream = get_resnet_layers(resnet_mg)
            self.us_stream = get_resnet_layers(resnet_us)
            
        elif backbone == 'densenet121':
            densenet_mg = models.densenet121(pretrained=True)
            densenet_us = models.densenet121(pretrained=True)
            self.mg_stream = get_densenet_layers(densenet_mg)
            self.us_stream = get_densenet_layers(densenet_us)
            
        elif backbone == 'inceptionv3':
            inception_mg = models.inception_v3(pretrained=True)
            inception_us = models.inception_v3(pretrained=True)
            self.mg_stream = get_inception_layers(inception_mg)
            self.us_stream = get_inception_layers(inception_us)
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Determine backbone output channels dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 512, 512)
            dummy_out = self.mg_stream(dummy_input)
            self.backbone_channels = dummy_out.shape[1]
            print(f"Backbone: {backbone}, Output Channels: {self.backbone_channels}")
            
        C = self.backbone_channels
        
        self.sa1 = SelfAttention(C)
        
        # Downsampling Models
        self.ds_11 = DownsamplingModel(C)
        self.ds_12 = DownsamplingModel(C)
        
        # US Branch
        self.sa2 = SelfAttention(C)
        self.ds_us = DownsamplingModel(C)
        
        C_down = C * 2
        self.C_down = C_down
        
        self.sa3 = SelfAttention(C_down)
        
        # CSA
        self.csa = CSA(in_channels=3 * C_down)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.cl = nn.Sequential(
            nn.Linear(8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.mf = MetaFusion(img_dim=3 * C_down, meta_dim=128)
        
        self.fc = nn.Sequential(
            nn.Linear(3 * C_down + 128, 512),
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
        f_mlo = self.mg_stream(x_mlo)
        f_cc = self.mg_stream(x_cc)
        f_us = self.us_stream(x_us)
        
        # 2. MG Intra-modality Attention
        # Concat spatially along Width (dim 3)
        c1 = torch.cat([f_mlo, f_cc], dim=3)
        
        a1 = self.sa1(c1)
        
        # Split
        w = f_mlo.size(3)
        f_mlo_att = a1[:, :, :, :w]
        f_cc_att = a1[:, :, :, w:]
        
        # Downsample
        x11 = self.ds_11(f_mlo_att)
        x12 = self.ds_12(f_cc_att)
        
        # 3. US Intra-modality Attention
        a2 = self.sa2(f_us)
        x2 = self.ds_us(a2)
        
        # 4. Inter-modality Attention
        # Concat spatially
        ccc = torch.cat([x11, x12, x2], dim=3)
        
        a3 = self.sa3(ccc)
        
        # Split
        w2 = x11.size(3)
        x61 = a3[:, :, :, :w2]
        x62 = a3[:, :, :, w2:2*w2]
        x63 = a3[:, :, :, 2*w2:]
        
        # 5. Channel-Spatial Attention
        c6 = torch.cat([x61, x62, x63], dim=1)
        
        x3 = self.csa(c6)
        
        C_d = self.C_down
        x31 = x3[:, :C_d, :, :]
        x32 = x3[:, C_d:2*C_d, :, :]
        x33 = x3[:, 2*C_d:, :, :]
        
        p1 = self.avgpool(x31).view(x31.size(0), -1)
        p2 = self.avgpool(x32).view(x32.size(0), -1)
        p3 = self.avgpool(x33).view(x33.size(0), -1)
        
        concat_img = torch.cat([p1, p2, p3], dim=1)
        
        # 6. Meta Fusion
        img_cl = self.mf(concat_img, self.cl(clinical))
        
        concat_final = torch.cat([img_cl, self.cl(clinical)], dim=1)
        
        # 7. Classification
        out = self.fc(concat_final)
        
        return out
