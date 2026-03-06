import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MetaFusion(nn.Module):
    def __init__(self, img_dim, meta_dim):
        super(MetaFusion, self).__init__()
        self.meta_projector = nn.Linear(meta_dim, img_dim)
    
    def forward(self, img_feat, meta_feat):
        meta_gate = self.meta_projector(meta_feat)
        gate = torch.tanh(meta_gate)
        out = img_feat + (img_feat * gate)
        return out


class ViewAwareAttention(nn.Module):
    def __init__(self, dim):
        super(ViewAwareAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        out = torch.sum(x * attn_weights, dim=1)  # (B, C)
        return out, attn_weights

class _TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CAMILFCrossModalFusion(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.mg_cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.us_cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mg_cls_token, std=1e-6)
        nn.init.normal_(self.us_cls_token, std=1e-6)
        self.mg_encoder = _TransformerBlock(dim, num_heads, dropout)
        self.us_encoder = _TransformerBlock(dim, num_heads, dropout)

        self.norm_mg_cross = nn.LayerNorm(dim)
        self.norm_us_cross = nn.LayerNorm(dim)
        self.cross_attn_mg_to_us = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.cross_attn_us_to_mg = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.mg_decoder = _TransformerBlock(dim, num_heads, dropout)
        self.us_decoder = _TransformerBlock(dim, num_heads, dropout)

        self.norm_mg_dec = nn.LayerNorm(dim)
        self.norm_us_dec = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, mg_patches, us_patches):
        B = mg_patches.size(0)

        mg_cls = self.mg_cls_token.expand(B, -1, -1)
        us_cls = self.us_cls_token.expand(B, -1, -1)

        mg_seq = torch.cat([mg_cls, mg_patches], dim=1)
        us_seq = torch.cat([us_cls, us_patches], dim=1)

        mg_enc = self.mg_encoder(mg_seq)
        us_enc = self.us_encoder(us_seq)

        mg_patch_enc = mg_enc[:, 1:, :]
        us_patch_enc = us_enc[:, 1:, :]

        mg_q  = self.norm_mg_cross(mg_patch_enc)
        us_kv = self.norm_us_cross(us_patch_enc)
        mg_cross, _ = self.cross_attn_mg_to_us(mg_q, us_kv, us_kv)
        mg_cross = mg_patch_enc + mg_cross

        us_q  = self.norm_us_cross(us_patch_enc)
        mg_kv = self.norm_mg_cross(mg_patch_enc)
        us_cross, _ = self.cross_attn_us_to_mg(us_q, mg_kv, mg_kv)
        us_cross = us_patch_enc + us_cross

        mg_dec_seq = torch.cat([mg_cls, mg_cross], dim=1)
        us_dec_seq = torch.cat([us_cls, us_cross], dim=1)

        mg_dec = self.mg_decoder(mg_dec_seq)
        us_dec = self.us_decoder(us_dec_seq)

        mg_cls_dec = self.norm_mg_dec(mg_dec[:, 0, :])
        us_cls_dec = self.norm_us_dec(us_dec[:, 0, :])

        fused_feat = self.ffn(torch.cat([mg_cls_dec, us_cls_dec], dim=1))
        return fused_feat


class MSHF(nn.Module):
    def __init__(self, backbone='ResNet50', num_classes=2):
        super(MSHF, self).__init__()
        self.backbone_name = backbone
        path_to_weights = "pretrained/RadImageNet_pytorch/" + backbone + ".pt"

        self.mg_backbone, feature_dim = self.get_backbone(backbone, path_to_weights)
        self.us_backbone, _           = self.get_backbone(backbone, path_to_weights)

        self.reduce_mlo = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.reduce_cc  = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.reduce_us  = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.2))

        self.clinical_net = nn.Sequential(
            nn.Linear(8, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.view_attention = ViewAwareAttention(256)

        self.cross_modal_fusion = CAMILFCrossModalFusion(dim=256, num_heads=4, dropout=0.1)

        self.meta_fusion = MetaFusion(256, 32)
        
        self.aux_classifier_mg = nn.Linear(256, num_classes)
        self.aux_classifier_us = nn.Linear(256, num_classes)

        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(128, num_classes)
        )

        self.init_head_weights()

    def get_backbone(self, name, weight_path):
        model = None
        feat_dim = 0
        
        if name == 'ResNet50':
            model = models.resnet50(pretrained=False)
            if weight_path:
                self.load_weights(model, weight_path)
            backbone = nn.Sequential(*list(model.children())[:-2])
            feat_dim = 2048
            
        elif name == 'DenseNet121':
            model = models.densenet121(pretrained=False)
            if weight_path:
                self.load_weights(model, weight_path)
            backbone = model.features
            feat_dim = 1024
            
        elif name == 'InceptionV3':
            model = models.inception_v3(pretrained=False, aux_logits=False)
            if weight_path:
                self.load_weights(model, weight_path)
            backbone = nn.Sequential(
                model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3,
                model.maxpool1, model.Conv2d_3b_1x1, model.Conv2d_4a_3x3,
                model.maxpool2, model.Mixed_5b, model.Mixed_5c, model.Mixed_5d,
                model.Mixed_6a, model.Mixed_6b, model.Mixed_6c, model.Mixed_6d,
                model.Mixed_6e, 
                model.Mixed_7a, model.Mixed_7b, model.Mixed_7c
            )
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        
        return backbone, feat_dim

    def load_weights(self, model, path):
        state_dict = torch.load(path, map_location='cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") 
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)

    def init_head_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_one_stream(self, x, mask, backbone, reducer):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        features = backbone(x)   # (B, C, H, W)
        
        if mask is not None:
            if mask.size(1) != 1: mask = mask.unsqueeze(1)
            
            if mask.max() > 1.0:
                mask = mask / 255.0

            mask_down = F.interpolate(mask, size=features.shape[2:], mode='nearest')
            
            features = features * (1.0 + mask_down)
        
        B, C, H, W = features.shape

        vec_raw = F.adaptive_max_pool2d(features, (1, 1)).view(B, -1)
        vec = reducer(vec_raw)

        patch_raw = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        patches   = reducer(patch_raw)

        return vec, patches

    def forward(self, img_mlo, img_cc, img_us, clinical, mask_mlo, mask_cc, mask_us):
        vec_mlo, mlo_patches = self.forward_one_stream(img_mlo, mask_mlo, self.mg_backbone, self.reduce_mlo)
        vec_cc,  cc_patches  = self.forward_one_stream(img_cc,  mask_cc,  self.mg_backbone, self.reduce_cc)
        vec_us,  us_patches  = self.forward_one_stream(img_us,  mask_us,  self.us_backbone, self.reduce_us)

        mg_patches = torch.cat([mlo_patches, cc_patches], dim=1)

        stacked_mg_feats = torch.stack([vec_mlo, vec_cc], dim=1)
        vec_mg, mg_attn_weights = self.view_attention(stacked_mg_feats)
        
        fused_img_feat = self.cross_modal_fusion(mg_patches, us_patches)
        clinical_feat = self.clinical_net(clinical)
        meta_feat = self.meta_fusion(fused_img_feat, clinical_feat)
        
        out_main = self.classifier(torch.cat([meta_feat, clinical_feat], dim=1))
        
        if self.training:
            out_mg = self.aux_classifier_mg(vec_mg)
            out_us = self.aux_classifier_us(vec_us)
            return out_main, out_mg, out_us
            
        return out_main
