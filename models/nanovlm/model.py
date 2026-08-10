import torch
import torch.nn as nn
from transformers import SiglipVisionModel
import torch.nn.functional as F

class NanoVLM(nn.Module):
    def __init__(self, vision_encoder="google/siglip-base-patch16-224", hidden_size=512, num_classes=1):
        super().__init__()
        self.vision_encoder = SiglipVisionModel.from_pretrained(vision_encoder)
        self.projection = nn.Linear(768, hidden_size)
        self.projection_norm = nn.LayerNorm(hidden_size)
        self.seg_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )
        
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        self.image_size = 224
        print(f"✅ NanoVLM initialized with SiglipVisionModel")
        print(f"✅ Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, frames):
        B, T, C, H, W = frames.shape
        
        # Resize به 224x224
        frames_resized = F.interpolate(
            frames.view(B*T, C, H, W), 
            size=(self.image_size, self.image_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        with torch.no_grad():
            vision_out = self.vision_encoder(pixel_values=frames_resized)
            features = vision_out.last_hidden_state.mean(dim=1)  # [B*T, 768]
        
        features = self.projection(features)
        features = self.projection_norm(features)
        seg = self.seg_head(features)  # [B*T, num_classes]
        
        # تبدیل به [B, T, num_classes, 224, 224]
        seg = seg.view(B, T, -1, 1, 1)  # [B, T, num_classes, 1, 1]
        
        # Upsample به اندازه اصلی
        seg = F.interpolate(
            seg.view(B*T, -1, 1, 1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        seg = seg.view(B, T, -1, H, W)
        
        return seg
