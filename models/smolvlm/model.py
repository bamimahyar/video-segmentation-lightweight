import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
import torch.nn.functional as F

class SmolVLM(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", num_classes=1):
        super().__init__()
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        
        self.seg_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.image_size = 224
        print(f"✅ SmolVLM initialized with ViT")
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
            outputs = self.model(pixel_values=frames_resized)
            features = outputs.last_hidden_state.mean(dim=1)  # [B*T, 768]
        
        features = features.view(B, T, -1)
        seg = self.seg_head(features)  # [B, T, num_classes]
        
        # تبدیل به [B, T, num_classes, 1, 1] و سپس upsample
        seg = seg.view(B, T, -1, 1, 1)
        seg = F.interpolate(
            seg.view(B*T, -1, 1, 1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        seg = seg.view(B, T, -1, H, W)
        
        return seg
