import sys
from pathlib import Path
import os

# اضافه کردن مسیر پروژه به sys.path
project_path = Path(__file__).parent.parent
sys.path.insert(0, str(project_path))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

from utils import VideoSegmentationDataset, collate_fn

class NanoVLM(nn.Module):
    """مدل nanoVLM برای video segmentation"""
    def __init__(self, vision_encoder="google/siglip-base-patch16-224", 
                 hidden_size=512, num_classes=1):
        super().__init__()
        
        print(f"📦 Loading vision encoder: {vision_encoder}")
        self.vision_encoder = AutoModel.from_pretrained(vision_encoder)
        
        # پروجکشن لایه
        self.projection = nn.Linear(768, hidden_size)
        self.projection_norm = nn.LayerNorm(hidden_size)
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        print(f"✅ NanoVLM initialized with {self._count_parameters():,} trainable parameters")
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, frames):
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)
        
        with torch.no_grad():
            vision_out = self.vision_encoder(pixel_values=frames_flat)
            features = vision_out.last_hidden_state.mean(dim=1)
        
        features = self.projection(features)
        features = self.projection_norm(features)
        seg = self.seg_head(features)
        seg = seg.view(B, T, self.num_classes, H, W)
        
        return seg

def train(config_path, override_args=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # اگر آرگومان‌های override داده شده، آنها را اعمال کن
    if override_args:
        for key, value in override_args.items():
            if key in config['training']:
                config['training'][key] = value
    
    # خواندن تنظیمات
    data_path = config['data']['path']
    batch_size = int(config['training']['batch_size'])
    num_epochs = int(config['training']['num_epochs'])
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    num_frames = int(config['training']['num_frames'])
    image_size = int(config['training']['image_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Training nanoVLM on {device}")
    print(f"📁 Data path: {data_path}")
    print(f"📊 Batch size: {batch_size}, Epochs: {num_epochs}")
    print(f"📈 Learning rate: {learning_rate}")
    print(f"📐 Image size: {image_size}, Frames: {num_frames}")
    
    # بارگذاری دیتاست
    dataset = VideoSegmentationDataset(
        data_path=data_path,
        split='train',
        num_frames=num_frames,
        frame_size=(image_size, image_size)
    )
    
    if len(dataset) == 0:
        print("❌ No data found!")
        return
    
    print(f"✅ Loaded {len(dataset)} videos")
    
    # تقسیم به train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # مدل
    model = NanoVLM(
        vision_encoder=config['model']['vision_encoder'],
        hidden_size=int(config['model']['hidden_size']),
        num_classes=int(config['model']['num_classes'])
    ).to(device)
    
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    criterion = nn.BCELoss()
    
    print(f"\n📊 Training for {num_epochs} epochs...")
    best_val_loss = float('inf')
    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            frames = batch['frames'].to(device)
            masks = batch['masks'].to(device).unsqueeze(2)
            
            optimizer.zero_grad()
            preds = model(frames)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(device)
                masks = batch['masks'].to(device).unsqueeze(2)
                preds = model(frames)
                loss = criterion(preds, masks)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_dir / 'nanovlm_best.pth')
            print(f"✅ Best model saved to {save_dir}/nanovlm_best.pth")
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/nanovlm_config.yaml')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--num_epochs', type=int, help='Override number of epochs')
    parser.add_argument('--num_frames', type=int, help='Override number of frames')
    parser.add_argument('--image_size', type=int, help='Override image size')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    
    args = parser.parse_args()
    
    # جمع‌آوری آرگومان‌های override
    override_args = {}
    if args.batch_size:
        override_args['batch_size'] = args.batch_size
    if args.num_epochs:
        override_args['num_epochs'] = args.num_epochs
    if args.num_frames:
        override_args['num_frames'] = args.num_frames
    if args.image_size:
        override_args['image_size'] = args.image_size
    if args.learning_rate:
        override_args['learning_rate'] = args.learning_rate
    
    train(args.config, override_args if override_args else None)
