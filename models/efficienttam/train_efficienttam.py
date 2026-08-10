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
import numpy as np
from tqdm import tqdm
import yaml
import argparse

from utils import VideoSegmentationDataset, collate_fn

class SimpleSegmentationModel(nn.Module):
    """یک مدل ساده CNN برای segmentation - جایگزین EfficientTAM"""
    def __init__(self, num_classes=1):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_classes, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        
        features = self.encoder(x)
        out = self.decoder(features)
        
        # Resize به اندازه اصلی
        out = nn.functional.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        out = out.view(B, T, -1, H, W)
        
        return out

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # خواندن تنظیمات با تبدیل نوع داده
    data_path = config['data']['path']
    batch_size = int(config['training']['batch_size'])
    num_epochs = int(config['training']['num_epochs'])
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    num_frames = int(config['training']['num_frames'])
    image_size = int(config['training']['image_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Training Model on {device}")
    print(f"📁 Data path: {data_path}")
    print(f"📊 Batch size: {batch_size}, Epochs: {num_epochs}")
    print(f"📈 Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # مدل
    model = SimpleSegmentationModel(num_classes=int(config['model']['num_classes'])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    print(f"\n📊 Training for {num_epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            frames = batch['frames'].to(device)
            masks = batch['masks'].to(device)
            
            # masks: [B, T, H, W] -> [B, T, 1, H, W]
            masks = masks.unsqueeze(2)
            
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
        
        # ذخیره بهترین مدل
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = Path(config['logging']['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / 'model_best.pth')
            print(f"✅ Best model saved to {save_dir}/model_best.pth")
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/efficienttam_config.yaml')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--num_epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--num_frames', type=int, default=None, help='Override number of frames')
    parser.add_argument('--image_size', type=int, default=None, help='Override image size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Override learning rate')
    
    args = parser.parse_args()
    
    # اگر آرگومان‌های خط فرمان داده شده، config را override کن
    if args.batch_size or args.num_epochs or args.num_frames or args.image_size or args.learning_rate:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
        if args.num_epochs:
            config['training']['num_epochs'] = args.num_epochs
        if args.num_frames:
            config['training']['num_frames'] = args.num_frames
        if args.image_size:
            config['training']['image_size'] = args.image_size
        if args.learning_rate:
            config['training']['learning_rate'] = args.learning_rate
        
        # ذخیره config موقت
        temp_config = Path(args.config).parent / 'temp_config.yaml'
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        train(temp_config)
        temp_config.unlink()
    else:
        train(args.config)
