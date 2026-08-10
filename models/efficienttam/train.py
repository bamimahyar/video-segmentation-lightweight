import sys
from pathlib import Path
import os

# اضافه کردن مسیر پروژه
project_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_path))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm

from utils import VideoSegmentationDataset, collate_fn
from models.efficienttam.model import SimpleSegmentationModel

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config['data']['path']
    batch_size = int(config['training']['batch_size'])
    num_epochs = int(config['training']['num_epochs'])
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    num_frames = int(config['training']['num_frames'])
    image_size = int(config['training']['image_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Training EfficientTAM on {device}")
    print(f"📁 Data path: {data_path}")
    
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
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = SimpleSegmentationModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
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
            torch.save(model.state_dict(), save_dir / 'model_best.pth')
            print(f"✅ Best model saved to {save_dir}/model_best.pth")
    
    print("✅ Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/efficienttam_config.yaml')
    args = parser.parse_args()
    train(args.config)
