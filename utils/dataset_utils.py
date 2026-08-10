import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

class VideoSegmentationDataset(Dataset):
    """دیتاست سفارشی برای Video Segmentation"""
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 transform=None,
                 num_frames: int = 8,
                 frame_size: Tuple[int, int] = (224, 224)):
        
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        self.video_folders = self._load_video_folders()
        print(f"Loaded {len(self.video_folders)} videos for {split} split")
        
    def _load_video_folders(self):
        video_folders = []
        
        # ساختار: data_path/split/frames/video_id/ و data_path/split/masks/video_id/
        frames_dir = self.data_path / self.split / "frames"
        masks_dir = self.data_path / self.split / "masks"
        
        if frames_dir.exists() and masks_dir.exists():
            for video_dir in frames_dir.iterdir():
                if video_dir.is_dir():
                    video_id = video_dir.name
                    mask_dir = masks_dir / video_id
                    
                    if mask_dir.exists():
                        frames = sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.jpeg")))
                        masks = sorted(list(mask_dir.glob("*.png")))
                        
                        if frames and masks:
                            # اطمینان از تعداد مساوی
                            min_count = min(len(frames), len(masks))
                            video_folders.append({
                                'video_path': video_dir,
                                'frames': frames[:min_count],
                                'masks': masks[:min_count]
                            })
        
        # اگر ساختار بالا کار نکرد، از valid/JPEGImages و valid/Annotations استفاده کن
        if not video_folders:
            valid_images = self.data_path / "valid" / "JPEGImages"
            valid_annos = self.data_path / "valid" / "Annotations"
            
            if valid_images.exists() and valid_annos.exists():
                for video_dir in valid_images.iterdir():
                    if video_dir.is_dir():
                        video_id = video_dir.name
                        anno_dir = valid_annos / video_id
                        
                        if anno_dir.exists():
                            frames = sorted(list(video_dir.glob("*.jpg")))
                            masks = sorted(list(anno_dir.glob("*.png")))
                            
                            if frames and masks:
                                min_count = min(len(frames), len(masks))
                                video_folders.append({
                                    'video_path': video_dir,
                                    'frames': frames[:min_count],
                                    'masks': masks[:min_count]
                                })
        
        return video_folders
    
    def __len__(self):
        return len(self.video_folders)
    
    def __getitem__(self, idx):
        video_data = self.video_folders[idx]
        
        total_frames = len(video_data['frames'])
        
        if total_frames == 0:
            return {
                'frames': torch.zeros(self.num_frames, 3, self.frame_size[0], self.frame_size[1]),
                'masks': torch.zeros(self.num_frames, self.frame_size[0], self.frame_size[1]),
                'video_name': 'empty'
            }
        
        # انتخاب فریم‌های تصادفی
        if total_frames >= self.num_frames:
            indices = sorted(random.sample(range(total_frames), self.num_frames))
        else:
            indices = sorted(random.choices(range(total_frames), k=self.num_frames))
        
        frames = []
        masks = []
        
        for i in indices:
            frame_path = video_data['frames'][i]
            mask_path = video_data['masks'][i] if i < len(video_data['masks']) else None
            
            try:
                # بارگذاری فریم
                frame = Image.open(frame_path).convert('RGB')
                frame = frame.resize(self.frame_size)
                frame = torch.tensor(np.array(frame) / 255.0, dtype=torch.float32).permute(2, 0, 1)
                
                # بارگذاری ماسک
                if mask_path and mask_path.exists():
                    mask = Image.open(mask_path).convert('L')
                    mask = mask.resize(self.frame_size)
                    mask_np = np.array(mask, dtype=np.float32)
                    
                    # مهم: تبدیل به 0 و 1 (نه 0 و 255)
                    # اگر ماسک بین 0 تا 255 است، به 0 و 1 تبدیل کن
                    if mask_np.max() > 1:
                        mask_np = mask_np / 255.0
                    
                    # باینری کردن با آستانه 0.5
                    mask_np = (mask_np > 0.5).astype(np.float32)
                    mask = torch.tensor(mask_np, dtype=torch.float32)
                else:
                    mask = torch.zeros(self.frame_size[0], self.frame_size[1], dtype=torch.float32)
                
                frames.append(frame)
                masks.append(mask)
                
            except Exception as e:
                print(f"Error loading frame {frame_path}: {e}")
                frames.append(torch.zeros(3, self.frame_size[0], self.frame_size[1]))
                masks.append(torch.zeros(self.frame_size[0], self.frame_size[1]))
        
        # اطمینان از تعداد فریم‌ها
        while len(frames) < self.num_frames:
            frames.append(torch.zeros(3, self.frame_size[0], self.frame_size[1]))
            masks.append(torch.zeros(self.frame_size[0], self.frame_size[1]))
        
        frames = torch.stack(frames[:self.num_frames])
        masks = torch.stack(masks[:self.num_frames])
        
        return {
            'frames': frames,
            'masks': masks,
            'video_name': str(video_data['video_path'].name)
        }

def collate_fn(batch):
    frames = torch.stack([item['frames'] for item in batch])
    masks = torch.stack([item['masks'] for item in batch])
    video_names = [item['video_name'] for item in batch]
    
    return {
        'frames': frames,
        'masks': masks,
        'video_names': video_names
    }
