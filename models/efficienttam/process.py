"""
EfficientTAM - پردازش تصویر و ویدیو
تشخیص خودکار ورودی (تصویر یا ویدیو)
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.efficienttam.inference import EfficientTAMInference


class EfficientTAMProcessor:
    """کلاس پردازش تصویر و ویدیو با EfficientTAM"""
    
    def __init__(self):
        self.inference = EfficientTAMInference()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def is_video(self, path):
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        return Path(path).suffix.lower() in video_extensions
    
    def is_image(self, path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return Path(path).suffix.lower() in image_extensions
    
    def is_folder(self, path):
        return Path(path).is_dir()
    
    def process_image(self, image_path, output_dir="outputs/masks", show_result=False):
        """پردازش یک تصویر"""
        print(f"🖼️ Processing image: {image_path}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mask = self.inference.generate_mask(
            image_path, 
            output_dir / f"{Path(image_path).stem}_mask.png",
            show_result=show_result
        )
        
        result = {
            "type": "image",
            "input": str(image_path),
            "mask": str(output_dir / f"{Path(image_path).stem}_mask.png"),
            "overlay": str(output_dir / f"{Path(image_path).stem}_mask_overlay.png"),
            "comparison": str(output_dir / f"{Path(image_path).stem}_comparison.png")
        }
        
        return result
    
    def process_video(self, video_path, output_dir="outputs/video_masks", frame_interval=5, max_frames=50, show_result=False):
        """پردازش یک ویدیو"""
        print(f"🎬 Processing video: {video_path}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   FPS: {fps}, Total frames: {total_frames}")
        
        frame_paths = []
        count = 0
        temp_dir = output_dir / "temp_frames"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frame_path = temp_dir / f"frame_{count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(frame_path)
            count += 1
            if len(frame_paths) >= max_frames:
                break
        
        cap.release()
        print(f"   Extracted {len(frame_paths)} frames")
        
        results = []
        for frame_path in tqdm(frame_paths, desc="Processing frames"):
            mask_path = output_dir / f"{frame_path.stem}_mask.png"
            mask = self.inference.generate_mask(frame_path, mask_path, show_result=False)  # فقط برای اولین فریم نمایش داده میشه
            results.append({
                "frame": frame_path.name,
                "frame_number": int(frame_path.stem.split('_')[1]),
                "mask": str(mask_path)
            })
        
        if results and show_result:
            # نمایش اولین فریم با ماسک
            self.inference.generate_mask(frame_paths[0], output_dir / f"{frame_paths[0].stem}_mask.png", show_result=True)
        
        if results:
            video_output = self._create_video_with_masks(frame_paths, output_dir, fps)
        else:
            video_output = None
        
        return {
            "type": "video",
            "input": str(video_path),
            "total_frames": len(results),
            "fps": fps,
            "output_video": str(video_output) if video_output else None,
            "masks": results
        }
    
    def _create_video_with_masks(self, frame_paths, output_dir, fps):
        if not frame_paths:
            return None
        
        first_frame = cv2.imread(str(frame_paths[0]))
        height, width = first_frame.shape[:2]
        
        video_output = output_dir / "output_with_masks.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_output), min(fps, 10), (width, height))
        
        for frame_path in tqdm(frame_paths, desc="Creating video"):
            frame = cv2.imread(str(frame_path))
            mask_path = output_dir / f"{frame_path.stem}_mask.png"
            
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask_colored = np.zeros_like(frame)
                mask_colored[:, :, 2] = mask
                alpha = 0.4
                frame = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)
            
            out.write(frame)
        
        out.release()
        return video_output
    
    def process_folder(self, folder_path, output_dir="outputs/masks", show_result=False):
        print(f"📁 Processing folder: {folder_path}")
        folder_path = Path(folder_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")) + \
                 list(folder_path.glob("*.jpeg")) + list(folder_path.glob("*.bmp"))
        
        if not images:
            print("❌ No images found in folder")
            return
        
        print(f"   Found {len(images)} images")
        
        results = []
        for img_path in tqdm(images, desc="Processing images"):
            result = self.process_image(img_path, output_dir, show_result=False)
            results.append(result)
        
        if show_result and results:
            # نمایش اولین تصویر
            self.inference.generate_mask(images[0], output_dir / f"{images[0].stem}_mask.png", show_result=True)
        
        json_path = output_dir / "results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def process(self, input_path, output_dir="outputs", show_result=False):
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        
        print("="*60)
        print("🎨 EFFICIENTTAM - PROCESSING")
        print("="*60)
        print(f"📁 Input: {input_path}")
        print("="*60)
        
        if not input_path.exists():
            print(f"❌ Input not found: {input_path}")
            return None
        
        if self.is_image(input_path):
            result = self.process_image(input_path, output_dir / "masks", show_result)
            self._save_summary(result, output_dir / "masks" / "summary.json")
            return result
        
        elif self.is_video(input_path):
            result = self.process_video(input_path, output_dir / "video_masks", show_result=show_result)
            self._save_summary(result, output_dir / "video_masks" / "summary.json")
            return result
        
        elif self.is_folder(input_path):
            video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi"))
            if video_files:
                return self.process_video(video_files[0], output_dir / "video_masks", show_result=show_result)
            else:
                result = self.process_folder(input_path, output_dir / "masks", show_result)
                self._save_summary(result, output_dir / "masks" / "summary.json")
                return result
        
        else:
            print(f"❌ Unsupported file type: {input_path}")
            return None
    
    def _save_summary(self, result, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "result": result if isinstance(result, dict) else {"count": len(result) if result else 0}
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Summary saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='EfficientTAM - Image/Video Segmentation',
        epilog='Example: python -m models.efficienttam.process --input sample_input/00035.jpg'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to image, video, or folder')
    parser.add_argument('--output', '-o', type=str, default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--max_frames', type=int, default=50,
                       help='Max frames to process for video (default: 50)')
    parser.add_argument('--frame_interval', type=int, default=5,
                       help='Process every Nth frame (default: 5)')
    parser.add_argument('--show', '-s', action='store_true',
                       help='Show result image')
    
    args = parser.parse_args()
    
    processor = EfficientTAMProcessor()
    result = processor.process(args.input, args.output, show_result=args.show)
    
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE")
    print("="*60)
    print(f"📁 Output: {args.output}/")

if __name__ == "__main__":
    main()
