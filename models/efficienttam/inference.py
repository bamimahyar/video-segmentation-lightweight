import torch
import sys
from pathlib import Path
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import hashlib

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.efficienttam.model import UNetLike

class EfficientTAMInference:
    """Class for generating masks with EfficientTAM"""
    
    def __init__(self, model_path="checkpoints/unet_model/model_best.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNetLike()
        
        model_path = Path(model_path)
        if not model_path.exists():
            alt_paths = [
                "checkpoints/unet_model/model_best.pth",
                "checkpoints/efficienttam/model_best.pth",
                "checkpoints/unet_20epochs/model_best.pth"
            ]
            for alt in alt_paths:
                if Path(alt).exists():
                    model_path = Path(alt)
                    break
        
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ EfficientTAM loaded from {model_path}")
        else:
            print(f"⚠️ Model not found at {model_path}")
            return
        
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ EfficientTAM loaded for mask generation")
    
    def _get_unique_name(self, image_path):
        """تولید نام یکتا برای هر تصویر بر اساس مسیر کامل"""
        path_str = str(Path(image_path).resolve())
        # استفاده از هش برای ایجاد نام یکتا
        hash_obj = hashlib.md5(path_str.encode())
        hash_id = hash_obj.hexdigest()[:8]
        
        # ترکیب نام پوشه و نام فایل
        parent_name = Path(image_path).parent.name
        file_name = Path(image_path).stem
        
        # اگر parent_name معتبر نیست (مثل sample_input)، فقط از file_name استفاده کن
        if parent_name in ['JPEGImages', 'sample_input', '']:
            return file_name
        else:
            return f"{parent_name}_{file_name}"
    
    def generate_comparison(self, image_path, output_dir="outputs/masks", show_result=False):
        """Generate and save ONLY comparison image (Original + Mask + Overlay)"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # بارگذاری تصویر
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # تولید نام یکتا
        unique_name = self._get_unique_name(image_path)
        
        # Resize به 224x224
        image_resized = image.resize((224, 224))
        image_np = np.array(image_resized, dtype=np.float32) / 255.0
        image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mask = self.model(image_tensor)
            mask = mask.squeeze().cpu().numpy()
            
            if mask.ndim == 3:
                mask = mask[0]
            elif mask.ndim == 4:
                mask = mask[0, 0]
            
            mask = (mask > 0.5).astype(np.uint8) * 255
        
        mask_pil = Image.fromarray(mask, mode='L')
        mask_pil = mask_pil.resize(original_size, Image.Resampling.NEAREST)
        
        # ایجاد Overlay
        image_np = np.array(image.resize(original_size))
        mask_np = np.array(mask_pil)
        
        overlay = image_np.copy()
        mask_bool = mask_np > 0
        overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0] + 150, 0, 255)
        overlay[mask_bool, 1] = np.clip(overlay[mask_bool, 1] - 50, 0, 255)
        overlay[mask_bool, 2] = np.clip(overlay[mask_bool, 2] - 50, 0, 255)
        
        overlay_pil = Image.fromarray(overlay)
        
        # ذخیره با نام یکتا
        comparison_path = output_dir / f"{unique_name}_comparison.png"
        self._save_comparison(image, mask_pil, overlay_pil, comparison_path)
        print(f"✅ Comparison saved to {comparison_path}")
        
        if show_result:
            self._show_comparison(image, mask_pil, overlay_pil)
        
        return {
            "comparison": str(comparison_path),
            "unique_name": unique_name,
            "original_path": str(image_path)
        }
    
    def _save_comparison(self, original_image, mask_image, overlay_image, save_path):
        """ذخیره تصویر مقایسه (Original + Mask + Overlay) با کیفیت بالا"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # تنظیم فاصله بین تصاویر
        plt.subplots_adjust(wspace=0.05)
        
        # تصویر اصلی
        axes[0].imshow(np.array(original_image.resize(mask_image.size)))
        axes[0].set_title('Original Image', fontsize=14, pad=10)
        axes[0].axis('off')
        
        # ماسک
        axes[1].imshow(np.array(mask_image), cmap='gray')
        axes[1].set_title('Generated Mask', fontsize=14, pad=10)
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(np.array(overlay_image))
        axes[2].set_title('Overlay Result', fontsize=14, pad=10)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   📸 Comparison saved: {save_path}")
    
    def _show_comparison(self, original_image, mask_image, overlay_image):
        """نمایش تصویر مقایسه"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        plt.subplots_adjust(wspace=0.05)
        
        axes[0].imshow(np.array(original_image.resize(mask_image.size)))
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(np.array(mask_image), cmap='gray')
        axes[1].set_title('Generated Mask', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(np.array(overlay_image))
        axes[2].set_title('Overlay Result', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def process_video(self, video_path, output_dir="outputs/masks", show_result=False):
        """پردازش تصویر یا پوشه و ذخیره فقط comparison"""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        if video_path.is_dir():
            images = list(video_path.glob("*.jpg")) + list(video_path.glob("*.png"))
            print(f"📁 Processing {len(images)} images from folder...")
            print("-"*50)
            
            for idx, img in enumerate(images, 1):
                print(f"[{idx}/{len(images)}] {img.name}")
                result = self.generate_comparison(img, output_dir, show_result=False)
                results.append({
                    "frame": img.name,
                    "comparison": result["comparison"],
                    "unique_name": result["unique_name"]
                })
        else:
            print(f"📸 Processing single image: {video_path.name}")
            result = self.generate_comparison(video_path, output_dir, show_result)
            results = [{
                "frame": video_path.name,
                "comparison": result["comparison"],
                "unique_name": result["unique_name"]
            }]
        
        # ذخیره نتایج
        json_path = output_dir / "results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {json_path}")
        print(f"✅ Total comparisons generated: {len(results)}")
        
        return results

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='EfficientTAM - Generate comparison images (Original + Mask + Overlay)'
    )
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to image or folder containing images')
    parser.add_argument('--output', '-o', type=str, default='outputs/masks',
                       help='Output directory (default: outputs/masks)')
    parser.add_argument('--show', '-s', action='store_true',
                       help='Show result image')
    
    args = parser.parse_args()
    
    inference = EfficientTAMInference()
    results = inference.process_video(args.image, args.output, args.show)
    print(f"\n🎉 Done! Generated {len(results)} comparison images")

if __name__ == "__main__":
    main()
