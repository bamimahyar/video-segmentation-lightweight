#!/usr/bin/env python3
"""
پردازش تصویر/ویدیو با هر ۳ مدل
هر مدل خروجی مخصوص خودش را تولید می‌کند:
- SmolVLM: خروجی متنی (description)
- nanoVLM: خروجی متنی (description)
- EfficientTAM: خروجی ماسک (segmentation mask)

نحوه استفاده:
    python process_all_models.py --image sample_input/00035.jpg
    python process_all_models.py --image /path/to/your/image.jpg
    python process_all_models.py --image sample_input/ --model smolvlm
"""

import sys
from pathlib import Path
import argparse
import json
import time
import os

# اضافه کردن مسیر پروژه
sys.path.append(str(Path(__file__).parent))

def find_sample_image():
    """پیدا کردن یک تصویر نمونه از دیتاست"""
    possible_paths = [
        Path("sample_input"),
        Path("valid/JPEGImages"),
        Path("train/frames"),
        Path("val/frames"),
        Path("test/frames"),
    ]
    
    for path in possible_paths:
        if path.exists():
            if path.is_dir():
                # جستجو در پوشه
                for video_dir in path.iterdir():
                    if video_dir.is_dir():
                        images = list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))
                        if images:
                            return images[0]
                # اگر ویدیو داخل پوشه نبود، خود پوشه را بررسی کن
                images = list(path.glob("*.jpg")) + list(path.glob("*.png"))
                if images:
                    return images[0]
    return None

def process_with_smolvlm(image_path, output_dir):
    """پردازش با SmolVLM - خروجی متنی"""
    print("\n🔮 Processing with SmolVLM (Text Generation)...")
    from models.smolvlm.inference import SmolVLMInference
    
    output_file = Path(output_dir) / "text" / "smolvlm_output.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    inference = SmolVLMInference()
    results = inference.process_video(image_path, output_file)
    print(f"✅ SmolVLM: {len(results)} descriptions generated")
    return results

def process_with_nanovlm(image_path, output_dir):
    """پردازش با nanoVLM - خروجی متنی"""
    print("\n🔮 Processing with nanoVLM (Text Generation)...")
    from models.nanovlm.inference import NanoVLMInference
    
    output_file = Path(output_dir) / "text" / "nanovlm_output.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    inference = NanoVLMInference()
    results = inference.process_video(image_path, output_file)
    print(f"✅ nanoVLM: {len(results)} descriptions generated")
    return results

def process_with_efficienttam(image_path, output_dir):
    """پردازش با EfficientTAM - خروجی ماسک"""
    print("\n🎨 Processing with EfficientTAM (Mask Generation)...")
    from models.efficienttam.inference import EfficientTAMInference
    
    output_path = Path(output_dir) / "masks"
    output_path.mkdir(parents=True, exist_ok=True)
    
    inference = EfficientTAMInference()
    results = inference.process_video(image_path, output_path)
    print(f"✅ EfficientTAM: {len(results)} masks generated")
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Process images/videos with all 3 models',
        epilog='Example: python process_all_models.py --image sample_input/00035.jpg'
    )
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image or folder containing images')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--model', type=str, choices=['all', 'smolvlm', 'nanovlm', 'efficienttam'],
                       default='all', help='Which model to run (default: all)')
    
    args = parser.parse_args()
    
    # تعیین مسیر تصویر
    image_path = args.image
    
    # اگر آدرس داده نشده، از نمونه استفاده کن
    if image_path is None:
        print("🔍 No image path provided. Looking for sample image...")
        sample = find_sample_image()
        if sample:
            image_path = str(sample)
            print(f"📸 Using sample image: {image_path}")
        else:
            print("❌ No sample image found!")
            print("\nPlease provide an image path:")
            print("  python process_all_models.py --image /path/to/your/image.jpg")
            print("  python process_all_models.py --image sample_input/00035.jpg")
            return
    
    # بررسی وجود فایل
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Path not found: {image_path}")
        print("\nAvailable sample images:")
        sample = find_sample_image()
        if sample:
            print(f"  - {sample}")
        print("\nOr provide your own image:")
        print("  python process_all_models.py --image /path/to/your/image.jpg")
        return
    
    print("="*60)
    print("🎬 VIDEO SEGMENTATION - MULTI-MODEL PROCESSING")
    print("="*60)
    print(f"📁 Input: {image_path}")
    print(f"📁 Output: {args.output}")
    print(f"🤖 Model: {args.model}")
    print("="*60)
    
    start_time = time.time()
    results = {}
    
    if args.model in ['all', 'smolvlm']:
        results['smolvlm'] = process_with_smolvlm(image_path, args.output)
    
    if args.model in ['all', 'nanovlm']:
        results['nanovlm'] = process_with_nanovlm(image_path, args.output)
    
    if args.model in ['all', 'efficienttam']:
        results['efficienttam'] = process_with_efficienttam(image_path, args.output)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE")
    print("="*60)
    print(f"⏱️  Total time: {elapsed:.2f} seconds")
    
    # خلاصه نتایج
    print("\n📊 Summary:")
    for model_name, result in results.items():
        if result:
            print(f"  ✅ {model_name}: {len(result)} items processed")
        else:
            print(f"  ❌ {model_name}: failed")
    
    print(f"\n📁 Results saved to: {args.output}/")
    
    # نمایش مسیرهای خروجی
    print("\n📄 Output files:")
    if 'smolvlm' in results:
        print(f"   📄 SmolVLM: {args.output}/text/smolvlm_output.txt")
    if 'nanovlm' in results:
        print(f"   📄 nanoVLM: {args.output}/text/nanovlm_output.txt")
    if 'efficienttam' in results:
        print(f"   🖼️ EfficientTAM: {args.output}/masks/*_mask.png")

if __name__ == "__main__":
    main()
