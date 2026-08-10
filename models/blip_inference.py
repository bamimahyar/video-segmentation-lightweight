import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys
from pathlib import Path

class BLIPInference:
    """کلاس برای تولید توضیحات دقیق تصویر با BLIP"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📦 Loading BLIP model on {self.device}...")
        
        # بارگذاری مدل BLIP
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model.to(self.device)
        self.model.eval()
        print("✅ BLIP loaded successfully!")
    
    def generate_description(self, image_path, prompt="A picture of"):
        """تولید توضیحات دقیق با پرامپت دلخواه"""
        image = Image.open(image_path).convert('RGB')
        
        # پردازش با پرامپت
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                temperature=0.7,
                do_sample=True
            )
        
        description = self.processor.decode(out[0], skip_special_tokens=True)
        return description
    
    def describe_objects(self, image_path):
        """پرسیدن دقیق درباره اشیاء تصویر"""
        prompts = [
            "What objects are in this image?",
            "Describe the scene in detail",
            "What can you see in this picture?",
            "List the main objects in this image",
            "What is happening in this image?"
        ]
        
        results = {}
        for prompt in prompts:
            desc = self.generate_description(image_path, prompt)
            results[prompt] = desc
        
        return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='sample_input/00035.jpg', help='Path to image')
    parser.add_argument('--prompt', type=str, default='What objects are in this image?', 
                       help='Custom prompt for description')
    args = parser.parse_args()
    
    # بارگذاری مدل
    model = BLIPInference()
    
    # تولید توضیحات با پرامپت دلخواه
    print(f"\n📸 Image: {args.image}")
    print(f"📝 Prompt: {args.prompt}")
    print("-"*50)
    
    desc = model.generate_description(args.image, args.prompt)
    print(f"\n✅ Description: {desc}")
    
    # اگر پرامپت پیش‌فرض نبود، چند پرامپت مختلف امتحان کن
    if args.prompt == "What objects are in this image?":
        print("\n" + "="*50)
        print("📊 Testing multiple prompts:")
        print("="*50)
        
        results = model.describe_objects(args.image)
        for prompt, desc in results.items():
            print(f"\n📝 {prompt}")
            print(f"   → {desc}")

if __name__ == "__main__":
    main()
