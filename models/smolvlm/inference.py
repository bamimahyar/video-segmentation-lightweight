import torch
import sys
from pathlib import Path
import json
from PIL import Image
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

class SmolVLMInference:
    def __init__(self, model_path="checkpoints/smolvlm/model_best.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        self.model_name = "microsoft/Florence-2-base"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"✅ SmolVLM (Florence-2) loaded for text generation")

    def generate_description(self, image_path, prompt=None):
        if prompt is None:
            prompt = "<DETAILED_CAPTION>"
        
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        inputs["input_ids"] = inputs["input_ids"].to(self.device).long()
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device).to(self.torch_dtype)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=100,
                num_beams=3,
                do_sample=False
            )
            description = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            description = self.processor.post_process_generation(description, task=prompt, image_size=(image.width, image.height))
            if isinstance(description, dict):
                description = str(description)
        return description

    def process_video(self, video_path, output_dir="outputs/text"):
        """پردازش و ذخیره با فرمت smol.اسم_تصویر.txt"""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        image_name = video_path.stem  # اسم فایل بدون پسوند
        
        if video_path.is_dir():
            images = list(video_path.glob("*.jpg")) + list(video_path.glob("*.png"))
            for img in images[:10]:
                desc = self.generate_description(img)
                # ذخیره با فرمت smol.اسم_تصویر.txt
                txt_path = output_dir / f"smol.{img.stem}.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(desc)
                results.append({
                    "frame": img.name,
                    "description": desc,
                    "file": str(txt_path)
                })
        else:
            desc = self.generate_description(video_path)
            # ذخیره با فرمت smol.اسم_تصویر.txt
            txt_path = output_dir / f"smol.{image_name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(desc)
            results = [{
                "frame": video_path.name,
                "description": desc,
                "file": str(txt_path)
            }]
        
        print(f"✅ SmolVLM output saved to: {txt_path}")
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path to image or video folder')
    parser.add_argument('--output', type=str, default='outputs/text')
    args = parser.parse_args()
    inference = SmolVLMInference()
    results = inference.process_video(args.image, args.output)
    print(f"✅ Generated {len(results)} descriptions")
