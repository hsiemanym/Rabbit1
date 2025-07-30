import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import random
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv
import numpy as np
from datetime import datetime

# OpenAI API key (replace this with your actual key or use environment variable)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load attributes
with open("data/OpenAI_API/rabbit_attributes.json", "r") as f:
    attributes = json.load(f)

# Output directory
output_dir = "data/OpenAI_API/generated_rabbits"
os.makedirs(output_dir, exist_ok=True)

# Prompt generator
def generate_prompt(attributes):
    combo = {k: random.choice(v) for k, v in attributes.items()}
    prompt = (
        f"A single, directly facing forward in A-pose, typical character rabbit, full body and centered, with {combo['ears']}, {combo['mouth_nose']}, {combo['eyes']}, "
        f"{combo['accessory']}, showing {combo['style_feature']}, {combo['color']}"
        f"and a {combo['personality']} personality. The background is completely plain white, with no shadows, no props, no other objects, animals or characters. character is isolated."
        f""
    )
    return prompt

# Relaxed edge cleanliness check
def is_image_background_clean(image_path, margin_ratio=0.3, threshold=0.10):
    image = Image.open(image_path).convert("RGB")
    np_img = np.array(image) / 255.0
    h, w, _ = np_img.shape

    white = np.array([1.0, 1.0, 1.0])
    diff = np.abs(np_img - white)
    mask = (diff > 0.35).any(axis=-1)  # MORE lenient: only strongly non-white counts

    margin_h = int(h * margin_ratio)
    margin_w = int(w * margin_ratio)

    outer_mask = np.ones((h, w), dtype=bool)
    outer_mask[margin_h:h - margin_h, margin_w:w - margin_w] = False

    border_nonwhite = (mask & outer_mask).sum()
    border_total = outer_mask.sum()
    border_ratio = border_nonwhite / border_total

    return border_ratio < threshold

# 이미지 생성 + 검사 + 재시도
def generate_image_with_edge_check(prompt, filename, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                n=1,
                response_format="b64_json"
            )
            image_data = response.data[0].b64_json
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            image = image.resize((512, 512), resample=Image.LANCZOS)
            image.save(filename)

            if is_image_background_clean(filename):
                return True
            else:
                print(f"❌ Image has noisy edges. Retrying... ({attempt+1})")
        except Exception as e:
            print(f"Error generating image: {e}")
    return False

# Generate multiple images
def generate_batch(total_images=1000):
    used_prompts = set()
    for i in range(total_images):
        prompt = generate_prompt(attributes)
        if prompt in used_prompts:
            continue
        used_prompts.add(prompt)
        filename = os.path.join(output_dir, f"rabbit_{i + 1:04}.png")
        print(f"[{i + 1}/{total_images}] Generating image: {filename}")
        success = generate_image_with_edge_check(prompt, filename)
        if not success:
            print(f"⚠️ Failed to generate valid image for: {filename}")
            continue
    print("Done!")

if __name__ == "__main__":
    generate_batch(10)  # Adjust this number as needed
