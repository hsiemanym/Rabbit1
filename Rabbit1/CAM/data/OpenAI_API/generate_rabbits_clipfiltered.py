import os
import sys
import json
import random
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv
import torch
from transformers import CLIPProcessor, CLIPModel

# GPU 설정 비활성화 시: os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load attributes
with open("data/OpenAI_API/rabbit_attributes.json", "r") as f:
    attributes = json.load(f)

# Output dir
output_dir = "data/OpenAI_API/generated_rabbits"
os.makedirs(output_dir, exist_ok=True)

# CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Prompt generator
def generate_prompt(attributes):
    combo = {k: random.choice(v) for k, v in attributes.items()}
    prompt = (
        f"A single, directly facing forward in A-pose, typical character rabbit, full body and centered, "
        f"with {combo['ears']}, {combo['mouth_nose']}, {combo['eyes']}, {combo['accessory']},  {combo['color']}, {combo['styles']}, "
        f"showing {combo['style_feature']}, and a {combo['personality']} personality. "
        "The background is completely plain white, with no shadows, no props, no other objects, animals or characters. "
        "Character is isolated."
    )
    return prompt


# CLIPScore filter
def is_semantically_valid(image_path, processor, model, device, caption, threshold=0.31):
    image = Image.open(image_path).convert("RGB").resize((512, 512))

    # 긍정 텍스트 임베딩
    with torch.no_grad():
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_inputs = processor(text=[caption], return_tensors="pt").to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        score = torch.matmul(image_features, text_features.T).item()
        if score < threshold:
            print(f"❌ Failed Positive CLIPScore ({score:.3f} < {threshold})")
            return False

        # 부정 문장 리스트
        negative_captions = [
            "multiple characters",
            "background is not plain",
            "not a bunny",
            "diagram or sketch",
            "group of rabbits",
            "comic panel",
        ]

        for neg_caption in negative_captions:
            neg_text_input = processor(text=[neg_caption], return_tensors="pt").to(device)
            neg_text_features = model.get_text_features(**neg_text_input)
            neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)

            trshd = 0.286
            neg_score = torch.matmul(image_features, neg_text_features.T).item()
            if neg_score > trshd:  # cosine similarity 기준
                print(f"❌ Detected negative concept: '{neg_caption}' ({neg_score:.3f} > {trshd:.3f})")
                return False

    return True


# Image generation
def generate_image_with_clipscore(prompt, filename, max_retries=3, i=1):
    caption = "A single cute bunny character, A-pose, frontside, centered, full body, plain white background"
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
            fname = filename.replace(".png", f"_{attempt + 1}.png")  # Corrected this line
            image.save(fname)

            if is_semantically_valid(fname, clip_processor, clip_model, device, caption=caption, threshold=0.28):
                return [True, fname]
            else:
                print(f"CLIPScore check failed. Retrying... ({attempt + 1})")

        except Exception as e:
            print(f"Error generating image: {e}")

    return [False, fname]  # In case of failure, return False and filename


# Batch run
def generate_batch(total_images=100):
    used_prompts = set()
    for i in range(total_images):
        prompt = generate_prompt(attributes)
        if prompt in used_prompts:
            continue
        used_prompts.add(prompt)
        fname = os.path.join(output_dir, f"rabbit_{i + 400:03}.png")
        success, generated_fname = generate_image_with_clipscore(prompt, fname)
        print(f"[{i + 1}/{total_images}] Generating image: {generated_fname}")
        if not success:
            print(f"⚠️ Failed to generate valid image for: {generated_fname}")
    print("✅ Done!")


if __name__ == "__main__":
    generate_batch(100)
