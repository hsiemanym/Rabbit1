import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# utils/image_utils.py

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def load_and_preprocess_image(path, transform):
    """
    Loads an image from disk and applies a transform (e.g., resize + normalize).
    Returns: Tensor [3, H, W]
    """
    image = Image.open(path).convert('RGB')
    return transform(image)


def overlay_heatmap(image, heatmap, colormap='JET', alpha=0.5):
    """
    Overlays a heatmap onto a PIL image.
    Args:
        image: PIL.Image (RGB), size (W, H)
        heatmap: numpy array [H, W], values in [0,1]
        colormap: OpenCV colormap string (e.g., 'JET')
        alpha: blending ratio
    Returns:
        PIL.Image with heatmap overlay
    """
    image = image.convert('RGB')
    orig = np.array(image)

    heatmap = np.uint8(255 * heatmap)  # [H, W] → [0, 255]
    color_map = getattr(cv2, f'COLORMAP_{colormap.upper()}')
    heatmap_color = cv2.applyColorMap(heatmap, color_map)  # [H, W, 3]
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    blended = np.uint8(orig * (1 - alpha) + heatmap_color * alpha)
    return Image.fromarray(blended)


def annotate_image(img, text):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    draw.rectangle([(0, 0), (img.width, 30)], fill=(0, 0, 0, 180))
    draw.text((10, 5), text, fill="white", font=font)
    return img

def assemble_2x2_grid(img_list, labels=None, rows=2, cols=2, font_size=20):
    assert len(img_list) == rows * cols, f"Expected {rows * cols} images, got {len(img_list)}"

    w, h = img_list[0].size
    grid = Image.new('RGB', (cols * w, rows * h), color=(255, 255, 255))

    font = ImageFont.load_default()

    draw = ImageDraw.Draw(grid)

    for idx, img in enumerate(img_list):
        row = idx // cols
        col = idx % cols
        grid.paste(img, (col * w, row * h))
        if labels and idx < len(labels):
            draw.text((col * w + 10, row * h + 10), labels[idx], fill=(255, 0, 0), font=font)

    return grid



def save_image(image, path):
    """
    Saves a PIL.Image to disk.
    """
    image.save(path)
