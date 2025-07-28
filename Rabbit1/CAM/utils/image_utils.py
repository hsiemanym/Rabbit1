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

def assemble_2x2_grid(img_list, rows=2, cols=2, labels=None):
    assert len(img_list) == rows * cols
    if labels is not None:
        img_list = [annotate_image(img.copy(), lbl) for img, lbl in zip(img_list, labels)]

    w, h = img_list[0].size
    grid = Image.new('RGB', (cols * w, rows * h))
    for i in range(rows):
        for j in range(cols):
            grid.paste(img_list[i * cols + j], (j * w, i * h))
    return grid


def save_image(image, path):
    """
    Saves a PIL.Image to disk.
    """
    image.save(path)
