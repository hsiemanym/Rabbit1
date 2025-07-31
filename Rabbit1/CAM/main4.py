import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from models.backbone import SimSiamBackbone
from features.embeddings import compute_embedding
from features.generic_feature_bank import build_gfb
from models.gradcam import GradCAM
from utils.image_utils import (
    load_and_preprocess_image,
    overlay_heatmap,
    assemble_2x2_grid,
    save_image
)
from utils.similarity_utils import compute_counterfactual_score
from sklearn.metrics.pairwise import cosine_similarity


def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)


def extract_reference_embeddings(model, config, transform):
    image_dir = config['data']['reference_dir']
    save_path = config['features']['embedding_cache']
    embeddings = {}

    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        path = os.path.join(image_dir, fname)
        image = load_and_preprocess_image(path, transform).unsqueeze(0).to(device)
        emb = compute_embedding(model, image)
        embeddings[fname] = emb.squeeze(0).cpu()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(embeddings, save_path)
    print(f"[✓] Saved reference embeddings to {save_path}")
    return embeddings


def retrieve_top1_clipfiltered(test_img_path, model, reference_dict, config, transform):

    """
    SimSiam cosine similarity top-5 → 그 중 CLIPScore 최고 후보 반환
    """
    from utils.image_utils import save_image
    import clip
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    names = list(reference_dict.keys())
    mat = torch.stack([reference_dict[n] for n in names])
    test_img = load_and_preprocess_image(test_img_path, transform).unsqueeze(0).to(device)
    test_emb = compute_embedding(model, test_img, no_grad=True)

    sims = cosine_similarity(test_emb.cpu().numpy(), mat.numpy())  # [1, N]
    idxs = sims[0].argsort()[::-1][:5]
    top5_names = [names[i] for i in idxs]

    def clip_score(img1_path, img2_path):
        image1 = preprocess(Image.open(img1_path)).unsqueeze(0).to(device)
        image2 = preprocess(Image.open(img2_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb1 = clip_model.encode_image(image1).float()
            emb2 = clip_model.encode_image(image2).float()
            emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
            emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
        return (emb1 * emb2).sum().item()

    best_score = -1
    best_name = None
    test_img_abs = os.path.abspath(test_img_path)
    ref_dir = config['data']['reference_dir']

    for name in top5_names:
        ref_path = os.path.join(ref_dir, name)
        score = clip_score(test_img_abs, ref_path)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name, best_score



def get_masked_cam(model, gradcam, query_img, target_emb, fmap, gfb_tensor):
    feat = model.forward_backbone(query_img)
    pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
    pooled.requires_grad_(True)
    z1 = model.projector(pooled)
    z2 = model.projector(target_emb.detach())
    sim = F.cosine_similarity(z1, z2, dim=1).sum()
    cam = gradcam.generate(query_img, sim)[0]
    return filter_with_gfb(cam, fmap, gfb_tensor)


def filter_with_gfb(heatmap, fmap, gfb, threshold=0.7):
    C, H, W = fmap.shape
    mask = torch.zeros((H, W), dtype=torch.float32, device=fmap.device)
    for y in range(H):
        for x in range(W):
            patch = fmap[:, y, x]
            sims = F.cosine_similarity(patch.unsqueeze(0), gfb)
            if sims.max() < threshold:
                mask[y, x] = 1.0
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=heatmap.shape, mode='bilinear', align_corners=False)
    return (heatmap * mask.squeeze().cpu().numpy())


def run_pipeline(test_img_path, gfb_option='A'):
    config = load_config()
    model = SimSiamBackbone(pretrained=True).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(config['image']['size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['image']['normalization']['mean'],
                             std=config['image']['normalization']['std'])
    ])

    reference_embeddings = extract_reference_embeddings(model, config, transform)

    gfb_path = config['features']['generic_bank']
    patch_file = config['features']['patch_file']
    from scripts.extract_patches import extract_all_patches
    if not os.path.exists(patch_file):
        print(f"[!] Patch file not found: {patch_file}")
        extract_all_patches(model, config, transform, device)
    gfb_tensor = build_gfb(patch_file, save_path=gfb_path, option=gfb_option).to(device)

    # Load test image
    img_test = load_and_preprocess_image(test_img_path, transform).unsqueeze(0).to(device)
    emb_test = compute_embedding(model, img_test, no_grad=True)
    top1_name, sim_score = retrieve_top1_clipfiltered(test_img_path, model, reference_embeddings, config, transform)

    top1_path = os.path.join(config['data']['reference_dir'], top1_name)
    img_ref = load_and_preprocess_image(top1_path, transform).unsqueeze(0).to(device)
    emb_ref = compute_embedding(model, img_ref, no_grad=True)

    print(f"[✓] Top-1 for {os.path.basename(test_img_path)} → {top1_name}  ({sim_score:.4f})")

    # Grad-CAM 준비
    gradcam = GradCAM(model, ['encoder.7'])
    fmap_test = model.get_feature_map(img_test).squeeze(0)
    fmap_ref = model.get_feature_map(img_ref).squeeze(0)

    raw_test = Image.open(test_img_path).convert('RGB').resize(config['image']['size'])
    raw_ref = Image.open(top1_path).convert('RGB').resize(config['image']['size'])

    # -------- 1. Grad-CAM only (NO GFB) --------
    z_test = model.get_embedding(img_test)
    z_ref = model.get_embedding(img_ref)
    score_test = F.cosine_similarity(z_test, z_ref).sum()
    score_ref = F.cosine_similarity(z_ref, z_test).sum()
    cam0_test = gradcam.generate(img_test, score_test)[0]
    cam0_ref = gradcam.generate(img_ref, score_ref)[0]
    vis0 = overlay_heatmap(raw_test, cam0_test)
    vis1 = overlay_heatmap(raw_ref, cam0_ref)

    # -------- 2. Factual (GFB 필터) --------
    cam1 = get_masked_cam(model, gradcam, img_test, emb_ref, fmap_test, gfb_tensor)
    cam2 = get_masked_cam(model, gradcam, img_ref, emb_test, fmap_ref, gfb_tensor)
    vis2 = overlay_heatmap(raw_test, cam1)
    vis3 = overlay_heatmap(raw_ref, cam2)

    # -------- 3. Counterfactual (Disjoint 영역 강조) --------
    inv_emb_test = -emb_test
    inv_emb_ref = -emb_ref
    cam3 = get_masked_cam(model, gradcam, img_test, inv_emb_ref, fmap_test, gfb_tensor)
    cam4 = get_masked_cam(model, gradcam, img_ref, inv_emb_test, fmap_ref, gfb_tensor)
    vis4 = overlay_heatmap(raw_test, cam3)
    vis5 = overlay_heatmap(raw_ref, cam4)

    # -------- Save Visualization --------
    labels = [
        "Test - Factual (No GFB)", "Ref - Factual (No GFB)",
        "Test - Factual (GFB)", "Ref - Factual (GFB)",
        "Test - Counterfactual", "Ref - Counterfactual"
    ]
    grid = assemble_2x2_grid([vis0, vis1, vis2, vis3, vis4, vis5], labels=labels, rows=3, cols=2)
    os.makedirs('output', exist_ok=True)
    fname = os.path.splitext(os.path.basename(test_img_path))[0]
    save_image(grid, f'output/main4_{fname}_explanation_full.png')
    print(f"[✓] Saved 2x3 explanation to output/main4_{fname}_explanation_full.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img", type=str, required=True)
    parser.add_argument("--gfb_option", type=str, default='A', choices=['A', 'B'])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_pipeline(args.test_img, args.gfb_option)
