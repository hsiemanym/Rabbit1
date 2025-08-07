# features/retrieval_features.py
import os
import torch
from PIL import Image
import clip
from sklearn.metrics.pairwise import cosine_similarity

from utils.image_utils import load_and_preprocess_image
from features.embeddings import compute_embedding


def extract_reference_embeddings(model, config, transform, device):
    """참조 이미지 디렉터리에서 임베딩을 추출하고 캐시 파일로 저장합니다."""
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


def retrieve_top1_clipfiltered(test_img_path, model, reference_dict, config, transform, device):
    """SimSiam top-5 후보 중 CLIP 점수가 가장 높은 이미지를 검색합니다."""
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