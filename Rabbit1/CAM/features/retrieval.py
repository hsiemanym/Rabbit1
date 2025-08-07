'''
features/embeddings.py
features/retrieval_features.py
'''

# features/retrieval.py
import os
import torch
from PIL import Image
import clip
from sklearn.metrics.pairwise import cosine_similarity
from utils.image_utils import load_and_preprocess_image
import torch.nn.functional as F

# --- 임베딩 계산 로직 (기존 embeddings.py) ---
def compute_embedding(model, x, no_grad=True):
    """SimSiamBackbone 모델로부터 이미지 임베딩을 계산합니다."""
    if no_grad:
        with torch.no_grad():
            return model.get_embedding(x)
    else:
        return model.get_embedding(x)


# --- 임베딩 추출 및 검색 로직 (기존 retrieval_features.py) ---
def extract_reference_embeddings(model, config, transform, device):
    """참조 이미지 디렉터리에서 임베딩을 추출하고 캐시 파일로 저장합니다."""
    image_dir = config['data']['reference_dir']
    save_path = config['features']['embedding_cache']

    # 캐시 파일이 있으면 바로 로드
    if os.path.exists(save_path):
        print(f"[✓] Loading reference embeddings from cache: {save_path}")
        return torch.load(save_path)

    print(f"[!] No cache found. Extracting reference embeddings...")
    embeddings = {}
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        path = os.path.join(image_dir, fname)
        image = load_and_preprocess_image(path, transform).unsqueeze(0).to(device)
        emb = compute_embedding(model, image, no_grad=True)
        embeddings[fname] = emb.squeeze(0).cpu()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(embeddings, save_path)
    print(f"[✓] Saved new reference embeddings to {save_path}")
    return embeddings

def retrieve_top1_clipfiltered(test_img_path, model, reference_dict, config, transform, device):
    """SimSiam top-5 후보 중 CLIP 점수가 가장 높은 이미지를 검색합니다."""
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    names = list(reference_dict.keys())
    mat = torch.stack(list(reference_dict.values()))
    test_img = load_and_preprocess_image(test_img_path, transform).unsqueeze(0).to(device)
    test_emb = compute_embedding(model, test_img, no_grad=True)

    sims = F.cosine_similarity(test_emb, mat.to(device))
    top5_indices = torch.topk(sims, k=5).indices
    top5_names = [names[i] for i in top5_indices]

    def clip_score(img1_path, img2_path):
        image1 = preprocess(Image.open(img1_path)).unsqueeze(0).to(device)
        image2 = preprocess(Image.open(img2_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb1 = clip_model.encode_image(image1).float().T
            emb2 = clip_model.encode_image(image2).float()
            emb1 = F.normalize(emb1, p=2, dim=0)
            emb2 = F.normalize(emb2, p=2, dim=1)
        return (emb2 @ emb1).item()

    best_score = -1
    best_name = None
    ref_dir = config['data']['reference_dir']

    for name in top5_names:
        ref_path = os.path.join(ref_dir, name)
        score = clip_score(test_img_path, ref_path)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name, best_score