# utils/gradcam_utils.py
import torch.nn.functional as F

def get_factual_gradcam(model, gradcam, query_tensor, target_emb):
    """쿼리 이미지와 타겟 임베딩 사이의 유사성을 기반으로 Grad-CAM 히트맵을 생성합니다."""
    query_emb = model.get_embedding(query_tensor)
    similarity = F.cosine_similarity(query_emb, target_emb.detach(), dim=1)
    target_score = similarity.sum()
    cam = gradcam.generate(input_tensor=query_tensor, target_score=target_score)[0]
    return cam

def generate_gradcam_heatmap(gradcam, input_tensor, target_tensor):
    """두 텐서 간의 코사인 유사도를 타겟으로 Grad-CAM을 생성합니다."""
    sim = F.cosine_similarity(input_tensor, target_tensor, dim=1)
    score = sim.sum()
    cam = gradcam.generate(input_tensor=input_tensor, target_score=score)[0]
    return cam