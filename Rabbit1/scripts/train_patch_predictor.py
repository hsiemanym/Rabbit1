import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# scripts/train_patch_predictor.py
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from models.patch_sim_predictor import PatchSimPredictor

def train_patch_predictor(patch_pair_path, save_path, epochs=20, batch_size=128, lr=1e-3):
    # 데이터 로드
    data = torch.load(patch_pair_path)
    X = data['pairs']     # [N, 4096]
    y = data['labels'].float()  # [N]

    input_dim = X.shape[1] // 2
    x1 = X[:, :input_dim]
    x2 = X[:, input_dim:]

    dataset = TensorDataset(x1, x2, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchSimPredictor(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # 훈련 루프
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x1, batch_x2, batch_y in loader:
            batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)

            pred = model(batch_x1, batch_x2).squeeze(1)   # [B]
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x1.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), save_path)
    print(f"[✓] Trained model saved to {save_path}")

    # ----------------------------
    # ✅ sigmoid 출력 분포 시각화
    # ----------------------------
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for batch_x1, batch_x2, batch_y in loader:
            batch_x1, batch_x2 = batch_x1.to(device), batch_x2.to(device)
            preds = model(batch_x1, batch_x2).squeeze(1).cpu()
            all_preds.append(preds)
            all_targets.append(batch_y)

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

    # 히스토그램 시각화
    os.makedirs("output", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(all_preds[all_targets == 1], bins=50, alpha=0.6, label='Positive')
    plt.hist(all_preds[all_targets == 0], bins=50, alpha=0.6, label='Negative')
    plt.xlabel("Predicted similarity")
    plt.ylabel("Count")
    plt.title("Patch Similarity Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/sim_score_hist.png")
    print("[✓] Saved similarity score histogram to output/sim_score_hist.png")

    # 예시 출력 확인
    print("Positive examples (pred[:5]):", all_preds[all_targets == 1][:5])
    print("Negative examples (pred[:5]):", all_preds[all_targets == 0][:5])

if __name__ == "__main__":
    patch_pair_path = "data/patch_pairs/patch_pairs.pth"
    save_path = "models/patch_sim_predictor.pth"
    train_patch_predictor(patch_pair_path, save_path)
