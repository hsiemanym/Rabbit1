import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# scripts/demo_pipeline.py

import os
import argparse
import subprocess

def run_step(description, command):
    print(f"\n[▶] {description}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"[✗] 실패: {description}")
        exit(1)
    print(f"[✓] 완료: {description}")

def main(test_img_path):
    run_step("1. 레퍼런스 이미지 임베딩 추출", "python scripts/extract_features.py")
    run_step("2. Generic Feature Bank 생성 (Option A)", "python scripts/build_gfb.py --option A")
    run_step("3. Top-1 유사 이미지 검색", f"python scripts/retrieve_top1.py --test_dir {os.path.dirname(test_img_path)}")
    run_step("4. Grad-CAM + GFB 설명 시각화", f"python scripts/explain_similarity.py --test_img {test_img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img", type=str, required=True, help="Path to a test image")
    args = parser.parse_args()

    main(args.test_img)
