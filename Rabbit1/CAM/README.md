Rabbit1/

├── data/

│ ├── images/ # 토끼 캐릭터 이미지 (원본)

│ ├── test/ # 테스트용 이미지

├── features/

│ ├── embeddings.pth # 레퍼런스 이미지 임베딩 벡터 (사전 계산 캐시)

│ ├── generic_bank.pth # Generic Feature Bank (패치 벡터 모음)

├── models/

│ ├── backbone.py # SimSiam 백본 모델

│ ├── gradcam.py # Grad-CAM/Grad-CAM++ 모듈

├── scripts/

│ ├── extract_features.py # 이미지 임베딩 추출

│ ├── build_gfb.py # GFB 생성

│ ├── retrieve_top1.py # 유사 이미지 검색

│ ├── explain_similarity.py# heatmap + GFB + 시각화

│ ├── demo_pipeline.py # 전체 파이프라인 실행 데모

├── utils/

│ ├── image_utils.py # 시각화, overlay, 이미지 입출력

│ ├── similarity_utils.py # 코사인 유사도, 정렬 등

│ ├── eval_metrics.py # (선택) heatmap 정량 평가 지표

├── output/ # 시각화 결과 저장

├── config.yml # 파라미터 설정

└── README.md