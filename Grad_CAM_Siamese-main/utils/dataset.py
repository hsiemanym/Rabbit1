import torch
from torchvision import transforms
from PIL import Image

class Dataset(): # 이미지 파일을 불러오고, ㅈㄴ처리(trnsformations)를 적용하는 역할
    def __init__(self, data:list=None, tfms:transforms=None)->None:
        '''
            Create dataset used in data loader

            Parameters
            ----------
            data: list
                [anchor image, positive/negative image, class]
            tfms: transforms
                transformator

        '''
        # 초기화할 때 이미지 쌍 목록과 전처리 함수를 받음
        self.data = data # list of [img1_path, img2_path, label]
        self.tfms = tfms # transformations to apply (전처리 함수 e.g resize, normalize)
    
    def __len__(self):
        return len(self.data) # 전체 데이터 개수 반환 (필수)
    
    def __getitem__(self, idx): # 주어진 이미지 파일을 열어 RGB로 변환
        data = self.data[idx] # 주어진 인덱스의 이미지 경로 2개와 라벨을 꺼내고

        image1 = Image.open(data[0]).convert('RGB') # 각 이미지를 RGB 포맷으로 열어옴
        image2 = Image.open(data[1]).convert('RGB')

        # Apply image transformations
        if self.tfms is not None: # 전처리 함수 tfms가 주어졌다면
            image1 = self.tfms(image1) # 각 이미지에 적용
            image2 = self.tfms(image2) # (예: transforms.Compose([...]) 형태로 resize, toTensor 등 포함 가능)
        
        return image1, image2, data[2]   # 최종적으로 (변환된 이미지1, 이미지2, 라벨) 튜플 반환
                                        # -> PyTorch 텐서로 반환되어, 모델 학습 시 (tensor1, tensor2, label) 형태로 바로 사용