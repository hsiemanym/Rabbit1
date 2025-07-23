import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from   torch.autograd   import Variable 
from typing import Any, Dict, Sequence, Union
TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, label, distance):
        '''
        Contrastive loss

        Parameters
        ----------
        label: true label
        distance: distance between instances

        Returns
        -------
        Loss value
        '''        
        loss_contrastive = torch.mean( (1.0-label) * torch.pow(distance, 2) + (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return loss_contrastive
        
        

class Softplus(nn.Module): # 일반적인 Softplus가 아니라 사용자 정의 구현
    def __init__(self):
        super(Softplus, self).__init__()
    def forward(self, x):
        return torch.where(x < 1.0, x, 1.0)  # 거리 값을 1.0으로 상한capping (1보다 큰 값은 1로, 1 이하는 그대로 유지) -> 0~1
                                            # 해석이 쉬워지고, 출력값을 0~1의 비유사도 점수로 바로 사용 가능 (0이면 동일, 1이면 상이(or 최소한 margin 이상 다름)

class Flatten(nn.Module):
    def forward(self, *args: TorchData, **kwargs: Any) -> torch.Tensor:
        assert len(args) == 1
        x = args[0]
        assert isinstance(x, torch.Tensor)
        return x.view(x.size(0), -1)
    

class SiameseNetwork(nn.Module):
    def __init__(self, image_size=(3, 224, 224), backbone_model:str='MobileNet', pretrained:bool=True):
        super(SiameseNetwork, self).__init__()
        
        self.backbone_model = backbone_model
        self.distance = F.pairwise_distance # 두 벡터 사이의 유클라디안 거리 계산 (임베딩된 이미지 쌍 서리로 유사도 수치화)
        self.loss_fnc = ContrastiveLoss() # 이미지 쌍 유사: 거리 작게 // 비유사: 거리 크게 학습 유도
        # Placeholder for the gradients
        self.gradients_1 = None # image1, 2에 대한 Grad-CAM 시각화를 위해 필요한 gradient 저장
        self.gradients_2 = None # Siamese에서는 두 이미지가 각각 독립된 백본 경로(sub-network)를 통과하기 때문에 grad-CAM 시각화를 위해 각각 저장해야 함





        # Backbone model and pooling layer
        # -------------------------------------------------------------------------------------------
        if self.backbone_model == 'MobileNet':
            # Siamese network backbone: MobileNet_V2
            self.backbone = torchvision.models.mobilenet_v2(pretrained=pretrained).features
            # Pooling
            self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        elif self.backbone_model == 'ResNet50':
            # Siamese network backbone: ResNet50
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            self.backbone = nn.Sequential(*list(model.children())[:-2])
            # Pooling
            self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        else:
            raise Exception('Not known pre-trained model')



        # Dense-layer
        # ------------------------------------------------------------------------------------------- # full connected layers to produce embedding vector
        n_size = self._get_conv_output( image_size )  # determine the number of features coming out of the backbone's pooling layer
        self.fc = nn.Sequential(
            nn.Linear(in_features=n_size, out_features=256), # n_size 정확히 몇인지 미리 알아야 함: 이미지 크기나 백본 모델에 따라 피처 크기가 달라짐
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=2),  
        )    



    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable( torch.rand(bs, *shape) ) # 더미 텐서 생성
        output_feat = self._forward_features( input ) # 백본 + 풀링 통과
        return output_feat.data.view(bs, -1).size(1) # flatten 후 차원 길이 반환

    def _forward_features(self, x):
        output = self.backbone(x)
        return self.pooling(output)

    # method for the activation extraction
    def get_activations(self, x): # 입력 x를 풀링 없이 백본 CNN에 통과시켜 마지막 합성곱 층의 피처맵 반환
        return self.backbone(x) # 풀링 없이 raw cone features을 얻기 위해 사용 (Grad-CAM에서 중요)
    
    # hook for the gradients of the activations
    def activations_hook_image1(self, grad): # gradient 저장하는 콜백 함수
        self.gradients_1 = grad # 각각의 서브(백본) 네트워크에서 역전파(backprop)가 일어날 때 그 지점의 grad를 저장
    def activations_hook_image2(self, grad):
        self.gradients_2 = grad

    def get_activations_gradient(self, sub_network=1): # 이미지 1인지 2인지에 따라 해당 grad 반환
        if sub_network == 1:
            return self.gradients_1
        else:
            return self.gradients_2


    def forward(self, image1, image2, label=None):
        # Sub-Network 1
        # ----------------------------------------------------------
        embeddings1 = self.backbone(image1) # image1을 self.backbone(conv layers)을 통과시켜 embeddings1(합성곱 피쳐맵) 얻음
        # register the hook
        _ = embeddings1.register_hook(self.activations_hook_image1)   # embeddings1 얻은 직후 훅 호출 -> embd1 계산 시 a_h_i1 호출하고 grad 넘겨줌
        # apply the remaining pooling                                   (파이토치, pred backward() 호출되면 자동으로 self.gradient = grad 실행)
        embeddings1 = self.pooling(embeddings1) # embd1은 self.pooling 통해 1*1로 aver- pool되고
        embeddings1 = Flatten()(embeddings1) # Flatten()을 이용해 벡터 형태로 납작하게 (텐서의 형태만 바꿔줌)
        embeddings1 = self.fc(embeddings1) # embd1은 이후 self.fc 통과하여 image1에 대한 2차원 임베딩 벡터 생성


        # Sub-Network 2
        # ----------------------------------------------------------
        embeddings2 = self.backbone(image2)      # image2도 동일한 연산 수행(이 때는 activations_hook_image2가 grad 저장)
        # register the hook
        _ = embeddings2.register_hook(self.activations_hook_image2)
        # apply the remaining pooling
        embeddings2 = self.pooling(embeddings2)
        embeddings2 = Flatten()(embeddings2)
        embeddings2 = self.fc(embeddings2)   # 결과 : 2차원 임베딩 생성
        # i1 i2의 두 2D vectors: [batch_size, 2] 형태

        distance = self.distance(embeddings1, embeddings2) # distance = F.pairwise_distance(embd1, embd2) (배치 내의 각 쌍에 대해 유클라디안 거리 계산 -> distance(출력)는 [batch_size]크기의 텐서. 입력 쌍마다 하나의 거리 값 들어감)
        distance = Softplus()(distance) # distance에 Softplus 적용: 일반적인 softplus가 아니라 사용자 정의 함수임 (line 38)
        if label is not None:  # label이 제공되었다면 ( = 학습 모드이며 정답 유사도가 주어진 경우)
            loss = self.loss_fnc(label, distance.double())
            return distance, loss # forward는 distance와 해당 라벨에 대해 계산된 loss를 함께 반환
        else: # 라벨이 없을 경우 (추론 모드)
            return distance # 오직 distance만 반환
        '''
        ┌──────────────┐
        │ forward()    │ ← image1, image2 입력 → embeddings1,2 → distance 계산
        └──────┬───────┘
               ↓
        [외부 코드]
        distance, loss = model(img1, img2, label)
        loss.backward()        ← 여기서 hook 발동
        ┌──────────────┐
        │  hook 실행    │ → self.gradients_1 = grad
        └──────────────┘
        '''


'''
Siamese network는 이미지 쌍마다 하나의 숫자를 출력하며, 두 이미지가 얼마나 다른지 나타냄
0: 유사 ~ 1: 다름 (클래스)
0.5를 threshold로 사용 -> 평가 단계에서 distance > 0.5면 pred = 1, 아니면 0으로 설정해 정답과 비교

* 중요 *
이 모델은 ImageNet에서 사전 학습된 MobileNet / ResNet 백본을 출발점으로 사용
    (시각적 패턴을 잘 포착하는 특징들을 이미 학습해두었기 때문에 도움이 되므로)
마지막의 FC 계층(nn.Linear(...))은 랜덤 초기화 후 처음부터 새로 학습되며(사전학습 X), 이 피쳐들을 특정 유사도 판단 작업에 맞게 조정하여 사용
(백본(CNN)은 ImageNet 같은 일반 데이터셋에서 학습되어 일반적인 시각적 특징을 추출하는데, 유사도 판단이라는 특정 작업에 맞추려면 FC레이어가 이 특징들을 유사도 계산에 적합한 임베딩 벡터로 변환해줘야 함)
백본: "보는 법을 아는 전문가", 이미 학습된 “보는 능력”을 가진 상태
FC 레이어: "그걸 가지고 유사도를 판단하는 사람", 출력을 받아서, “이 두 이미지는 얼마나 비슷한가?”를 판단할 수 있는 2차원 벡터로 바꿔주는 역할을 새로 학습
'''







