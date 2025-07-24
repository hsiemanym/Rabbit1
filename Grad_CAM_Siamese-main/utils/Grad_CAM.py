import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from utils.Siamese import SiameseNetwork

def convert_to_tensor(image:np.ndarray=None, tfms:transforms=None, device:str='cpu')->torch.Tensor:
    # 1. tfms를 이용해 PIL 이미지 A에 대해 resize, tensor 변환, normalize 처리
    # 2. .to(device) 호출해서 GPU/CPU로 이동시킴
    # 3. 결과: image는 [C, H, W] 형태의 파이토치 텐서
    '''
        Convert an input image to torch.Tensor

        Parameters
        ----------
        image: input image
        tfms: list of transformations
        device: cpu/cude


        Returns
        -------
        image in torch.Tensor
    '''
    # Apply image transformations
    image = tfms(image).to(device)

    return image


    
def Gram_CAM_heatmap(image:torch.Tensor, model:SiameseNetwork=None, sub_network:int=None,\
image_size:tuple=None, figname:str=None, figsize:tuple=(3,3)):
    '''
        Calculates the heatmap for Gram-CAM procedure

        Parameters
        ----------
        image: requested image
        model: Similarity model (Siamese network)
        sub_network: requested sub_network 1 or 2
        image_size: image size
        figname: figure name (optional)
        fisize: figure size (optional)

        Returns
        -------
        heatmap
    '''

    # pull the gradients out of the model : .backward() 후에 hook을 통해 저장된 gradient(self.gradients_1(2))를 가져옴
    gradients = model.get_activations_gradient(sub_network=1) # sub_network가 1이면 image1, 2면 image2쪽 gradient [1(배치 크기), C(채널 수(예: 128개 feature map)), H,W(각 feature map의 세로, 가로 크기)]

    # pool the gradients across the channels: global average pooling(중요도 weight 만들기)
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) # 배치 차원(0), 공간 차원(2, 3)을 모두 평균내면, 남는 건 채널 차원 하나 뿐(각 채널별 gradient를 공간적으로 평균냄) -> shape:[C] (C개의 값으로 구성된 1차원 텐서. 나중에 각 채널별 activation에 곱해져서 중요도 계산하는 데 쓰임)

    # get the activations of the last convolutional layer
    activations = model.get_activations(image).detach() # forward() 안의 backbone CNN의 마지막 conv layer 출력 (hook 등록된 곳)
                                            # backward graph에서 제외하기 위해 사용
    # weight the channels by corresponding gradients
    for i in range(activations.shape[1]): # 채널별로 importance weight 적용
        activations[:, i, :, :] *= pooled_gradients[i] # 각 feature map 채널에 해당하는 중요도 weight를 곱함
        
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze() # 모든 채널을 평균 내서 하나의 2D heatmap 생성

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = torch.where(heatmap > 0, heatmap, 0) # 음수 값 제거 (Grad-CAM 공식 방식: class에 기여하지 않는 부분 제거)

    # normalize the heatmap
    heatmap /= torch.max(heatmap) # 정규화: 시각화를 위해 [0, 1] 범위로 맞춤

    # Reshape & Convert Tensor to numpy
    heatmap = heatmap.squeeze()
    heatmap = heatmap.detach().cpu().numpy()


    # Resize Heatmap : 원래 이미지 사이즈로 복원
    heatmap = cv2.resize(heatmap, image_size) # OpenCV를 이용해 heatmap을 원래 크기(224*224)로 리사이즈
    # Convert to [0,255]
    heatmap = np.uint8(255 * heatmap) # 픽셀 값을 0~255 범위로 변환 -> 이미지로 저장 가능


    if figname is not None: # 파일 이름이 주어졌다면 show, save
        plt.figure(figsize=figsize);
        fig = plt.imshow(heatmap);
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig(figname, dpi=300, format='png', 
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None, 
            )

    return heatmap


'''
.backward()로부터 gradient 저장
get_activations_gradient()로 불러옴
평균 냄 -> 채널별 중요도 계산
그 weight를 feature map에 곱하고
평균 -> 2D heatmap
ReLU, 정규화
OpenCV로 resize하고 저장
'''