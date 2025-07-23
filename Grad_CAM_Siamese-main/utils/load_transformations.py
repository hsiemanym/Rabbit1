from torchvision import transforms
# torchvision.transforms를 사용하여 전처리

def load_transformations(params:dict=None)->(transforms,transforms):
  '''
    Image transformation for data loader

    Parameters
    ----------
    params: dictionary with parameters

    Returns
    -------
    Tuple with transforms
  '''
                # 여러 전처리를 순서대로 묶는 함수
  train_tfms = transforms.Compose( [ transforms.Resize( params['image']['size'] ), # 모든 이미지를 동일한 크기로 조정
                                    # 데이터 증강
                                    # transforms.RandomResizedCrop(224),
                                    # transforms.CenterCrop(224),
                                    # transforms.RandomCrop(48, padding=8, padding_mode='reflect'),                                  
                                    # transforms.ColorJitter(brightness = params['image']['brightness'], 
                                    #                       contrast   = params['image']['contrast'], 
                                    #                       saturation = params['image']['saturation'], 
                                    #                       hue        = params['image']['hue']),
                                    # transforms.RandomHorizontalFlip(p = params['image']['HorizontalFlip']),
                                    # transforms.RandomVerticalFlip(p = params['image']['VerticalFlip']),
                                    # transforms.RandomRotation(degrees = params['image']['RandomRotationDegrees']),
                                    transforms.ToTensor(), # PIL 이미지를 PyTorch 텐서로 변환
                                    transforms.Normalize(params['image']['normalization']['mean'], params['image']['normalization']['std']), # 픽셀 값을 평균/표준편차 기준으로 정규화
                                  ] )
  #
  # Testing set
  #
  # 데이터 증강 없이 3개만 사용하는 testing set
  test_tfms  = transforms.Compose([ transforms.Resize( params['image']['size'] ),
                                    transforms.ToTensor(),
                                    transforms.Normalize(params['image']['normalization']['mean'], params['image']['normalization']['std']),
                                  ] )
  
  return train_tfms, test_tfms

'''
기본적으로는 이미지는 정해진 크기 (예: 224*224)로 resize되고
ImageNet의 평균/표준편차를 이용해 정규화
augmentation증강은 현재 비활성
'''