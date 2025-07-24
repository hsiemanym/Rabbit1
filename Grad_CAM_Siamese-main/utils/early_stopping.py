import torch

class EarlyStopping(): # validation 성능을 관찰하면서 '과적합을 피하기 위해 조기에 학습을 종료할지' 결정하는 클래스
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=3, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving (성능 개선 없이 기다릴 수 있는 epoch 수)
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement (개선으로 간주할 최소 변화량)
        """
        # 만약 validation 성능이 min_data만큼도 개선되지 않는 상태가 patience 횟수만큼 지속되면 학습 중단 신호를 보냄

        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss        
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return ( True )
            else:
                return ( False )

# 이 코드는 기본적으로 loss를 기준으로 동작하지만, AUC처럼 높을수록 좋은 지표를 사용할 경우엔 반대로 작동하도록 주의해야 함