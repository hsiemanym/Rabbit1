import torch


class LRScheduler(): # PyTorch의 ReduceLROnPlateau를 감싼 wrapper
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    # 이 클래스는 validation loss 또는 validation AUC(이 코드)를 모니터링
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5, verbose = True):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience  = patience # 이 값이 일정 epoch 수(patience)동안 개선되지 않으면
        self.min_lr    = min_lr # min_lr보다 작아지지 않도록 보장하면서
        self.factor    = factor # 학습률을 factor 배수만큼 줄임 (예: 0.5면 절반으로 감소)
        self.verbose   = verbose
        # 모델의 성능이 정체될 때, 학습률을 조정해 학습을 계속 진행할 수 있도록 도와줌

#         self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = self.optimizer, 
#                                                                        steps     = self.patience, 
#                                                                        verbose   = self.verbose )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode      = 'min',
                patience  = self.patience,
                factor    = self.factor,
                min_lr    = self.min_lr,
                verbose   = self.verbose 
            )
        
    def __call__(self, val_loss):
        self.lr_scheduler.step( val_loss )