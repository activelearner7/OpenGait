import torch
from .base import BaseLoss
import torch
import torch.nn as nn


class ReconsLoss(BaseLoss):
    def __init__(self):
        super(ReconsLoss, self).__init__()

    def forward(self, ypred, ytrue):
        """
            ypred: [n, 1, s, h, w]
            ytrue: [n, 1, s, h, w]
        """
        ypred = ypred.float()
        ytrue = ytrue.float()

        mse_loss = nn.MSELoss()

        # Compute the reconstruction loss
        loss = mse_loss(ypred, ytrue)

        # loss = - (labels * torch.log(logits + self.eps) +
        #           (1 - labels) * torch.log(1. - logits + self.eps))

        # n = loss.size(0)
        # loss = loss.view(n, -1)
        # mean_loss = loss.mean()
        
        self.info.update({
            'loss': loss.detach().item()
            })

        return loss.detach(), self.info


if __name__ == "__main__":
    # loss_func = ReconsLoss()
    # ipts = torch.randn(1, 1, 128, 64)
    # tags = (torch.randn(1, 1, 128, 64) > 0.).float()
    # loss = loss_func(ipts, tags)
    # print(loss)

    print("Just the main function, no use :)")
