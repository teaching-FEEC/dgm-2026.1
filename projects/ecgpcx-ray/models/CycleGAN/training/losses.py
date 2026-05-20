import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    LSGAN loss used in CycleGAN.
    """

    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward(self, prediction, target_is_real):

        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)

        return self.loss(prediction, target)


class CycleConsistencyLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(self, reconstructed, real):
        return self.loss(reconstructed, real)


class IdentityLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(self, same_domain_output, real):
        return self.loss(same_domain_output, real)