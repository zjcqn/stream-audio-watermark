import torch
from .modules.msstft import MultiScaleSTFTDiscriminator

class Adversary(torch.nn.module):
    def __init__(self, *args, **kwargs):
        super(Adversary, self).__init__()
        self.d = MultiScaleSTFTDiscriminator(*args, **kwargs)

    def forward(self, x):
        return self.d(x)
