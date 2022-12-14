'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei
'''
import torch.nn as nn

__all__ = ['AlexNet_Mirror']

class AlexNet_Mirror(nn.Module):

    def __init__(self, num_classes=2):
        super(AlexNet_Mirror, self).__init__()
        self.features = nn.Sequential( 			#416 256
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), #104  128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          #52 128
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),	#26 64
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),	#26 32
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),	#26 32
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #13 8
        )
        self.avgpool = nn.AvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        y = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, y


def Alexnet_mirror(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet_Mirror(**kwargs)
    return model

