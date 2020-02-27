import torch.nn as nn
from lib.nn.ops import Flatten


class Classifier(nn.Module):
    def __init__(self, num_labels, num_classes, nc=3, ndf=64):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.num_labels = num_labels

        main_layers = [
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16 * ndf, out_channels=16 * ndf, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16 * ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(16 * ndf, self.num_classes * self.num_labels, bias=False)
        ]
        self.main = nn.Sequential(*[x for x in main_layers if x is not None])

    def forward(self, x):
        y = self.main(x)
        return y.view((-1, self.num_classes, self.num_labels))
