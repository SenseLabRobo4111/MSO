import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticModel(nn.Module):
    def __init__(self):
        super(CriticModel, self).__init__()
        
        self.header = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.encoder1 = self.conv_block(32, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512, use_dropout=False)

        self.final = nn.Conv2d(512, 1, kernel_size=1, bias=False)

    def conv_block(self, in_channels, out_channels, use_dropout=True):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
        )


    def forward(self, x, y):
        x = self.header(torch.cat([x, y], dim=1))
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        out = self.final(enc4)
        out = out.view(out.size(0), -1)

        return out