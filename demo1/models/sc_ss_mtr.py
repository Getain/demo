import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.conv(x)
        return x * attention

class SpectralAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SC_SS_MTr(nn.Module):
    def __init__(self, num_bands, num_classes, patch_size=3, depth=4, drop_rate=0.2):
        super(SC_SS_MTr, self).__init__()

        # 增强特征提取能力
        self.feature_reduction = nn.Sequential(
            ConvBlock(num_bands, 128, kernel_size=1, padding=0),
            nn.Dropout(drop_rate),
            ConvBlock(128, 64, kernel_size=1, padding=0)
        )

        # 增加网络深度和宽度
        self.spatial_blocks = nn.ModuleList([
            nn.Sequential(
                ConvBlock(64 if i == 0 else 128, 128, kernel_size=3, padding=1),
                nn.Dropout(drop_rate)
            ) for i in range(depth)
        ])

        # 注意力机制
        self.spectral_attention = SpectralAttention(128)
        self.spatial_attention = SpatialAttention(128)

        # 增强特征融合
        self.fusion = nn.Sequential(
            ConvBlock(128, 256, kernel_size=1, padding=0),
            nn.Dropout(drop_rate)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 增强分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature_reduction(x)
        for block in self.spatial_blocks:
            x = block(x)
        x = self.spectral_attention(x)
        x = self.spatial_attention(x)
        x = self.fusion(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)