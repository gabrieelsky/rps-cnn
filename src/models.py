import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=3, input_shape=(3, 100, 150)):
        super(BaselineCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=1, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=1, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self._to_linear = self._get_conv_output(input_shape)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def _get_conv_output(self, shape):
        batch_size = 1
        dummy_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.features(dummy_input)
        return int(torch.prod(torch.tensor(output_feat.size()[1:])))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MediumCNN(nn.Module):
    def __init__(self, num_classes=3, input_shape=(3, 100, 150), dropout_rate=0.5):
        super(MediumCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self._to_linear = self._get_conv_output(input_shape)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(128, num_classes)
        )

    def _get_conv_output(self, shape):
        batch_size = 1
        dummy_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.features(dummy_input)
        return int(torch.prod(torch.tensor(output_feat.size()[1:])))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MicroResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MicroResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual Layers (wider + two blocks per stage)
        self.layer1 = ResidualBlock(in_channels=32, out_channels=32, stride=1)
        self.layer1_b = ResidualBlock(in_channels=32, out_channels=32, stride=1)

        self.layer2 = ResidualBlock(in_channels=32, out_channels=64, stride=2)
        self.layer2_b = ResidualBlock(in_channels=64, out_channels=64, stride=1)

        self.layer3 = ResidualBlock(in_channels=64, out_channels=128, stride=2)
        self.layer3_b = ResidualBlock(in_channels=128, out_channels=128, stride=1)

        self.layer4 = ResidualBlock(in_channels=128, out_channels=256, stride=2)
        self.layer4_b = ResidualBlock(in_channels=256, out_channels=256, stride=1)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2_b(x)
        x = self.layer3(x)
        x = self.layer3_b(x)
        x = self.layer4(x)
        x = self.layer4_b(x)
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        return x