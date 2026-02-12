import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super(AudioCNN, self).__init__()
        
        def conv_block(in_c, out_c, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size, stride, padding),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
                nn.MaxPool1d(4)
            )
            
        self.block1 = conv_block(input_channels, 16, kernel_size=80, stride=4, padding=0) # M5-ish first layer usually large stride
        self.block2 = conv_block(16, 32)
        self.block3 = conv_block(32, 64)
        self.block4 = conv_block(64, 128)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (N, 1, L)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
