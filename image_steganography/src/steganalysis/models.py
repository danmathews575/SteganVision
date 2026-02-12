import torch
import torch.nn as nn
import torch.nn.functional as F

class SRNet(nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()
        
        # Layer 1: No residual, just Conv
        self.layer1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Layer 2-7: Type 1 Blocks (Unpooled)
        self.layer2 = Type1Block(64, 16)
        self.layer3 = Type1Block(16, 16)
        self.layer4 = Type1Block(16, 16)
        self.layer5 = Type1Block(16, 16)
        self.layer6 = Type1Block(16, 16)
        self.layer7 = Type1Block(16, 16)
        
        # Layer 8-11: Type 2 Blocks (Pooled)
        self.layer8 = Type2Block(16, 16)
        self.layer9 = Type2Block(16, 64)
        self.layer10 = Type2Block(64, 128)
        self.layer11 = Type2Block(128, 256)
        
        # Layer 12: Type 3 Block (Pooled, 3x3 -> GAP)
        self.layer12 = Type3Block(256, 512)
        
        # Fully Connected Layer
        self.fc = nn.Linear(512, 1) # Binary classification
        
    def forward(self, x):
        # Input standard: (N, 1, H, W)
        x = x.float()
        
        # Layer 1
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Layers 2-7
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        
        # Layers 8-11
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        
        # Layer 12
        x = self.layer12(x)
        
        # Global Average Pooling (handled in Type3Block mostly, or explicit here)
        # Type3Block output is already average pooled to 1x1 roughly or specific size
        # Actually SRNet paper uses global average pooling after layer 12
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        return x

class Type1Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Type1Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Residual connection
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        identity = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        # No ReLU after second BN in Type 1 usually? 
        # Standard ResNet uses ReLU after addition. SRNet paper: "Element-wise addition... then ReLU"
        # Wait, SRNet specific: 
        # Type 1: Conv-BN-ReLU -> Conv-BN -> Add -> (No ReLU here in some variances, but standard ResNet does)
        # However, Boroumand et al implementation:
        # PReLU is often used. We stick to ReLU for simplicity.
        
        out += identity
        out = F.relu(out) # Standard ResNet behavior
        return out

class Type2Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Type2Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual connection needs to account for pooling and channel change
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.residual_bn = nn.BatchNorm2d(out_channels) 
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.pool(out) # Average pooling
        
        identity = self.residual_conv(identity)
        identity = self.residual_bn(identity)
        
        out += identity
        out = F.relu(out)
        return out

class Type3Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Type3Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Global Average Pooling effectively if we want 1x1 at end
        # But usually Type3 is just another block ending with Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # No spatial residual needed if we go to 1x1, usually just linear projection if dims don't match
        # But let's follow standard Conv structure
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.global_pool(out)
        
        # For Type 3, strictly speaking in SRNet:
        # conv -> bn -> relu -> conv -> bn -> GAP
        # And no residual usually? Or residual is also GAP'd.
        # Let's simple sum if channels match, else ignore residual for the last block mostly.
        # We will assume simple FF here as it's the last feature extractor
        
        return out
