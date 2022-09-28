import torch
import torch.nn as nn
import torch.nn.functional as F


KERNEL_SIZE = 3 # Kernel/Filter size - NxN format (N = integer)
STRIDE = 1 # Stride size - How many pixels the kernel/filter moves across image
PADDING = 1 # Padding size - Padding pixels edge width around image

class downConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, KERNEL_SIZE, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class upConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(int(in_channels*1.5), out_channels, KERNEL_SIZE, STRIDE, PADDING, bias=False),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32,64,128,256]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.finalconv = nn.Conv2d((features[0] + in_channels), out_channels, KERNEL_SIZE, STRIDE, PADDING)
        self.finalconv2 = nn.Conv2d(out_channels, out_channels, 1, 1, 0)

        # Downsampling
        for feature in features:
            self.downs.append(downConv(in_channels, feature))
            in_channels = feature

        # Upsampling
        for feature in reversed(features[:-1]):
            self.ups.append(upConv(in_channels, out_channels=feature))
            in_channels = feature      
            
    def forward(self, x):
        skipConnections = []
        
        for down in self.downs:
            skipConnections.append(x)
            x = down(x)
            x = self.pool(x)

        skipConnections = skipConnections[::-1]
        print(x.shape)
        for index, up in enumerate(self.ups):
            x = self.upsample(x)
            x = torch.cat([skipConnections[index], x], dim=1)
            x = up(x)
        
        x = self.upsample(x)
        x = torch.cat([skipConnections[-1], x], dim=1)
        x = torch.relu(self.finalconv(x))
        x = torch.sigmoid(self.finalconv2(x))

        return x

## TEST CODE TO CHECK NETWORK WORKS
# model = UNET()
# tensor = torch.randn(1,2,256,256)
# x = model(tensor)
# print(f'Success! The output shape is: {x.shape}')