import torch
import torch.nn as nn
import torch.nn.functional as F

"""
U-NET Filter Paramaters

These should remain such that the output after convolution
remains the same dimension.
"""
# kernel/filter size - NxN format (N = integer)
KERNEL_SIZE = 3
# stride size - How many pixels the kernel/filter moves across image
STRIDE = 1 
# padding size - Padding pixels edge width around image
PADDING = 1 

"""
U-NET

This video is helpful: https://youtu.be/IHq1t7NxS8k

FEATURE_LAYERS can be changed both in length and values to build
a U-NET with different depths and features per layer.

Test line

The downConv and upConv classes are used as templates for building
respective regions of the U-Net. The UNET class loops through these
templated under __init__() method to build these sections with the use of the
FEATURE_LAYERS array, defining the depth and feautures of these sections.

The forward() method handles how data is passed through the model. Firstly,
downsampled and then upsampled to the final output. 'x' is the data variable handled.
The video referenced above is extremely helpful.
"""

# array for layers and features per layer.
FEATURE_LAYERS = [32,64,128,256]

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
    def __init__(self, in_channels=1, out_channels=1, features=FEATURE_LAYERS):
        super(UNET, self).__init__()

        # instantiate empty moduleList for downsampling 
        self.downs = nn.ModuleList()
        # instantiate empty moduleList for upsampling 
        self.ups = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.finalconv = nn.Conv2d((features[0] + in_channels), out_channels, KERNEL_SIZE, STRIDE, PADDING)
        self.finalconv2 = nn.Conv2d(out_channels, out_channels, 1, 1, 0)

        # downsampling 
        # builds the downsampling section of U-Net using the downConv class
        for feature in features:
            self.downs.append(downConv(in_channels, feature))
            in_channels = feature

        # upsampling
        # builds the upsampling section of U-Net using the upConv class
        for feature in reversed(features[:-1]):
            self.ups.append(upConv(in_channels, out_channels=feature))
            in_channels = feature      
            
    def forward(self, x):

        # instantiate skip connections
        skipConnections = []
        
        for down in self.downs:
            # add skip connection before every downsample
            skipConnections.append(x)
            x = down(x)
            x = self.pool(x)

        # reverse the skip connections array
        skipConnections = skipConnections[::-1]

        for index, up in enumerate(self.ups):
            x = self.upsample(x)
            x = torch.cat([skipConnections[index], x], dim=1)
            x = up(x)
        
        # final upsample and produce model output prediction
        x = self.upsample(x)
        x = torch.cat([skipConnections[-1], x], dim=1)
        x = torch.relu(self.finalconv(x))
        x = torch.sigmoid(self.finalconv2(x))

        return x

# # TEST CODE TO CHECK NETWORK WORKS
# model = UNET()
# tensor = torch.randn(1,1,256,256)
# x = model(tensor)
# print(f'Success! The output shape is: {x.shape}')