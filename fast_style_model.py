import torch
import torch.nn as nn

"""
The ConvLayer class defines a standard convolutional block commonly used in neural networks.
 - It bundles together a convolutional operation (nn.Conv2d) with optional padding (either ReflectionPad2d or ZeroPad2d) and an optional normalization layer (nn.InstanceNorm2d).
 - It's a flexible building block for processing image features.
"""
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer=True, reflect=False):
        super(ConvLayer, self).__init__()
        padding_size = kernel_size // 2
        pad_layer = nn.ReflectionPad2d(padding_size) if reflect else nn.ZeroPad2d(padding_size)
        self.conv = nn.Sequential(
            pad_layer,
            nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if norm_layer else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

"""
The ResidualBlock class implements a core component of many deep neural networks, particularly effective in deep image-processing models like the TransformerNet.
"""
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvLayer(channels, channels, kernel_size=3, stride=1, norm_layer=True, reflect=True),
            nn.ReLU(),
            ConvLayer(channels, channels, kernel_size=3, stride=1, norm_layer=True, reflect=True)
        )

    def forward(self, x):
        return x + self.block(x)
