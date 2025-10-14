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
- It allows information to flow directly through the block (the x in x + self.block(x)), while simultaneously learning a transformation (self.block(x)) that refines the features.
- This helps prevent vanishing gradients.
- Allows for the construction of very deep networks that can learn complex mappings.
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

"""
The UpsampleConvLayer class is designed for increasing the spatial resolution of feature maps in a neural network.
- In a CNN, when an image (or the output of a previous layer) passes through a convolutional layer, the output is called a feature map.
- A feature map is a representation of the input image, highlighting the presence and strength of learned features at different spatial locations.
- The class first enlarges (increasing the spatial dimensions (Height and Width) of the feature map) the input feature map (upsampling), and then applies a convolution and normalization.
- Enlarging the input feature map happens through the interpolate function.
- Need to enlarge to reconstruct a high-resolution output image from smaller, abstract feature representations, effectively increasing the spatial detail from a compressed form back to a visible image.
- This specific sequence helps to expand the image resolution while also avoiding common visual distortions known as "checkerboard artifacts" that can occur if upsampling is handled purely by transposed convolutions.
"""
class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then applies a convolution. This is useful
    to avoid checkerboard artifacts that can arise from training purely
    with strided convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        padding_size = kernel_size // 2
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding_size),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        x_in = self.conv(x_in)
        return self.norm(x_in)

"""
The TransformerNet class defines the entire feed-forward neural network for fast neural style transfer.
- Takes an input image and transforms it into a stylized output with learned brushstrokes.
- Encoder (self.initial_layers): the feature extractor.
  - It takes the input image (3 color channels) and uses ConvLayers with stride=2 to progressively downsample the image's spatial dimensions while increasing the number of feature channels (e.g., from 3 to 32, then to 64, then to 128).
  - Each convolution learns to extract increasingly complex features from the image (edges, textures, shapes).
  - By the end of initial_layers, the network has a compact, high-dimensional representation of the image's content.

"""
class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.initial_layers = nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1),
            nn.ReLU(),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            ConvLayer(64, 128, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.deconv_layers = nn.Sequential(
            UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2),
            nn.ReLU(),
            UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
            nn.ReLU(),
            ConvLayer(32, 3, kernel_size=9, stride=1, norm_layer=False, reflect=True)
        )

    def forward(self, X):
        y = self.initial_layers(X)
        y = self.res_blocks(y)
        return self.deconv_layers(y)
