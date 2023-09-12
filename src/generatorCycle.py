import torch
import torch.nn as nn

class _ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, down=True, useAct=True, **kwargs):
        super().__init__()

        # Define generic convolutional and transpose convolutional block
        self.conv = nn.Sequential(
            nn.Conv2d(
                inChannels,
                outChannels,
                padding_mode="reflect",
                **kwargs,
            )
            if down else nn.ConvTranspose2d(
                inChannels,
                outChannels,
                **kwargs,
            ),
            nn.InstanceNorm2d(outChannels),
            # Only pass through activation function if useAct is True
            nn.ReLU(inplace=True) if useAct else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)

class _ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        
        # Define convolutional block of the residual blocks
        self.block = nn.Sequential(
            _ConvBlock(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            _ConvBlock(
                channels,
                channels,
                useAct=False,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

    def forward(self, x):
        # Return residual component of block
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, imageChannels, numFeatures=64, numResiduals=9):
        super().__init__()

        # Define initial blocks of generator
        self.initialBlock = nn.Sequential(
            nn.Conv2d(imageChannels,
                      64,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      padding_mode="reflect"
                      ),
            nn.ReLU(inplace=True),
        ) 

        # Define downscaling section of generator
        self.downBlocks = nn.ModuleList([
            _ConvBlock(numFeatures, numFeatures*2, kernel_size=3, stride=2, padding=1),
            _ConvBlock(numFeatures*2, numFeatures*4, kernel_size=3, stride=2, padding=1),
        ])

        # Define residual section of generator
        self.residualBlocks = nn.Sequential(
            *[_ResidualBlock(numFeatures*4) for _ in range(numResiduals)]
        )

        # Define upsampling section of generator
        self.upBlocks = nn.ModuleList([
            _ConvBlock(numFeatures*4,
                       numFeatures*2,
                       down=False,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       output_padding=1),
            _ConvBlock(numFeatures*2,
                       numFeatures,
                       down=False,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       output_padding=1),
        ])

        # Rescale image channels to input size (ie: convert output to RGB or grayscale)
        self.lastBlock = nn.Conv2d(
            numFeatures,
            imageChannels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        # Pass inputs through initial block
        x = self.initialBlock(x)
        # Pass through all blocks in downsampling section
        for layer in self.downBlocks:
            x = layer(x)
        # Pass through all residual blocks
        x = self.residualBlocks(x)
        # Pass through all blocks in upsampling section
        for layer in self.upBlocks:
            x = layer(x)
        # Pass through final layer and tanh activation function
        return torch.tanh(self.lastBlock(x))

def test():
    image_channel = 3
    image_size = 256
    x = torch.randn((5, image_channel, image_size, image_size))
    gen = Generator(image_channel, numFeatures=64, numResiduals=9)
    prediction = gen(x)
    print(prediction.shape)

if __name__ == "__main__":
    test()
