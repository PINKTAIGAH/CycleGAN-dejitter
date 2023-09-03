import torch 
import torch.nn as nn

class _Block(nn.Module):

    def __init__(self, inChannels, outChannels, stride,):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                inChannels,
                outChannels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode="reflect"
            ),
            nn.InstanceNorm2d(outChannels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):

    def __init__(self, inChannels=3, features=[64, 128, 256, 512]):
        super().__init__()
        
        # Define initial block of PatchGAN w/o instance normalisation
        self.initialBlock = nn.Sequential(
            nn.Conv2d(
                inChannels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
                ),
            nn.LeakyReLU(0.2),
        )

        discriminatorLayers = []
        # Set outChannels of first layer as inChannels for next conv layer
        inChannels = features[0] 
        # Loop over features as input channels to add internal layers of PatchGAN
        for feature in features[1:]:
            discriminatorLayers.append(
                _Block(inChannels,
                       feature, 
                       stride=1 if feature==features[-1] else 2
                )
            )
            inChannels = feature

        # Final layer of PatchGAN which outputs a single channel matrix of patch scores
        discriminatorLayers.append(
            nn.Conv2d(
                inChannels,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect"
            )
        )

        # Define discriminator object (w/o first conv layer)
        self.model = nn.Sequential(*discriminatorLayers)

    def forward(self, x):
        # Pass input through initial block
        x = self.initialBlock(x)
        # Pass through rest of PatchGAN & apply sigmoud to output discriminator scores
        return torch.sigmoid(self.model(x))

def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(inChannels=3)
    prediciton = model(x)
    print(prediciton.shape)

if __name__ == "__main__":
    test()

