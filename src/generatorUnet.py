import torch
import torch.nn as nn

class Block(nn.Module):
    """
    A torch.nn.Module instance containing a template for a convolutional block 
    used in the UNET network used in the pix2pix Generator

    Atributes
    ---------
    block: torch.nn.Sequential
        Object that will return the output of a convolutional block of the
        a PatchGAN discriminator (kernal size 4)
        
        Object is capable of being a convolutional or a transpose convolutional
        block depending on input parameters
        
        Activation function used will either be LeakyReLU or ReLU depending on 
        input parameters

    Parameters
    ----------
    inChannels: int
        Number of image channels in input image

    outChannels: int
        Number of image channels that output image will have

    down: bool, optional
        If True, block will preform a 2D convolution, else will preform a 
        transpose convolution

    act: string, optional
        If "relu", activation function applied to output will be ReLU.
        If "leaky", activation function applied to output will be LeakyReLU

    useDropout: bool, optional
        If True, Dropot will be applied to the output. This will zero random 
        elements of the output image.
        This has been proben to improve regularsation and is required by the 
        pix2pix paper
    """
    def __init__(self, inChannels, outChannels, down=True, act="relu", useDropout=False):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            # Operation to preformed on input if down parameter is True
            nn.Conv2d(
                inChannels, outChannels, kernel_size=4, stride=2, padding=1,
                bias=False, padding_mode="reflect",
            )
            if down
            # Operation to preformed on input if down parameter is False
            else nn.ConvTranspose2d(inChannels, outChannels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(outChannels),
            # Apply activation function depending on the act parameter
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.useDropout = useDropout
        # Probability of element to be zerod in Dropout is 0.5, as described 
        # in original paper
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        """
        Returns output of convolutional blocks of a UNET block when called

        Parameters
        ----------
        x: torch.FloatTensor
            Input tensor to be passed through convolutional block

        Returns
        -------
        output: torch.FloatTensor
            Tensor containing output image of the convolutional block
        """
        x = self.block(x)
        return self.dropout(x) if self.useDropout else x


class Generator(nn.Module):
    """
    A torch.nn.Module instance containing a UNET neural network used by the 
    pix2pix generator. This UNET is designed to autoencode image tensors of 
    size 256*256.

    Atributes
    ---------
    initialDown: torch.nn.Sequential
        Object that will return the output of the initial convolution block of a 
        UNET model. Does not apply batch normalisation

    down1 - down7: torch.nn.Sequential
        Objects that represent each step in the encoder section of the UNET

    bottleneck: torch.nn.Sequential
        Object that returns as output a 1*1 image tensor. 

    up1 - up7: torch.nn.Sequential
        Objects that represent each step in the decoder section of the UNET.
        Also contains skip connections with the output of it's corresponding 
        block from the encoder section as outlined in the original UNET paper

    finalUp: torch.nn.Sequential
        Object that outputs the final generated image tensor of the pix2pix UNET
        generator. 

        A Tanh activation function is applied to the output image tensor

    Parameters
    ----------
    inChannels: int
        Number of colour channels in discriminator input

    features: int
        A coefficient used to compute the number of channels generated at each 
        convolution clock of the UNET

    Notes
    -----
    While the UNET can take input image tensors of various sizes, it is designed
    to optimally generate tensor images of size 256*256
    """
    def __init__(self, inChannels=3, features=64):
        super().__init__()
        self.initialDown = nn.Sequential(
            nn.Conv2d(
                inChannels, features, kernel_size=4, stride=2, padding=1,
                padding_mode="reflect"), 
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(
            features, features * 2, down=True, act="leaky", useDropout=False
        )
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", useDropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", useDropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", useDropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", useDropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", useDropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU(),
        )
        self.up1 = Block(
            features * 8, features * 8, down=False, act="relu", useDropout=True
        )
        # Input features are doubled as input tensor is concatinated due to skip 
        # conneciton
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", useDropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", useDropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", useDropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", useDropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", useDropout=False
        )
        self.up7 = Block(
            features * 2 * 2, features, down=False, act="relu", useDropout=False
        )
        self.finalUp = nn.Sequential(
            nn.ConvTranspose2d(
                features * 2, inChannels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Returns output of UNET generator when called

        Parameters
        ----------
        x: torch.FloatTensor
            Input tensor to be passed through generator 

        Returns
        -------
        output: torch.FloatTensor
            Tensor containing output of pix2pix generator. Output size is 
            equivalent to the size of the input image.
        """
        # Encoder
        d1 = self.initialDown(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        bottleneck = self.bottleneck(d7)
        # Decoder
        up1 = self.up1(bottleneck)
        # To enforce skip connection, concatinate output of last block with 
        # output of corresponding block from endocer.
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.finalUp(torch.cat([up7, d1], 1))

def test():
    x = torch.randn((1, 1, 256, 256))
    model = Generator(inChannels=1, features=64)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
