from torch.utils.data import Dataset
import torch
import config
import utils
import numpy as np
from time import time
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from torchvision.transforms import Pad
import torch.nn.functional as F

class ImageGenerator(Dataset):
    """
    A class used to generate white noise images, generate shift flow maps and 
    to shift images according to a given flow map.

    Parameters
    ----------
    imageHight: int
        Hight of image. (Currently class assumes square images, hence imageHight
        also represents width of image)

    correlationLength: float
        Represents correlation length of individual instances of jitter in an image.
        Corresponds to standard deviation of gaussian envelopes in shift generation.

    paddingWidth: int
        The number of pixels that will be added as padding on each edges of the image.

    maxJitter: float
        The maximum (and minimum) value of pixel shift. 

    Atributes
    ---------
    psf: torch.FloatTensor
        A 2D point spread function with equal hight and width dimention as the 
        imageHight parameter

    ftPsf: torch.ConplexFloatTensor
        The fourier transform of the psf parameter

    pad: torchvision.transforms.transforms.Pad instances
        The Pad instance initialised with the paddingWidth parameter

    identityFlowMap: torch.FloatTensor
        Flow map where each vector represents the position of it's corresponding 
        pixel in the flow map vector space. Tensor shape is (H, W, 2) 
    """

    def __init__(self, imageHeight, correlationLength, paddingWidth, maxJitter, randomSigma=False):
        
        self.sigma = config.SIGMA
        self.randomSigma = randomSigma
        self.psf = self._generatePSF()
        self.ftPsf = torch.fft.fft2(self.psf)
        self.imageHight = imageHeight
        self.correlationLength = correlationLength
        self.pad = Pad(paddingWidth)
        self.maxJitter = maxJitter

        # Using affine grid calculate identity flow map using identity matrix
        identifyMatrix = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        self.identityFlowMap = torch.squeeze(F.affine_grid(identifyMatrix,
                                             [1, 1, self.imageHight,
                                              self.imageHight]), 0) 

    def _normalise(self, x):
        """
        Normalise a ndArray

        Parameters
        ----------
        x: ndArray
            Input array containing dataset to be normalised

        Returns
        -------
        output: ndArray
            Normalised array
        """

        if np.sum(x) == 0:
            raise Exception("Divided by zero. Attempted to normalise a zero tensor")

        return x/np.sum(x**2)
    
    def _generatePSF(self,):
        """
        Generate a image tensor of a point spread function with size of defined 
        in config file (excluding padding)
    
        Returns
        -------
        pointFunction: FloatTensor 
            2D image tensor containing a point spreaffunction
        """
        if self.randomSigma:
            self.sigma = np.random.uniform(0.1, 20)

        # Generate a point spread finction by applying a gaussian blur to a point function
        pointFunction = np.zeros((config.NOISE_SIZE, config.NOISE_SIZE))
        pointFunction[config.NOISE_SIZE//2, config.NOISE_SIZE//2] = 1
        psf = self._normalise(gaussian(pointFunction, self.sigma))
        return torch.from_numpy(psf).type(torch.float32) 

    def _generateEnvelopeCenters(self):
        """
        Return array randomised containing centers for every gaussian envelope 
        in an image row

        Returns
        -------
        envelopeCenters: ndArray
        """
        envelopeCentersDistance = np.random.normal(self.correlationLength*4.5,
                                                  self.correlationLength)
        envelopeCenters = np.arange(0, self.imageHight, envelopeCentersDistance)
        return envelopeCenters

    def generateGroundTruth(self, padImage=True):
        """
        Return an image of white noise convolved with a point spread funtion.
        Shape of image is (1, H, W). 
        If padImage parameter is set to True, zero padding will be added to the
        image according to the paddingWidth parameter.

        Parameters
        ----------
        padImage: bool, optional
            Allow padding of generated image

        Returns
        -------
        groundTruth: torch.FloatTensor
            Tensor containing white noise image.
        """

        whiteNoise = torch.randn(*self.ftPsf.shape)
        # Convolve white noise with psf using convolution theorem
        groundTruth = torch.fft.ifft2(self.ftPsf * torch.fft.fft2(whiteNoise))  
        groundTruth = torch.unsqueeze(groundTruth, 0)
        
        if not padImage:
            # Return image without padding
            return torch.real(groundTruth)

        return self.pad(torch.real(groundTruth))

    def envelope(self, x, x0=0.0, std=1.0):
        """
        Return array containing gaussian envelope.

        Parameters
        ----------
        x: ndArray
            input array

        x0: float, optional
            mean of gaussian envelope

        std: float, optional
            Standard deviation of gaussian envelope

        Returns
        -------
        output: ndArray
            array containing gaussian envelope
        """
        return np.exp(-(x-x0)**2/(2*std**2))
    
    def generateSignal(self, x, frequency):
        """
        Returns array containing a sinusoidal signal with a randomised phase      

        Parameters
        ----------
        x: ndArray
            input array

        frequency: float
            frequency of signal

        Returns
        -------
        output: ndArray
            array containing signal
        """
        phase = np.random.uniform(0, 2*np.pi)
        return np.sin(2*np.pi*frequency*x + phase)

    def generateShiftMatrix(self):
        """
        Generate a matrix containing 1D vectors corresponding to the horizontal
        shift of each pixel in an image. Units of each vector are pixels. Shape of
        matrix is (H, W)

        Shifts are calculated by apeturing multiple sinusoidal signals with gaussian 
        envelopes in every row of an image.

        Returns
        -------
        shiftMatrix: torch.FloatTensor
            Matrix containing a 1D vector corresponding to the horizontal shift 
            of each pixel of an image.
        """
        
        shiftMatrix = np.empty((self.imageHight, self.imageHight))

        # Iterate over image hight
        for i in range(self.imageHight):
            x = np.arange(self.imageHight)
            yFinal = np.zeros_like(x, dtype=np.float64)

            # Random frequency of message signal for each row of image
            frequency = int(np.random.uniform(10, 100))
            envelopeCenters = self._generateEnvelopeCenters()
            # Iterate over each envelope in a row
            for _, val in enumerate(envelopeCenters):
                # Random amplitude for each messenge signal
                amplitude = np.random.uniform(0.5, self.maxJitter)
                y = self.generateSignal(x, frequency)
                yEnvelope = self.envelope(x, val, self.correlationLength)
                yFinal += utils.adjustArray(y * yEnvelope)*amplitude*2
            # Assign ndArray containing shift vectors of each row to output matrix
            shiftMatrix[i] = yFinal
        return torch.from_numpy(shiftMatrix)

    def generateFlowMap(self,):
        """
        Generate a flow map corresponding to a shift matrix.

        Returns a flow map corresponding to a shift matrix, a flow map
        corresponding to the inverse of the shift matrix and the shift matrix 
        used to generate the flow maps

        Shape of flow map is (H, W, 2)

        Return
        ------
        flowMapShift: torch.FloatTensor
            Flow map corresponding to an image shift
        
        flowMapUnshift: torch.FloatTensor
            Flow map corresponding to the inverse of an image shift

        shiftMatrix: torch.FloatTensor
            Matrix containing a 1D vector corresponding to the horizontal shift 
            of each pixel of an image.
        """
        shiftMatrix = self.generateShiftMatrix()
        # Compute the unit length of the vector space in the identity flow map
        step = self.identityFlowMap[0, 1, 0] - self.identityFlowMap[0, 0, 0]   
        
        flowMapShift, flowMapUnshift = (torch.clone(self.identityFlowMap),
                                        torch.clone(self.identityFlowMap))
        # Compute the shift in the flow map vector space and add/subtract from 
        # identity flow map
        flowMapShift[:, :, 0] += shiftMatrix*step 
        flowMapUnshift[:, :, 0] -= shiftMatrix*step
        
        return flowMapShift, flowMapUnshift, shiftMatrix

    def shift(self, input, flowMap, isBatch=True):
        """
        Shift an image using optical flow according to the inputted flow map.
        Input tensor and flowmap can either be a 3D tensors of shape (C, H, W) and
        (H, W, 2) respectivly if isBatch is False or a 4D tensor of shape (B, C, H, W)
        and (B, H, W, 2) if isBatch is True

        Parameters
        ----------
        input: torch.FloatTensor
            Image to be shifted 

        flowMap: torch.FloatTensor
            Flow map to be used in the optical flow transform

        isBatch: bool
            If False, the input tensor and the flowmap will be reshaped according 
            to the input shapes required by torch.nn.functional.grid_sample

        Returns
        -------
        output: torch.FloatTensor
            Shifted image 
        """

        if not isBatch:
            input = torch.unsqueeze(input, 0)
            flowMap = torch.unsqueeze(flowMap, 0)

        assert len(input.shape) == 4 and len(flowMap.shape) == 4,\
                "Input image and flowMap must have shape 4"
        
        output =  F.grid_sample(input, flowMap, mode="bicubic", padding_mode="zeros",
                         align_corners=False)

        if not isBatch:
            # Resize the output according to the input shape
            return torch.squeeze(output, 0)

        return output

def test():

    filter = ImageGenerator(config.IMAGE_SIZE, config.CORRELATION_LENGTH,
                            config.PADDING_WIDTH, config.MAX_JITTER, True)

    t1 = time()
    groundTruth = filter.generateGroundTruth()
    flowMapShift, flowMapUnshift, shiftMap = filter.generateFlowMap()
    shifted = filter.shift(groundTruth, flowMapShift, isBatch=False)
    t2 = time()
    unshifted = filter.shift(shifted, flowMapUnshift, isBatch=False)
    t3 = time()

    x = np.arange(config.IMAGE_SIZE)
    fig, (ax1,ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.scatter(x, shiftMap[0])
    ax2.scatter(x, shiftMap[1])
    ax3.scatter(x, shiftMap[2])
    ax4.scatter(x, shiftMap[3])
    plt.show()

    fig, ((ax1,ax2),(ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(groundTruth[0], cmap="gray")
    ax2.imshow(shifted[0], cmap="gray")
    ax3.imshow(unshifted[0], cmap="gray")
    ax4.imshow(groundTruth[0] - unshifted[0], cmap="gray")
    plt.show()
    print(f"Time taken to generate ground truth and shift: {t2-t1} s")
    print(f"Time taken to unshft image: {t3-t2} s")

if __name__ == "__main__":
    test()
