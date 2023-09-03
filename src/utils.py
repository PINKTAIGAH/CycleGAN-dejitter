import numpy as np
import torch
import config
from torchvision.utils import save_image, make_grid
from skimage import filters

def save_examples(gen, val_loader, epoch, folder):
    """
    Save examples of output from generator as png images at a specified folder

    Parameters
    ----------
    gen: torch.nn.Module instance
        A generator neural network that will output image (or data required to 
        generate an output image)

    val_loader: torch.utils.Data.DataLoader instance
        A dataloader containing dataset that will be input in the generator
    
    epoch: int
        Epoch at which example is being taken

    folder: string
        Directory where output image will be saved
    """
    # Unpack jittered (x) and ground truth (y) images from dataloader and send to device
    x, y, _ = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        # Generate unshifted image using the GAN's generator network
        y_fake = gen(x)
        # Remove normalisation
        y_fake = y_fake * 0.5 + 0.5 
        # Save png of jittered, ground truth and generated unjitted image respectivly
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
    gen.train()

def save_examples_concatinated(gen, val_loader, epoch, folder, save_sobel_score=False):
    """
    Save examples of output from generator as png images at a specified folder
    As opposed to saving images seperatly, images will be concatinated and outputted
    as a single image to increase ease to visually compare outputs from GAN
    Can also output the sobel score of the generated images

    Parameters
    ----------
    gen: torch.nn.Module instance
        A generator neural network that will output image (or data required to 
        generate an output image)

    val_loader: torch.utils.Data.DataLoader instance
        A dataloader containing dataset that will be input in the generator
    
    epoch: int
        Epoch at which example is being taken

    folder: string
        Directory where output image will be saved

    filter: torch.utils.Data.Dataset instance
        Class constining method required to unshift image using generator's 
        outputted flowmap

    save_sobel_score: bool
        Computes and returns the sobel score of the images generated
    """
    # Unpack jittered (x) and ground truth (y) images from dataloader and send to device
    x, y, _ = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        # Generate unshifted image using the GAN's generator network
        y_fake = gen(y)
        # Remove normalisation
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        y_fake = y_fake * 0.5 + 0.5  
        # Concatinate all output images in Batch axis in order to have size (3, 1, H, W)
        output = torch.cat([y, y_fake], dim=0)
        # Make image grid containing all desired output images
        image_grid = make_grid(output)
        save_image(image_grid, folder + f"/output_{epoch}.png")

        if save_sobel_score:
            y_sobel = filters.sobel(y[:, :].numpy())
            y_fake_sobel = filters.sobel(y_fake[:, :].numpy())
            
            # Append value of L1 distance to list
            return np.abs(y_sobel.sum() - y_fake_sobel.sum())
            
    gen.train()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Save dictionary of parameters of model and optimiser to specidied directory 
    in order to be loaded at a later time.

    Parameters
    ----------
    model: torch.nn.Module instance
        Neural network model to be saved

    optimiser: torch.optim instance
        Optimiser of model to be saved

    filename: string, optional
        Directory where model and optimiser will be saved
    """
    print("=> Saving checkpoint")
    # Dictionary constaining model and optimiser state parameters
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Load previously saved model and optimisers by assingning saved dictionaries 
    containing state parameters to inputted model and optimiser.

    Parameters
    ----------
    checkpoint_file: string
        Directory of file containing state dictionaries of previously saved model
        and optimiser

    model: torch.nn.Module instance
        Neural network model where state dictionary will be loaded 

    optimiser: torch.optim instance
        Optimiser of model where state dictionary will be loaded 

    lr: torch.TensorFloat
        Value of learning rate that is currently being used to train model
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    # Load saved state dictionaries 
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Assign current learning rate to the optimiser
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def _findMin(tensor):
    """
    Find minimum value of each batch of image tensor of shape (B, ..., H, W) and
    return tensor of same shape containing minimum value at each element for each
    batch

    Parameters
    ----------
    tensor: torch.FloatTensor
       Input tensor  

    Returns
    -------
    minTensor: torch.FloatTensor
        Tensor containing the minimum value of an image at each batch.
        Returns tensor of same shape as input

    Note
    ----
    Function assumes square images, hence H == W
    Function assumes only one colour channel
    """
    # Size of image
    N = tensor.shape[-1]
    # Reshape tensor to (B, H*W) and find minimum value along axis 1
    ###Change .view parametes if want to allow for multichannel images###
    minVals, _ = tensor.view(-1, N*N).min(axis=1) 
    minTensor = torch.ones_like(tensor)
    # Iterate over batches
    for i in range(minVals.shape[0]):
        # Assing minimum pixel value to each element of the image
        minTensor[i] = minTensor[i]*minVals[i]
    return minTensor

def _findMax(tensor):
    """
    Find maximum value of each batch of image tensor of shape (B, ..., H, W) and
    return tensor of same shape containing maximum value at each element for each
    batch

    Parameters
    ----------
    tensor: torch.FloatTensor
       Input tensor  

    Returns
    -------
    maxTensor: torch.FloatTensor
        Tensor containing the maximum value of an image at each batch.
        Returns tensor of same shape as input

    Note
    ----
    Function assumes square images, hence H == W
    Function assumes only one colour channel
    """
    # Size of image
    N = tensor.shape[-1]
    # Reshape tensor to (B, H*W) and find minimum value along axis 1
    ###Change .view parametes if want to allow for multichannel images###
    maxVals, _ = tensor.view(-1, N*N).max(axis=1)
    maxTensor = torch.ones_like(tensor)
    # Iterate over batches
    for i in range(maxVals.shape[0]):
        # Assing minimum pixel value to each element of the image
        maxTensor[i] = maxTensor[i]*maxVals[i]
    return maxTensor

def normaliseTensor(input):
    """
    Normalise image tensor that contains batches. Shape of input is (B, ..., H, W)

    Parameters
    ----------
    input: torch.FloatTensor
        Image tensor to be normalised
    
    Returns
    -------
    output: torch.FloatTensor
        Normalised image tensor

    Note
    ----
    Function assumes square images, hence H == W
    Function assumes only one colour channel
    """
    return (input-_findMin(input))/(_findMax(input) - _findMin(input))

def adjustArray(array):
    """
    Normalise ndArray 

    Parameters
    ----------
    input: ndArray
        array to be normalised
    
    Returns
    -------
    output: ndArray
        Normalised array
    """
    return (array) / (array.max() - array.min())

def write_out_value(val, filename, new_line=False):
    """
    Write out value to specified file. If new_line parameter is True, function
    will write out a new line character after having written out the val parameter

    Parameters
    ----------
    val: float or string
        Value to be written into file

    filename: string
        Directory of file where val will be written into

    new_line: bool, optional
        If True, function will write out a new line character after having written
        out the specified val parameter
    """
    with torch.no_grad():
        f = open(filename, "a+")
        f.write(' ' + str(val) + ',')
        # Add new line character to file if specified
        if new_line:
            f.write('\n')
        f.close()
    return

def write_out_titles(titles, filename,):
    """
    Write out strings in a list to a file and once finished add a new line character.
    Designed to write titles of a data file at the start of a file.

    Parameters
    ----------
    titles: list of strings 
        List containing titles to be wrtten into file

    filename: string
        Directory of file where val will be written into
    """
    with torch.no_grad():
        # Iterate over titles in list
        for idx, title in enumerate(titles):
            # Do not add new line if not final element in list
            if not len(titles)-1 == idx:
                write_out_value(title, filename, new_line=False)
            # Add new line if final element in list
            else:
                write_out_value(title, filename, new_line=True)

    def findSobelScore(y, y_fake):
        """
        Insert comments here
        """
            

def test():
    titles = ["gen_loss", "disc_loss", "random_string"]
    write_out_titles(titles, "../raw_data/test.txt")
    write_out_value(4.653, "../raw_data/test.txt", new_line=False)
    write_out_value(5.7657, "../raw_data/test.txt", new_line=False)
    write_out_value("hello", "../raw_data/test.txt", new_line=True)

if __name__ == "__main__":
    test()
