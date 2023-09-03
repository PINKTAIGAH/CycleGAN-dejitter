from skimage.filters import gaussian
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import transforms as transform
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
Hyper Parameters
"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Directory of files containing image datasets
TRAIN_DIR_HORSE = "/home/giorgio/CycleGAN_data/horse2zebra/horse2zebra/train_horse/"
TRAIN_DIR_ZEBRA = "/home/giorgio/CycleGAN_data/horse2zebra/horse2zebra/train_zebra/"
# TRAIN_DIR = "/home/brunicam/myscratch/p3_scratch/p06_images/train/"
VAL_DIR_HORSE = "/home/giorgio/CycleGAN_data/horse2zebra/horse2zebra/test_horses/"
VAL_DIR_ZEBRA = "/home/giorgio/CycleGAN_data/horse2zebra/horse2zebra/test_zebra/"
# VAL_DIR = "/home/brunicam/myscratch/p3_scratch/p06_images/val/"
SIEMENS_VAL_DIR = "/home/giorgio/Desktop/val_siemens/"

LEARNING_RATE = 2e-4
BATCH_SIZE = 1
SCHEDULAR_DECAY = 0.5
SCHEDULAR_STEP = 20                         # Step size of learning rate schedular
OPTIMISER_WEIGHTS = (0.5, 0.999)            # Beta parameters of Adam optimiser
NUM_WORKERS = 4
PADDING_WIDTH = 30
IMAGE_SIZE = 256 
NOISE_SIZE = IMAGE_SIZE - PADDING_WIDTH*2
CHANNELS_IMG = 3                            # Colour channels of input image tensors 
L1_LAMBDA = 100
LAMBDA_GP = 10
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0.0
NUM_RESIDUALS = 9                           # 9 if image 256p and 6 if image 128p
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = False

CHECKPOINT_DISC_H_LOAD = "../models/discH.pth.tar"
CHECKPOINT_DISC_Z_LOAD = "../models/discZ.pth.tar"
CHECKPOINT_GEN_H_LOAD = "../models/genH.pth.tar"
CHECKPOINT_GEN_Z_LOAD = "../models/genZ.pth.tar"

CHECKPOINT_DISC_H_SAVE = "../models/discH.pth.tar"
CHECKPOINT_DISC_Z_SAVE = "../models/discZ.pth.tar"
CHECKPOINT_GEN_H_SAVE = "../models/genH.pth.tar"
CHECKPOINT_GEN_Z_SAVE = "../models/genZ.pth.tar"

MODEL_LOSSES_FILE = "../raw_data/model_losses.txt"
MODEL_LOSSES_TITLES = ["epoch", "disc_loss", "gen_loss"]
TRAIN_IMAGE_FILE= "../evaluation/default"
EVALUATION_IMAGE_FILE = "../evaluation/metric"

CRITIC_SCORE_FILE = "../raw_data/critic_score.txt"
CRITIC_SCORE_TITLES = ["epoch", "disc_real", "disc_fake"]
# WRITER_REAL = SummaryWriter("/home/brunicam/myscratch/p3_scratch/runs/real")
# WRITER_FAKE = SummaryWriter("/home/brunicam/myscratch/p3_scratch/runs/fake")

"""
Jitter hyperparameters
"""
MAX_JITTER = 3
SIGMA = 20                                  # Standard deviation of gaussian kernal for PSF
CORRELATION_LENGTH = 10

# Evaluation hyperparameters
EVALUATION_EPOCHS = 100
EVALUATION_METRIC_FILE = "../raw_data/siemens_sigma.txt"

EVALUATION_IMAGE_FILE = "../evaluation/module/"

CHECKPOINT_DISC_LOAD = "../models/disc.p06_img.tar"
CHECKPOINT_GEN_LOAD = "../models/gen.p06_img.tar"

"""
Tensor Transformations
"""

transforms = transform.Compose([
    transform.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)],   # generalise for multi channel
        [0.5 for _ in range(CHANNELS_IMG)],
    ),
])

transformsFile = transform.Compose([
    transform.ToTensor(),
    transform.RandomCrop(IMAGE_SIZE),
    transform.Grayscale(),
])

transforms_concatinated = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        #A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0,5, 0,5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"}
)

"""
Hyperparameter overwriting for automatic bash scripts 
"""
