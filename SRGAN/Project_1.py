#Name: Sumegha Singhania, Kishore Reddy Pagidi
#Date: 09/22/2022
#Class name: CS7180 Advanced Perception
#Assignment1: Image Enhancement

import torch
from importlib.util import LazyLoader
from nis import maps
from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import math
from torch import nn
from torchvision.models.vgg import vgg16

# Prepare the training dataset:

# In the paper, this is done in the following way:
# - The HR images are cropped into 96x96 sub images
# - Get LR images from HR images, using a
# Gaussian filter and downsamlping by a factor r. 
# - Scale range of input LR images: [0,1], HR images: [-1,1]
# Dataset I have used: RELLISUR Dataset

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

# This is done to ensure that the crop size isn't bigger than the actual
# image, since we will be downsampling it as well.
def valid_crop(crop_size,scale_factor):
    valid_size = crop_size-(crop_size%scale_factor)
    return valid_size

def crop_hr_imgs(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor()
    ])

def get_lr_imgs(crop_size,scale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size//scale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

class TrainingDataset(Dataset):
    def __init__(self,dataset_path,crop_size,scale_factor):
        super(TrainingDataset,self).__init__()
        self.image_filenames = [join(dataset_path, x) for x in listdir(dataset_path) if is_image_file(x)]
        crop_size = valid_crop(crop_size,scale_factor)
        self.prep_hr_imgs = crop_hr_imgs(crop_size)
        self.prep_lr_imgs = get_lr_imgs(crop_size,scale_factor)

    def __getitem__(self,iter):
        hr_image = self.prep_hr_imgs(Image.open(self.image_filenames[iter]))
        lr_image = self.prep_lr_imgs(hr_image)
        return hr_image,lr_image

    def __len__(self):
        return len(self.image_filenames)


# Preparing the test dataset
class TestingDataset(Dataset):
    def __init__(self,dataset_path, scale_factor):
        super(TestingDataset,self).__init__()
        self.lr_imgs_path = dataset_path+'/Test/LLLR/'
        self.hr_imgs_path = dataset_path+'/Test/X2/'
        self.upscale_factor = scale_factor
        self.lr_files = [join(self.lr_imgs_path,x) for x in listdir(self.lr_imgs_path)]
        self.hr_files = [join(self.hr_imgs_path,x) for x in listdir(self.hr_imgs_path)]

    def __getitem__(self,iter):
        image_name = self.lr_files[iter].split('/')[-1]
        lr_img = Image.open(self.lr_files[iter])
        hr_img = Image.open(self.hr_files[iter])
        w,h = lr_img.size
        hr_upscale = Resize((self.scale_factor*h, self.scale_factor*w), Interpolation=Image.BICUBIC)
        hr_restored = hr_upscale(lr_img)
        return image_name, ToTensor()(lr_img), ToTensor()(hr_restored), ToTensor(hr_img)

    def __len__(self):
        return len(self.lr_files)

# Preaping the generator model
# In paper: 
# - B residual blocks with the following layout:
    # - 2 convolutional layers with 3x3 kernels and 64 feature maps
    # - Batch Normalization Layer
    # - Parametric ReLu activation function

# Creating the residual B block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

# Last block after residual block, upsamples the generated image
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

# Creating the generator
class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block = int(math.log(scale_factor,2))

        super(Generator,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        # tanh activation function, returns the image in range [0,1]
        return (torch.tanh(block8) + 1) / 2

# Creating the discriminator
# From the paper:
# Uses LeakyReLu activation (alpha = 0.2)
# Contains 8 convolutional layers with increasing number of 3x3 kernels
# increased by a factor of 2 from 64 to 512 kernels like VGG
# 512 feature maps followed by 2 dense layers and sigmoid activation
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


# Defining the Loss function
# From paper:
#It is a combination of content loss(image_loss here) which is based is dependednt on the MSE loss derived from the VGG loss, and the adversarial loss which is dependent on the perception loss.
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        return image_loss + 0.001 * adversarial_loss 


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)

    

