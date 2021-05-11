'''This module contains a generative adversarial network that generates photorealistic sketches from the 
per-pixel feature vectors output by the featureMapping module. It is trained The discriminator is trained 
on a mixture of true images and the output of the generator, which is trained by generating the error from the 
descriminator'''
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from buildingBlocks import ConvBlock, ResnetBlock, ConvTransBlock, FullyConnected, makeOptimizer

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

class ImageGenerator(nn.Module):
    def __init__(self):
        super(ImageGenerator, self).__init__()
        #Input size: Nx32x256x256

        self.conv1 = ConvBlock(inputSize=(32, 256, 256), outputSize=(56, 256, 256))
        self.conv2 = ConvBlock(inputSize=(56, 256, 256), outputSize=(112, 128, 128))
        self.conv3 = ConvBlock(inputSize=(112, 128, 128), outputSize=(224, 64, 64))
        self.conv4 = ConvBlock(inputSize=(224, 64, 64), outputSize=(448, 32, 32))

        self.resBlock1 = ResnetBlock(32)
        self.resBlock2 = ResnetBlock(32)
        self.resBlock3 = ResnetBlock(32)
        self.resBlock4 = ResnetBlock(32)
        self.resBlock5 = ResnetBlock(32)
        self.resBlock6 = ResnetBlock(32)
        self.resBlock7 = ResnetBlock(32)
        self.resBlock8 = ResnetBlock(32)
        self.resBlock9 = ResnetBlock(32)

        self.deconv1 = ConvTransBlock(inputSize=(448, 32, 32), outputSize=(224, 64, 64))
        self.deconv2 = ConvTransBlock(inputSize=(224, 64, 64), outputSize=(112, 128, 128))
        self.deconv3 = ConvTransBlock(inputSize=(112, 128, 128), outputSize=(224, 256, 256))
        self.deconv4 = ConvTransBlock(inputSize=(224, 256, 256), outputSize=(3, 256, 256))

        self.optim = makeOptimizer(self)

    def forward(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        x = self.resBlock4(self.resBlock3(self.resBlock2(self.resBlock1(x))))
        x = self.resBlock9(self.resBlock8(self.resBlock7(self.resBlock6(self.resBlock5(x)))))
        x = self.deconv4(self.deconv3(self.deconv2(self.deconv1(x))))
        return x

    def weightUpdate(self):
        self.optim.step()
        self.optim.zero_grad()

class DisUnit(nn.Module):
    def __init__(self, size):
        super(DisUnit, self).__init__()

        self.conv1 = ConvBlock(inputSize=(35, size, size), outputSize=(64, size/2, size/2))
        self.conv2 = ConvBlock(inputSize=(64, size/2, size/2), outputSize=(128, size/4, size/4))
        self.conv3 = ConvBlock(inputSize=(128, size/4, size/4), outputSize=(256, size/8, size/8))
        self.conv4 = ConvBlock(inputSize=(256, size/8, size/8), outputSize=(512, size/8, size/8))
        self.conv5 = ConvBlock(inputSize=(512, size/8, size/8), outputSize=(512, size/8, size/8))
        

    def forward(self, x):
        out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        return out 

class ImageDiscriminator(nn.Module):
    def __init__(self, depth):
        super(ImageDiscriminator, self).__init__()
        self.depth = depth

        dim = 256 
        #K = in/out
        if depth == 2:
            self.pool1 = nn.AvgPool2d(2)
            dim = 128 
        elif depth == 3:
            self.pool1 = nn.AvgPool2d(2)
            self.pool2 = nn.AvgPool2d(2)
            dim = 64

        self.disUnit = DisUnit(dim)

    def forward(self, x):
        if self.depth == 2:
            x = self.pool1(x)
        elif self.depth == 3:
            x = self.pool2(self.pool1(x))

        x = self.disUnit(x)
        return x

class CombinedDiscriminator:
    def __init__(self):
        super(CombinedDiscriminator, self).__init__()
        self.discriminators = [
            ImageDiscriminator(1), 
            ImageDiscriminator(2), 
            ImageDiscriminator(3)
        ]

        self.optimizers = [makeOptimizer(discriminator) for discriminator in self.discriminators]

    def weightUpdate(self):
        for optim in self.optimizers:
            optim.step()
            optim.zero_grad()

    def __call__(self, images, features):
        x = torch.cat((images, features), dim=1) #Nx35x256x256
        return [discriminator(x) for discriminator in self.discriminators]
        



