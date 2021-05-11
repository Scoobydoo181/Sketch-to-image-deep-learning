import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from math import sqrt

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

def makeOptimizer(encoder):
    return torch.optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))

class ResnetBlock(nn.Module):
    def __init__(self, size):
        ''' The input is the image width and height'''
        super(ResnetBlock, self).__init__()

        size = int(size)

        self.layer1 = nn.Linear(size, size)
        self.activation1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(size, size)
        self.activation2 = nn.LeakyReLU()

    def forward(self, x):
        f_x = lambda y: self.layer2(self.activation1(self.layer1(y)))
        return checkpoint(lambda y: self.activation2(f_x(y) + y), x)

class ConvBlock(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(ConvBlock, self).__init__()
        nIn, xIn, yIn = list(map(int, inputSize))
        nOut, xOut, yOut = list(map(int, outputSize))

        # O = [(W−K+2P)/S]+1
        # W-O+1 = K if padding = 0 and stride = 1

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=(xIn-xOut+1, yIn-yOut+1))
        self.batchNorm = nn.BatchNorm2d(nOut, momentum=sqrt(0.1))
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        return checkpoint(lambda y: self.activation(self.batchNorm(self.conv(y))), x)

    def eval(self):
        

class ConvTransBlock(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(ConvTransBlock, self).__init__()
        nIn, xIn, yIn = list(map(int, inputSize))
        nOut, xOut, yOut = list(map(int, outputSize))

        # O = W-2P+K-1
        # K = O-W+1

        self.conv = nn.ConvTranspose2d(nIn, nOut, kernel_size=(xOut-xIn+1, yOut-yIn+1))
        self.batchNorm = nn.BatchNorm2d(nOut, momentum=sqrt(0.1))
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        return checkpoint(lambda y: self.activation(self.batchNorm(self.conv(y))), x)

class FullyConnected(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(FullyConnected, self).__init__()

        if type(inputSize) == tuple and type(outputSize) == int:
            nIn, xIn, yIn = list(map(int, inputSize))
            self.reduce = True
            self.linear = nn.Linear(nIn*xIn*yIn, outputSize)

        elif type(inputSize) == int and type(outputSize) == tuple:
            nOut, xOut, yOut = list(map(int, outputSize))
            self.reduce = False
            self.outputSize = list(map(int, outputSize))
            self.linear = nn.Linear(inputSize, nOut*xOut*yOut)

        else:
            raise Exception('Wrong input types')

        self.activation = nn.LeakyReLU()

    def forward(self, x):
        def inner(x):
            if self.reduce:
                x = x.reshape(x.shape[0], -1)
                out = self.linear(x)
            else:
                x = self.linear(x)
                out = x.reshape(x.shape[0], *self.outputSize)
            return out
        return checkpoint(lambda y: self.activation(inner(y)), x)
