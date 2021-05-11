''' This module contains the autoencoder network that is trained to learn the mapping from sketch 
to feature vector back to sketch. Once trained, the encoder half is extracted and used with the featureMapping 
network and the imageSynthesis network to transform sketch to image'''
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from buildingBlocks import ConvBlock, ResnetBlock, ConvTransBlock, FullyConnected, makeOptimizer

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

class EncodeModel(nn.Module):
    def __init__(self, size):
        super(EncodeModel, self).__init__()
        #1*size*size -> 32*size*size
        self.conv1 = ConvBlock(inputSize=(1, size, size), outputSize=(32, size/2, size/2))
        self.resBlock1 = ResnetBlock(size/2)

        self.conv2 = ConvBlock(inputSize=(32, size/2, size/2), outputSize=(64, size/4, size/4))
        self.resBlock2 = ResnetBlock(size/4)

        self.conv3 = ConvBlock(inputSize=(64, size/4, size/4), outputSize=(128, size/8, size/8))
        self.resBlock3 = ResnetBlock(size/8)

        self.conv4 = ConvBlock(inputSize=(128, size/8, size/8), outputSize=(256, size/16, size/16))
        self.resBlock4 = ResnetBlock(size/16)

        self.conv5 = ConvBlock(inputSize=(256, size/16, size/16), outputSize=(512, size/32, size/32))
        self.resBlock5 = ResnetBlock(size/32)

        #Nx512xsize/32xsize/32 -> Nx512
        self.fullyConnected = FullyConnected(inputSize=(512, size/32, size/32), outputSize=512)

    def forward(self, x):
        x = self.resBlock1(self.conv1(x))
        x = self.resBlock2(self.conv2(x))
        x = self.resBlock3(self.conv3(x))
        x = self.resBlock4(self.conv4(x))
        x = self.resBlock5(self.conv5(x))
        x = self.fullyConnected(x)
        return x

class DecodeModel(nn.Module):
    def __init__(self, size):
        super(DecodeModel, self).__init__()
        #Nx512 -> Nx512xsize/32xsize/32
        self.fullyConnected = FullyConnected(inputSize=512, outputSize=(512, size/32, size/32))

        self.resBlock1 = ResnetBlock(size/32)
        self.conv1 = ConvTransBlock(inputSize=(512, size/32, size/32), outputSize=(256, size/16, size/16))

        self.resBlock2 = ResnetBlock(size/16)
        self.conv2 = ConvTransBlock(inputSize=(256, size/16, size/16), outputSize=(128, size/8, size/8))

        self.resBlock3 = ResnetBlock(size/8)
        self.conv3 = ConvTransBlock(inputSize=(128, size/8, size/8), outputSize=(64, size/4, size/4))

        self.resBlock4 = ResnetBlock(size/4)
        self.conv4 = ConvTransBlock(inputSize=(64, size/4, size/4), outputSize=(32, size/2, size/2))
        
        self.resBlock5 = ResnetBlock(size/2)
        self.conv5 = ConvTransBlock(inputSize=(32, size/2, size/2), outputSize=(32, size, size))

        self.resBlock6 = ResnetBlock(size)
        self.conv6 = ConvBlock(inputSize=(32, size, size), outputSize=(1, size,size))

    def forward(self, x):
        x = self.fullyConnected(x)
        x = self.conv1(self.resBlock1(x))
        x = self.conv2(self.resBlock2(x))
        x = self.conv3(self.resBlock3(x))
        x = self.conv4(self.resBlock4(x))
        x = self.conv5(self.resBlock5(x))
        x = self.conv6(self.resBlock6(x))
        return x

class AutoEncoder(nn.Module):
    def __init__(self, size):
        super(AutoEncoder, self).__init__()
        self.encode = EncodeModel(size)
        self.decode = DecodeModel(size)

    def forward(self, x):
        return self.decode(self.encode(x))

class ComponentEmbedding():
    def __init__(self):
        super(ComponentEmbedding, self).__init__()

        #Autoencoders for each facial feature:
        self.encoders = [
            AutoEncoder(64), #left eye
            AutoEncoder(64), #right eye
            AutoEncoder(84), #nose
            AutoEncoder(96), #mouth
            AutoEncoder(256) #rest of face
        ]

    def __splitImages(self, images):
        leftEyeInput = images[:, :, 90:154, 126:190]
        rightEyeInput = images[:, :, 90:154, 64:128]
        noseInput = images[:, :, 103:187, 86:170]
        mouthInput = images[:, :, 143:239, 80:176]
        
        faceInput = images.clone()
        faceInput[:, :, 90:154, 126:190] = 0
        faceInput[:, :, 90:154, 64:128] = 0
        faceInput[:, :, 103:187, 86:170] = 0
        faceInput[:, :, 143:239, 80:176] = 0
        
        return leftEyeInput, rightEyeInput, noseInput, mouthInput, faceInput

    #Publically accessable methods:
    def __call__(self, sketches):
        inputs = self.__splitImages(sketches)
        return [encoder.encode(_input) for encoder, _input in zip(self.encoders, inputs)]

    def train(self, sketchBatches):
        #Create optimizer for each autoencoder
        optimizers = (makeOptimizer(encoder) for encoder in self.encoders)

        loss = torch.nn.MSELoss()

        for sketches, _ in iter(sketchBatches):
            #Split batch of sketches into inputs for each autoencoder
            inputs = self.__splitImages(sketches)

            #Train each autoencoder
            for encoder, _input, optim in zip(self.encoders, inputs, optimizers):
                optim.zero_grad()
                output = encoder(_input)
                loss(output, _input).backward(retain_graph=True)
                optim.step()

    def eval(self):
        for encoder in self.encoders:
            encoder.eval()