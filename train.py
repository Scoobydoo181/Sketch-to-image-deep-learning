import os
import sys
import gc

import time

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision import datasets, transforms
import torchvision

from componentEmbedding import ComponentEmbedding
from featureMapping import FeatureMapping
from imageSynthesis import ImageGenerator, CombinedDiscriminator


sys.stdout = open('output.txt', 'wt')
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

def loadImages(folderName, greyscale=False):
    transform = [transforms.ToTensor(), transforms.Lambda(lambda x: x.cuda().requires_grad_())]

    if greyscale: 
        transform = [transforms.Grayscale(), *transform]

    loader = datasets.ImageFolder(folderName, transforms.Compose(transform))
    batches = torch.utils.data.DataLoader(loader, 50)
    return batches

def multiGANLoss(genErrors, faceErrors): 
    '''Calculate the total error from the multi-scale discriminator'''
    MSELoss = nn.MSELoss()
    return sum([MSELoss(genError, faceError) for genError, faceError in zip(genErrors, faceErrors)])

def trainStageII(sketchBatches, faceBatches, models):
    #Training stage II: Holding the autoencoders constant, train the Feature Embedding and Image Synthesis modules
    L1Loss = nn.L1Loss()
    
    componentEmbedding, featureMapping, discriminator, generator = models
    
    for ((sketches, _), (faces, _)) in zip(sketchBatches, faceBatches):
        #Forward propagation
        with torch.no_grad():
            features = componentEmbedding(sketches)
        features = featureMapping(features)

        generatedFaces = generator(features)

        #Part A: Train discriminator, holding the generator and FM module constant
        genErrors = discriminator(generatedFaces, features)
        faceErrors = discriminator(faces, features)
        error = multiGANLoss(genErrors, faceErrors)
        # Teach the discriminator to maximize the difference between the real and generated faces
        (-error).backward(retain_graph=True)
        discriminator.weightUpdate()

        #Train generator + featureMapping, holding the discriminator constant
        #Part B: Train with GAN loss
        genErrors = discriminator(generatedFaces, features)
        faceErrors = discriminator(faces, features)
        error = multiGANLoss(genErrors, faceErrors)
        error.backward(retain_graph=True)
        featureMapping.weightUpdate()
        generator.weightUpdate()

        #Part C: Train with L1 loss on image synthesis
        error = L1Loss(generatedFaces, faces)
        error.backward(retain_graph=True)
        featureMapping.weightUpdate()
        generator.weightUpdate()
        del error

    return featureMapping, discriminator, generator, sketches, faces

path = os.getenv('path') or ''

if __name__ == '__main__':
    startTime = time.time()

    #Create modules
    print("Creating CE")
    componentEmbedding = ComponentEmbedding()
    #Training stage I: train autoencoders
    for _ in range(200):
        print("Loading sketches")
        sketchBatches = loadImages(f'{path}sketchData', greyscale=True)
        print("Training CE")
        componentEmbedding.train(sketchBatches)
    torch.save(componentEmbedding, f'{path}savedData/componentEmbedding.model')

    #Create modules
    print("Creating FM")
    featureMapping = FeatureMapping()
    print("Creating Generator")
    generator = ImageGenerator()
    print("Creating Discriminator")
    discriminator = CombinedDiscriminator()

    #Training stage II: train FM + GAN
    for _ in range(200):
        print("Loading sketches again")
        sketchBatches = loadImages(f'{path}sketchData', greyscale=True)

        print("Loading faces")
        faceBatches = loadImages(f'{path}faceData')

        print("Training FM + IS")
        models = [componentEmbedding, featureMapping, discriminator, generator]
        featureMapping, discriminator, generator, testSketches, testFaces = trainStageII(sketchBatches, faceBatches, models)

    torch.save(featureMapping, f'{path}savedData/featureMapping.model')
    torch.save(generator, f'{path}savedData/generator.model')
    torch.save(discriminator, f'{path}savedData/discriminator.model')

    
    #Generate output
    print("Writing output images")
    with torch.no_grad():
        generator.eval()
        featureMapping.eval()
        componentEmbedding.eval()

        outputImages = generator(featureMapping(componentEmbedding(testSketches)))
        loss = nn.MSELoss()
        print(loss(outputImages, testFaces))
        torchvision.utils.save_image(outputImages, f'{path}outputGrid.png')

    print("Total time to execute: ", time.time() - startTime)
