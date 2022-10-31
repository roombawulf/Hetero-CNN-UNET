from pickle import TRUE
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from UNet import UNET
from Utils import getLoaders, checkAccuracy

"""
Misc Paramaters
"""
# keep False unless loading a saved/trained model.
LOAD_MODEL = False
LOAD_MODEL_PATH = '<path_to_model.tar>'
# keep True unless you want slower training.
PIN_MEMORY = True
# checkpoint to save the model at (100 = Save model state every 100 epochs).
EPOCH_CHECKPOINT_SAVE = 100

"""
Hyperparamaters of U-NET Model
"""
# checks if CUDA GPU is available for training, otherwise uses CPU.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 

# training hyperparameters.
LEARNING_RATE = 3e-4
NUM_EPOCHS = 200
BATCH_SIZE = 32

# split ratio for training and test dataset (0.2 = 20% test set).
TRAIN_TEST_SPLIT = 0.2
SHUFFLE_DATASET = True         

"""
Dataset loading

Strain and label directories are only used but the stress-fields are also available.
Default root directory is './dataset/'
Use images in /crop/ for stress and strain.
"""
# 'elastoplastic' or 'hyperelastic'
materialType = 'elastoplastic'
# 'tension' or 'compression'
loadingCondition = 'tension'
# 'equivalent', 'maxPrincipal' or 'minPrincipal'
resultType = 'equivalent'
# leave blank when training. Use 'artif' or 'test' when loading a model to see sample predictions
setType = '' 

# define directories. default root is './dataset/'
STRAIN_DIR = 'dataset/strain/' + loadingCondition + '/' + materialType + '/' + resultType + '/' + setType + '/crop/*.png'
STRESS_DIR = 'dataset/stress/' + loadingCondition + '/' + materialType + '/' + resultType + '/' + setType + '/crop/*.png'
LABEL_DIR = 'dataset/structure/' + loadingCondition + '/' + materialType + '/' + setType + '/image/*.png'

# function to save the model. Saved into './savedModels/'.
def saveModel(state, filename):
    print('=> Saving Model')
    filename = 'savedModels/' + filename
    torch.save(state, filename)

# function to write average loss per epoch. Saved into './lossHistory/'.
def writeLoss(loss, filename):
    filename = 'lossHistory/' + filename + '.txt'
    with open(filename, 'w') as file:
      for line in loss:
        file.write(f"{line}\n")

# function to train the model.
def train(loader, model, optimiser, lossFn):

    # terminal training visualiser
    loop = tqdm(loader)

    # keep track of every loss value each epoch
    lossList = []

    for idx, (image, label) in enumerate(loop):
        image = image.to(device=DEVICE)
        label = label.to(device=DEVICE)
        
        # zero gradients
        optimiser.zero_grad()

        # model prediction and then calculate loss
        predictions = model(image)
        loss = lossFn(predictions, label)
        lossList.append(loss.item())

        # back-propogation and gradient descent
        loss.backward()
        optimiser.step()

        # handle terminal visualiser
        loop.set_postfix(loss=loss.item())

    return sum(lossList)/len(lossList)

# main function
def main():

    # handle split ratio if model is loaded. i.e Do not split since training is not performed
    trainSplit = TRAIN_TEST_SPLIT
    if LOAD_MODEL: trainSplit = 1

    # getLoaders() in Utils.py
    trainLoader, testLoader = getLoaders(strainDir=STRAIN_DIR, 
                                         stressDir=STRESS_DIR, 
                                         labelDir=LABEL_DIR, 
                                         batchSize=BATCH_SIZE, 
                                         setSplit=trainSplit, 
                                         shuffle=SHUFFLE_DATASET)

    # instantiate model from UNet.py
    model = UNET().to(device=DEVICE)

    # loss Function and gradient descent optimiser
    lossFn = nn.BCELoss() 
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-7) 
    
    # array for average loss and epochs
    avgLossList = []
    epochList = []

    # checks to see if model needs to be loaded (ignores training).
    if LOAD_MODEL:
        print('Loading Model...')
        model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=torch.device(DEVICE)))

    # if model doesn't need to be loaded, train.
    if not LOAD_MODEL:
        print(f'Starting training using {DEVICE} as device... \n')
        for epoch in range(NUM_EPOCHS):
            print(f'Epoch No. = {epoch + 1}')
            avgLoss = train(trainLoader, model, optimiser, lossFn)
            avgLossList.append(avgLoss)
            epochList.append(epoch)

            if (epoch + 1) % EPOCH_CHECKPOINT_SAVE == 0:
                checkpoint = model.state_dict()
                saveModel(state=checkpoint, filename=('maxCoupled_4Layers_highRes_EPOCH_' + str(epoch+1) + '.pth.tar'))
        writeLoss(loss=avgLossList, filename='maxCoupled_4Layers_highRes_Loss')

    checkAccuracy(loader=testLoader, model=model, device=DEVICE, epoch=10)

# start
if __name__ == "__main__":
    main()
    plt.show()

