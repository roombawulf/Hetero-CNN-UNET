import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pyexpat import model
from tqdm import tqdm
from UNet import UNET
from Utils import getLoaders, checkAccuracy

# Hyperparameters
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 200
BATCH_SIZE = 32
PIN_MEMORY = True # Keep True unless you want slower training
LOAD_MODEL = False # Keep False unless loading a saved model                   

TRAIN_TEST_SPLIT = 0.2 # The split ratio for the training and test data (1 = 100% test set)
SHUFFLE_DATASET = False

dataType = 'normalRes2_3MAT'
resultType = 'equivalent'
setType = ''

STRAIN_DIR = 'Dataset/strain/' + dataType + '/' + resultType + '/' + setType + '/crop/*.png'
STRESS_DIR = 'Dataset/stress/' + dataType + '/' + resultType + '/' + setType + '/crop/*.png'
LABEL_DIR = 'Dataset/structure/' + dataType +  '/' + setType + '/image/*.png'

def saveModel(state, filename):
    print('=> Saving Model')
    filename = 'savedModels/' + filename
    torch.save(state, filename)

def writeLoss(loss, filename):
    filename = 'lossHistory/' + filename + '.txt'
    with open(filename, 'w') as file:
      for line in loss:
        file.write(f"{line}\n")

def train(loader, model, optimiser, lossFn):
    loop = tqdm(loader)
    lossList = []
    for idx, (image, label) in enumerate(loop):
        image = image.to(device=DEVICE)
        label = label.to(device=DEVICE)

        optimiser.zero_grad()

        predictions = model(image)
        loss = lossFn(predictions, label)

        lossList.append(loss.item())
        loss.backward()
        optimiser.step()

        loop.set_postfix(loss=loss.item())

    return sum(lossList)/len(lossList)



def main():
    trainLoader, testLoader = getLoaders(strainDir=STRAIN_DIR, 
                                         stressDir=STRESS_DIR, 
                                         labelDir=LABEL_DIR, 
                                         batchSize=BATCH_SIZE, 
                                         valSplit=TRAIN_TEST_SPLIT, 
                                         shuffle=SHUFFLE_DATASET)

    model = UNET().to(device=DEVICE)

    # Loss Function and Optimiser
    lossFn = nn.BCELoss() 
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-7) 

    avgLossList = []
    epochList = []

    if LOAD_MODEL:
        print('Loading Model...')
        model.load_state_dict(torch.load('finalModels/singleCompressiveHyper/min_4Layers_compressiveHyperElasticL1_EPOCH_100.pth.tar', map_location=torch.device(DEVICE)))

    if not LOAD_MODEL:
        print(f'Starting training using {DEVICE} as device... \n')
        for epoch in range(NUM_EPOCHS):
            print(f'Epoch No. = {epoch + 1}')
            avgLoss = train(trainLoader, model, optimiser, lossFn)
            avgLossList.append(avgLoss)
            epochList.append(epoch)

            plt.clf()
            plt.plot(epochList, avgLossList)
            plt.show()
            

            if (epoch + 1) % 100 == 0:
                checkpoint = model.state_dict()
                saveModel(state=checkpoint, filename=('maxCoupled_4Layers_highRes_EPOCH_' + str(epoch+1) + '.pth.tar'))
        writeLoss(loss=avgLossList, filename='maxCoupled_4Layers_highRes_Loss')

    checkAccuracy(loader=testLoader, model=model, device=DEVICE, epoch=10)

if __name__ == "__main__":
    main()
    plt.show()

