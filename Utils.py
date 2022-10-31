import torch
import numpy as np
import matplotlib.pyplot as plt
from getDataset import strainFieldDataset
from torch.utils.data import DataLoader

"""
DataLoader class handling dataset object from custom class in getDataset.py

The dataset is shuffled (if enabled) and then appropriately split 
into training and test sets where the split is defined by the user from
the gloval variable TRAIN_TEST_SPLIT (in trainUNet.py).
"""
def getLoaders(strainDir, stressDir, labelDir, batchSize, setSplit, shuffle, pin_memory=True):
    
    # get dataset from custom class in getDataset.py
    dataset = strainFieldDataset(strainDir, stressDir, labelDir)

    # random seed is used for controllable randomness
    random_seed= 42

    # split dataset
    datasetSize = len(dataset)
    indices = list(range(datasetSize))
    split = int(np.floor(setSplit * datasetSize))

    # shuffle dataset
    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # create train and test set and then return them
    train = DataLoader(dataset=dataset, batch_size=batchSize, sampler=train_indices)
    test = DataLoader(dataset=dataset, batch_size=batchSize, sampler=val_indices)

    return train, test

"""
Check accuracy and plot

USERNOTE: This function is very messy/inefficient but does work 
(will attempt to update some time). Probably split accuracy and plots
into two seperate functions. The plots also have the tendency to break 
when the batchsize is set too small (< 4)... need to fix. 

After training, this function checks the accuracy of the model on the test set and 
then plots the last 3 predictions, ground truth and residual error via matplotlib.pyplot.
"""
def checkAccuracy(loader, model, device, epoch):
    numCorrect = 0
    numPixels = 0
    model.eval()

    structureCmap = 'summer'
    errorCmap = 'winter'
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)

            prediction = model(x)
        
        fig, ax = plt.subplots(len(prediction),3, figsize=(6,4))
        for i, row in enumerate(ax):
            plotList = []
            predStruct = prediction[i].cpu().numpy().squeeze()
            trueStruct = y[i].cpu().numpy().squeeze()
            error = np.sqrt((trueStruct - predStruct)**2)
            round = np.where(predStruct > 0.499, 1, 0)
            correct = (round==trueStruct)

            plotList.append(trueStruct)
            plotList.append(predStruct)
            plotList.append(error)

            print(f' True struct = {trueStruct[0][0]} || Pred Struct = {predStruct[0][0]}')
            print(f' Unique values = {len(np.unique(predStruct))}')
            print(f' Round = {round[0][0]} || Correct = {correct[0][0]} || Error Max = {error.max()} || Error Min = {error.min()}')
            print('-----')
            acc = np.round(((correct.sum()/trueStruct.size)*100),2)
            accString = 'ACCURACY = ' + str(acc) + '%'

            for n, col in enumerate(row):
                if n == 2:
                    map = col.imshow(plotList[n], cmap=errorCmap)
                    col.text(0, 280, accString)
                    col.axis('off')

                else:
                    col.imshow(plotList[n], cmap=structureCmap)
                    col.axis('off')

        ax[0][0].set_title('TRUE')
        ax[0][1].set_title('PREDICTION')
        ax[0][2].set_title('RSE')
        fig.subplots_adjust(right=0.9, left=0.4, bottom=0.045, top=0.95, wspace=0, hspace=0.14)
        cbarAx = fig.add_axes([0.92, 0.27, 0.01, 0.45])
        fig.colorbar(map, cax=cbarAx, label='RSE')

    model.train()

