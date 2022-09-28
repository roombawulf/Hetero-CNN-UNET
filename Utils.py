from cProfile import label
from turtle import left
from cv2 import sqrt
from matplotlib.ft2font import LOAD_VERTICAL_LAYOUT
import torch
import numpy as np
import matplotlib.pyplot as plt
from getDataset import strainFieldDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from mpl_toolkits.axes_grid1 import make_axes_locatable

def getLoaders(strainDir, stressDir, labelDir, batchSize, valSplit, shuffle, pin_memory=True):
    
    dataset = strainFieldDataset(strainDir, stressDir, labelDir)

    random_seed= 42

    datasetSize = len(dataset)
    indices = list(range(datasetSize))
    split = int(np.floor(valSplit * datasetSize))
    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    #train_sampler = SubsetRandomSampler(train_indices)
    #valid_sampler = SubsetRandomSampler(val_indices)

    train = DataLoader(dataset=dataset, batch_size=batchSize, sampler=train_indices)
    test = DataLoader(dataset=dataset, batch_size=batchSize, sampler=val_indices)

    return train, test

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

