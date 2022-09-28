from sklearn.compose import TransformedTargetRegressor
import torch
import glob
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as form
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from skimage import io
from torch.utils.data.sampler import SubsetRandomSampler

class strainFieldDataset(Dataset):
    def __init__(self, strainDir, stressDir, labelDir, transform = transforms.ToTensor()):

        self.strains = sorted(glob.glob(strainDir), key=len)
        self.stresses = sorted(glob.glob(stressDir), key=len)
        self.labels = sorted(glob.glob(labelDir), key=len)
        self.transform = transform

    def __getitem__(self, index):

        strain = self.strains[index]
        strain = Image.open(strain).convert('L')

        #stress = self.stresses[index]
        #stress = Image.open(stress).convert('L')

        label = self.labels[index]
        label = Image.open(label).convert('L')

        if self.transform:
            strain = self.transform(strain)
            #stress = self.transform(stress)
            label = self.transform(label)
        
        strainMean = torch.mean(strain)
        strainStd = torch.mean(strain)
        #strain = form.normalize(strain, mean=strainMean, std=strainStd)

        #stressMean = torch.mean(stress)
        #stressStd = torch.std(stress)
        #stress = form.normalize(stress, mean=stressMean, std=stressStd)

        #strainStress = torch.cat([strain, stress], dim=0)

        return strain, label
    
    def __len__(self):
        return len(self.labels)





