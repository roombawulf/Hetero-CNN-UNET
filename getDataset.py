import glob
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

"""
Custom Class to Generate Dataset format

This class returns an object containing the strain-field images 
and corresponding structures (labels) to be directly used with 
PyTorch's DataLoader class. 

The DataLoader class will transform these into torch.tensors (used in Utils.py)
for use with PyTorch.
"""
class strainFieldDataset(Dataset):
    def __init__(self, strainDir, stressDir, labelDir, transform = transforms.ToTensor()):

        # load images and sort numerically.
        self.strains = sorted(glob.glob(strainDir), key=len)
        self.stresses = sorted(glob.glob(stressDir), key=len)
        self.labels = sorted(glob.glob(labelDir), key=len)
        self.transform = transform

    def __getitem__(self, index):

        strain = self.strains[index]
        # convert image into grayscale.
        strain = Image.open(strain).convert('L')

        label = self.labels[index]
        # convert image into grayscale.
        label = Image.open(label).convert('L')

        if self.transform:
            strain = self.transform(strain)
            label = self.transform(label)

        return strain, label
    
    def __len__(self):
        return len(self.labels)
