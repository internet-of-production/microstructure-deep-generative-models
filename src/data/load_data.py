# Import third party dependencies
# from torch.utils.data import random_split
# from torch.utils.data import DataLoader
# from torchvision import datasets
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision import transforms
import os
from pathlib import Path
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
def setup_filestructure(input):   
        dataroot=input["get_data_loaders"]["dataroot"];
        image_size=input["get_data_loaders"]["image_size"];
        resultdir=input["get_data_loaders"]["resultdir"]
        nz=input["Generator"]["nz"];
        Folder=dataroot+"_Results"
        if os.path.isdir(Folder):
            pass
        else:
            os.mkdir(Folder)
        Resultfolder=Folder+"/"+str(resultdir)
        if os.path.isdir(Resultfolder):
            pass
        else:
            os.mkdir(Resultfolder)
        return Resultfolder

def get_data_loaders(input,mode=0,noise=0):
    
    if mode==0:
        subfolder='Training'
    else:
        subfolder='Test'   
        
    dataroot=os.path.join(input["get_data_loaders"]["dataroot"],subfolder)
    image_size=input["get_data_loaders"]["image_size"];
    batch_size=input["get_data_loaders"]["batch_size"];
    workers=input["get_data_loaders"]["workers"];
    # Create the dataset
    if input["get_data_loaders"]["augmentation"]=="on":
        dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.Grayscale(num_output_channels=1),
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomVerticalFlip(p=0.5),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                               AddGaussianNoise(0., noise),
                           ]))
    else:
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.Grayscale(num_output_channels=1),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                   ]))
        

    dataloader = torch.utils.data.DataLoader(dataset, batch_size,shuffle=False)
    
    return dataloader

def read_modeldata(input):
    my_file=Path(input["get_data_loaders"]["modelname"])
    if my_file.is_file():
        checkpoint=torch.load(input["get_data_loaders"]["modelname"])
        #override input data
        input["get_data_loaders"]["batch_size"]=checkpoint['Config']["get_data_loaders"]["batch_size"]
        input["get_data_loaders"]["image_size"]=checkpoint['Config']["get_data_loaders"]['image_size']
        input["Discriminator"]["ndf"]=checkpoint['Config']["Discriminator"]['ndf']
        input["Generator"]["ngf"]=checkpoint['Config']["Generator"]['ngf']
        input["Generator"]["nz"]=checkpoint['Config']["Generator"]['nz']
        input["Training"]["beta1"]=checkpoint['Config']["Training"]['beta1']
        input["Training"]["lr"]=checkpoint['Config']["Training"]['lr']
    return input