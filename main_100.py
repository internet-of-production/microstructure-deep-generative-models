from __future__ import print_function
from pathlib import Path
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pandas as pd
from openpyxl import load_workbook

# Set random seed for reproducibility 
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Import local dependencies
from src.data.load_data import setup_filestructure
from src.data.load_data import get_data_loaders
from src.data.load_data import read_modeldata
from src.models.cgan import Generator
from src.models.cgan import Discriminator
from src.models.cgan import weights_init
from src.models.cgan import train
from src.data.visualizations.visualize import export_samples
from src.data.metrics.analyze_metrics import analyze_samples
from src.data.metrics.analyze_metrics import analyze_training_data
from src.data.metrics.TwoPointAnalysis import twopointclustering
from src.data.metrics.Analyze2Point import Analyze2Point
from src.data.ModelToCSV import ModelToCSV

CONFIG = {
    # Parameters for src.data.load_data.get_data_loadears()
    "get_data_loaders": {
        "dataroot": "100",
        "Analysis_File": "dummytobeoverwritten",
        'image_type' : 'binary',
        # Number of GPUs available. Use 0 for CPU mode.
        "ngpu" : 1,
        "nc" : 1,
        "modelname":"GAN_Initial",
        "resultdir":"dummytobeoverwritten",
        "resultpath": "dummytobeoverwritten",
        "batch_size": 20, 
        "augmentation": "on", 
        "image_size":512, #the image size needs to correspond to the chosen architecture
        "real_size":100, #insert the corresponding size of the image in microns
        "num_classes":1, #will be overwritten, once the label dic contains more than one label
        "workers":1
    },
    "Discriminator": {
        # Size of feature maps in discriminator
        "ndf" : 12, 
        "device" : "cuda",
         #Learning Rate
        "lr" : 0.00055,
        'gamma':0.985 
    },
    "Generator": {
        # Size of feature maps in generator
        "ngf" :16, 
        # Size of z latent vector 
        "nz" : 40, 
        #Size of embedding
        'embed_size':20,
        #Learning Rate
        "lr" : 0.00055,
        'gamma':0.985 
    },
    "Training": {
        "classify":"on",
        "num_epochs": 500,
        "beta1" : 0.5
    }
}
CONFIG["get_data_loaders"]["Analysis_File"]=os.path.join(CONFIG["get_data_loaders"]["dataroot"], 'Image_Descriptors.xlsx') 
CONFIG["get_data_loaders"]["resultdir"]=CONFIG["get_data_loaders"]["modelname"]
CONFIG["get_data_loaders"]["modelname"]=os.path.join(CONFIG["get_data_loaders"]["dataroot"], CONFIG["get_data_loaders"]["modelname"]+'.pt')
CONFIG["get_data_loaders"]["resultpath"]=os.path.join(CONFIG["get_data_loaders"]["dataroot"]+'_Results', CONFIG["get_data_loaders"]["resultdir"])

CONFIG["Discriminator"]["device"]: torch.device("cuda:0" if (torch.cuda.is_available() and CONFIG["get_data_loaders"]["ngpu"] > 0) else "cpu")

Metric = { 
    "Shape Descriptors": "Shape_Factor",
    "Num_images":500, #defines how many images are analyzed for every label if analysis is performed
    "Image Type":"Micrographx50",
    #select metrics to analyse, if "off", then parameter is skipped during analysis. Further parameters need to be added here
    "Skimage_Parameters":{
        "perimeter":"on",
        "area":"on",
        "max_feret":"on",
        "min_feret":"on",
        "orientation":"on",
        "porosity":"on",
        "solidity":"on",
        "curvature_pos":"on",
        "curvature_neg":"on"
    }
}

if CONFIG["Training"]["classify"]=="on":
    Label_Dic = { 
        "0": "0_Fine",
        "1": "1_Fine",
        "2": "2_Fine",
        "3": "3_Middle",
        "4": "4_Middle",
        "5": "5_Middle",
        "6": "6_Coarse",
        "7": "7_Coarse",
        "8": "8_Coarse"
    }
else:
    Label_Dic = { 
        "0": "Images"
    }
CONFIG["get_data_loaders"]["num_classes"]=len(Label_Dic)
        
if __name__ == "__main__":

    # Train GAN
    print(f"Using {CONFIG['Discriminator']['device']}")
    
    dataloader = get_data_loaders(CONFIG)
    dataloader_test=get_data_loaders(CONFIG,mode=1)
    
    targetloc=setup_filestructure(CONFIG)
    CONFIG["get_data_loaders"]["resultpath"]=targetloc
    
    #override configdata if trained model is used
    # Create the generator
    
    netG = Generator(CONFIG,len(dataloader.dataset.classes)).to(CONFIG["Discriminator"]["device"])

    # Handle multi-gpu if desired
    if (CONFIG["Discriminator"]["device"] == 'cuda') and (CONFIG["get_data_loaders"]["ngpu"] > 1):
        netG = nn.DataParallel(netG, list(range(CONFIG["get_data_loaders"]["ngpu"])))
        
    # Create the Discriminator
    netD = Discriminator(CONFIG,len(dataloader.dataset.classes)).to(CONFIG["Discriminator"]["device"])

    # Handle multi-gpu if desired
    if ((CONFIG["Discriminator"]["device"] == 'cuda') and (CONFIG["get_data_loaders"]["ngpu"] > 1)):
        netD = nn.DataParallel(netD, list(range(CONFIG["get_data_loaders"]["ngpu"])))   
        
    # Apply the weights_init function to randomly initialize all weights
    netD.apply(weights_init)
    # Apply the weights_init function to randomly initialize all weights
    netG.apply(weights_init)
    
   
    #Start Training of Generator and Discriminator
    epoch_count=0
        #epochstep defines, how many epochs are trained until the model is saved and the analysis is performed
    epochstep=100
        
    while epoch_count<(CONFIG["Training"]["num_epochs"]):
        
        #Load Training Data
        if CONFIG["get_data_loaders"]["augmentation"]=="on":
            
            del dataloader
            print('Create new dataloader as part of data augmentation...')
            
            dataloader = get_data_loaders(CONFIG,mode=0,noise=0.0)
                        
        #if reset, then the saved data of the model is deleted
        reset=0      
        # Perform Training    
        epoch_count,netG,netD=train(netG,netD,dataloader,dataloader_test,CONFIG,Label_Dic,epochstep=epochstep,reset=reset)
        
        #Export of Generator Images and diagrams of different metrics
        for i in range(0,(len(Label_Dic))):
            
            print(f'Export images for label {i}...')
            export_samples(netG,CONFIG,targetloc,label=i,currentepoch=epoch_count,num_samples=20)
                                 
            #Images are analyzed after every 10th step, may be changed    
            if (CONFIG["get_data_loaders"]["image_type"]=="binary"):        
                if ((epoch_count%10==0)&(CONFIG["Training"]["classify"]=="on")):
                    
                    print ("Analyzing image data using predefined metrics...")
                    Analyze2Point(CONFIG,i,Label_Dic,targetloc,epoch_count)
                    analyze_samples(netG,CONFIG,Metric,targetloc,Label_Dic,label=i,currentepoch=epoch_count)
                                       
    ModelToCSV(CONFIG)
