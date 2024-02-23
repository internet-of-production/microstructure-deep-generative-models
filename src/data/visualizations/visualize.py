# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 13:06:07 2021

@author: Schenk
"""
from __future__ import print_function
from pathlib import Path
import argparse
import os
from os.path import isfile, join
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
from skimage.morphology import area_closing
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from IPython.display import HTML
from skimage import measure, util
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import gaussian, threshold_otsu
from skimage.io import imread


def export_samples(Generator,configdata,Resultfolder,*args,num_samples=10,label=0,currentepoch=0,threshold=0):
    my_dpi=96
    
    if currentepoch==0:
        epoch=configdata["Training"]["num_epochs"]
    else:
        epoch=currentepoch
        
    nz=configdata["Generator"]["nz"]
    imagesize=configdata["get_data_loaders"]["image_size"]
    device=configdata["Discriminator"]["device"]
    
    Resultfolder=os.path.join(Resultfolder, f"Epoche_{epoch}",f"Label_{label}")
        
    os.makedirs(Resultfolder, exist_ok=True)
        
    for i in range(num_samples):
        
        export_flag=0
        while export_flag==0:
            noise = torch.randn(1, nz, device=device)
            noise = noise.to(configdata["Discriminator"]["device"])
            
            label_vec=torch.IntTensor([label])
            label_vec=label_vec.to(configdata["Discriminator"]["device"])
            
            plt.figure(figsize=(679/my_dpi, 679/my_dpi), dpi=my_dpi,frameon=False)
            fig1=plt.gcf()
            plt.axis("off")
            ax = plt.Axes(fig1, [0., 0., 1., 1.]) 
            image=Generator.forward(noise, label_vec).detach().cpu().numpy().reshape(imagesize,imagesize)
            inverted = util.invert(image)
            blurred = gaussian(inverted, sigma=.8)
            binary = blurred > threshold_otsu(blurred)
            image_porosity=np.mean(binary)
            
            if image_porosity>threshold:
                export_flag=1;
                
        if export_flag:
            
            if (configdata["get_data_loaders"]["image_type"]=="binary"):
                plt.imshow(image, interpolation='none', cmap=plt.cm.get_cmap('binary_r'))  
            else:
                plt.imshow(image, interpolation='none',cmap='gray')
                
            image_name=os.path.join(Resultfolder,f"Image_{i}.png")
            fig1.savefig(image_name,dpi=my_dpi, bbox_inches='tight',pad_inches = 0)
            plt.close()

