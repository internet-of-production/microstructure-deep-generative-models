# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:16:40 2023

@author: Schenk
"""
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

def ModelToCSV(configdata):
    subfolder=configdata["get_data_loaders"]["resultpath"] 
    my_file=Path(configdata["get_data_loaders"]["modelname"])
    if my_file.is_file():
        checkpoint=torch.load(configdata["get_data_loaders"]["modelname"])
        
        path = os.path.normpath(configdata["get_data_loaders"]["modelname"])
        name=os.path.splitext(path.split(os.sep)[-1])[0]
        
        G_losses=checkpoint['G_Losses']
        D_losses=checkpoint['D_Losses']
        D_Test=checkpoint['D_test']
        filename=os.path.join(*path.split(os.sep)[0:-1],name+'.txt')
        f=open(filename,'w')
        gamma_G=checkpoint['ExpLR_G']['gamma']
        f.write("gamma_G:"+str(gamma_G)+"\n")
        gamma_D=checkpoint['ExpLR_D']['gamma']
        f.write("gamma_D:"+str(gamma_D)+"\n")
        lr_G=checkpoint['ExpLR_G']['base_lrs']
        f.write("lr_G:"+str(lr_G)+"\n")
        lr_D=checkpoint['ExpLR_D']['base_lrs']
        f.write("lr_D:"+str(lr_D)+"\n")
        batchsize=checkpoint['Config']['get_data_loaders']['batch_size']
        f.write("batchsize:"+str(batchsize)+"\n")
        ngf=checkpoint['Config']['Generator']['ngf']
        f.write("ngf:"+str(ngf)+"\n")
        nz=checkpoint['Config']['Generator']['nz']
        f.write("nz:"+str(nz)+"\n")
        ndf=checkpoint['Config']['Discriminator']['ndf']
        f.write("ndf:"+str(ndf)+"\n")
        f.close()
    
    G_losses_list=[]
    D_losses_list=[]
    D_Test_list=[]
    #learning_rate_G_list=[]
    #learning_rate_D_list=[]
    for i in range(0,len(D_losses)):
        G_losses_list.append(G_losses[i].cpu().detach().numpy())
        D_losses_list.append(D_losses[i].cpu().detach().numpy())
        #D_Test_list.append(D_Test[i])
        #learning_rate_G_list.append(learning_rate_G[i].cpu().detach().numpy())
        #learning_rate_D_list.append(learning_rate_D[i].cpu().detach().numpy())
    
    #for i in range(0,len(D_Test)):
    #    D_Test_list.append(D_Test[i].cpu().detach().numpy())
        
    x=np.arange(1, len(G_losses_list)+1)
    x_2=np.arange(1, len(D_Test)+1)
    df = pd.DataFrame(index=x, columns={'G_Losses','D_Losses'})
    df2 = pd.DataFrame(index=x_2, columns={'Dtest'})
    df.at[:,'G_Losses'] = G_losses_list
    df.at[:,'D_Losses'] = D_losses_list
    df2.at[:,'D_Test'] = D_Test
    #df.at[:,'D_test'] = D_Test_list
    #df.at[:,'ExpLR_G'] = learning_rate_G
    #df.at[:,'ExpLR_D'] = learning_rate_D
    #Create excel file and write results
    excelfile=os.path.join(subfolder, "ModelData.xlsx")
    writer = pd.ExcelWriter(excelfile, engine='openpyxl')
    df.to_excel(
        writer,
        sheet_name=os.path.split(subfolder)[-1],
        startrow=0,
        header=True,
        index=False
    )
    df2.to_excel(
        writer,
        sheet_name=os.path.split(subfolder)[-1]+'_TestData',
        startrow=0,
        header=True,
        index=False
    )
    worksheet = writer.sheets[os.path.split(subfolder)[-1]]
    (max_row, max_col) = df.shape
    column_settings = [{'header': column} for column in df.columns]
    writer.save()
        
    #return(G_losses,D_losses,D_Test,learning_rate_G,learning_rate_D)