# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:45:32 2022

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
from src.data.metrics.TwoPointAnalysis import twopointclustering

def Analyze2Point(CONFIG,i,Label_Dic,targetloc,epoch_count):

    subfolder=os.path.join(targetloc, f"Epoche_{epoch_count}",f"Label_{i}")
                
    x=list(range(1,300))
    x=np.array(x)
    two_point_training=os.path.join(CONFIG["get_data_loaders"]["dataroot"],'TwoPoint.xlsx')
    
    #If training data does not exist yet, create excel file for training images
    if not os.path.isfile(two_point_training):
        
        #Loop over all classes
        for j in range(0,(len(Label_Dic))):
            
            #Get location of images to be analyzed. IMPORTANT: Images need to exist in that folder!
            training_subfolder=os.path.join(CONFIG["get_data_loaders"]["dataroot"],'Training',Label_Dic[str(j)])
            
            #Analyze all images in the subfolder
            data=twopointclustering(training_subfolder,CONFIG,training_subfolder)            
            data=np.array(data)
    
            #Split dataset in mean, upper and lower
            df = pd.DataFrame(index=x, columns={'Mean','Upper','Lower'})
            df.at[:,'Mean'] = data[0]
            df.at[:,'Upper'] = data[1]
            df.at[:,'Lower'] = data[2]
            
            #Get file location of result file
            excelfile=os.path.join(CONFIG["get_data_loaders"]["dataroot"], "TwoPoint.xlsx")
            
            #Write results to excel file
            if j>0:
                book = load_workbook(excelfile)
            writer = pd.ExcelWriter(excelfile, engine='openpyxl')
            if j>0:
                writer.book = book
            df.to_excel(
                writer,
                sheet_name=os.path.split(training_subfolder)[-1],
                startrow=0,
                header=True,
                index=False
            )
            worksheet = writer.sheets[os.path.split(training_subfolder)[-1]]   
            (max_row, max_col) = df.shape   
            column_settings = [{'header': column} for column in df.columns]   
            writer.save()
            
    else:
        
        #Read Training data from excel file
        training_data = pd.read_excel(two_point_training, sheet_name=Label_Dic[str(i)], header=0)   
        
        #Analyze generated images
        data=twopointclustering(subfolder,CONFIG,subfolder,training_data)
        data=np.array(data)
        
        #Split dataset in mean, upper and lower
        df = pd.DataFrame(index=x, columns={'Mean','Upper','Lower'})
        df.at[:,'Mean'] = data[0]
        df.at[:,'Upper'] = data[1]
        df.at[:,'Lower'] = data[2]
        
        #Create excel file and write results
        excelfile=os.path.join(subfolder, "TwoPoint.xlsx")
        writer = pd.ExcelWriter(excelfile, engine='openpyxl')
        df.to_excel(
            writer,
            sheet_name=os.path.split(subfolder)[-1],
            startrow=0,
            header=True,
            index=False
        )
        worksheet = writer.sheets[os.path.split(subfolder)[-1]]
        (max_row, max_col) = df.shape
        column_settings = [{'header': column} for column in df.columns]
        writer.save()