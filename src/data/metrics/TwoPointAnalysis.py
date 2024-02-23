# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 11:13:19 2022

@author: Schenk
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt
import os
import tikzplotlib
import scipy.ndimage as sp
import scipy.fftpack as sfft
from tqdm import tqdm

def twopointclustering(directory,config,savedir,trainingdata=[]):
    
    pixelsize=config["get_data_loaders"]["real_size"]/config["get_data_loaders"]["image_size"]
    
    THRESHOLD_VALUE = 75
    
    inclusion_white = True

    image_names = os.listdir(directory)

    plots=[]
    
    
    for name in image_names:
        
        if name.split('.')[-1]=='png':
            splitted_name = name.split('.')[0]
            image_name = directory+"\\"+name

            print(image_name)
            originalImage = cv2.imread(image_name)
        
            # ~1/3 * sidelength
            pad_size = 150 
           
            delta = 150
            
            # make grayscale
            gray=originalImage
            gray = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)  
            if inclusion_white == True:
                gray[gray < THRESHOLD_VALUE] = 1
                gray[gray >= THRESHOLD_VALUE] = 0
            else:
                gray[gray < THRESHOLD_VALUE ] = 1
                gray[gray >= THRESHOLD_VALUE] = 0
    
        
            np_im_array = np.asarray(gray)
            
            padded_img = np.pad(np_im_array, ((pad_size,pad_size),(pad_size,pad_size)), 'constant')
                    
            # autocorrelation for two point statistics
            fft_image = sfft.fftn(padded_img)
            A = sfft.fftshift(sfft.ifftn(fft_image*np.conjugate(fft_image)))
            gray[gray >= 0] = 1
            padded_img = np.pad(np_im_array, ((pad_size,pad_size),(pad_size,pad_size)), 'constant')
            fft_image = sfft.fftn(padded_img)
            B = sfft.fftshift(sfft.ifftn(fft_image*np.conjugate(fft_image)))
            C = A/B
            C = C.real    
        
            result_func = []
            
            half_width = int(padded_img.shape[0]/2)
            half_length = int(padded_img.shape[1]/2) 
            
           
            inside = np.zeros((int(half_width*2),int(half_length*2)))
            
            for i in tqdm(range(2*delta)):
                for x in range(i):
                    for y in range(i):
                        radius = x**2 + y**2
                        if radius < i**2:
                            inside[half_width+x,half_length+y] = 1
                            inside[half_width+x,half_length-y] = 1
                            inside[half_width-x,half_length+y] = 1
                            inside[half_width-x,half_length-y] = 1
    
                extracted_image = np.extract(inside, C)
        
    
                result_func.append([i, np.sum(extracted_image), np.sum(inside), np.sum(extracted_image)/(np.sum(inside))])
        
            result_func = np.asarray(result_func)
            
            result_func_x = result_func[:,0]
            result_func_y = result_func[:,3]
            
            result_func_x = result_func_x[1::]
            result_func_y = result_func_y[1::]
            plots.append(result_func_y)   

    data=np.array(plots)
    import matplotlib.pyplot as plt
    mean=np.mean(data,0)
    upper=np.percentile(data,95,0)
    lower=np.percentile(data,5,0)
    x=list(range(1,300))
    x=np.array(x)*pixelsize #TO do: make conditional on image type
    plt.figure(figsize=(512/96, 512/96), dpi=96)
    fig1 = plt.gcf()
    plt.xlim(0,300*pixelsize)
    plt.fill_between(x,upper,lower,color='red',alpha=0.2)
    plt.plot(x,mean,color='red',label='GAN')
    
    if len(trainingdata)>0:
        mean_training=trainingdata['Mean']
        upper_training=trainingdata['Upper']
        lower_training=trainingdata['Lower']
        
        plt.fill_between(x,upper_training,lower_training,color='blue',alpha=0.2)
        plt.plot(x,mean_training,color='blue',label='Training')
    
    plt.xlabel(r'Distance [$\mu$m]')
    plt.ylabel('Probability')
    plt.legend()
    import tikzplotlib
    name_tex=os.path.join(savedir, "2Point.tex")
    name_png=os.path.join(savedir, "2Point.png")
    tikzplotlib.save(name_tex)
    fig1.savefig(name_png)
    return(mean,upper,lower)
