# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:08:37 2022

@author: Schenk
"""

def SaveConfiguration(configdata):
    
    directory=configdata["get_data_loaders"]["modelname"]
    checkpoint=torch.load(directory)
    configlist=checkpoint['Config']
    
    f = open(directory+'\Setup.txt', 'w')
    
    for key in configlist:
        for subkey in configlist[key]:
            configlist[key][subkey]