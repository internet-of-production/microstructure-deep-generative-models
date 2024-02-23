"""Module with classes to define / train Variational Autoencoder"""

# Import third party dependencies
from __future__ import print_function
#%matplotlib inline
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
from tqdm import tqdm

######################################################################
# Implementation

# custom weights initialization called on netG and netD
def weights_init(m):
    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train(Generator,Discriminator,dataloader,test_data,configdata,labeldic,epochstep=0,reset=0):

    
    torch.cuda.empty_cache()
    
    #Setup Optimizer
    discriminator_loss = nn.BCELoss()
    generator_loss= nn.MSELoss()
    
    # Create batch of latent vectors
    fixed_noise = torch.randn(configdata["get_data_loaders"]["batch_size"], configdata["Generator"]["nz"], 1, 1, device=configdata["Discriminator"]["device"])
    
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    
    # Setup Adam optimizers 
    optimizerD = optim.Adam(Discriminator.parameters(), lr=configdata["Discriminator"]["lr"], betas=(configdata["Training"]["beta1"], 0.999))
    ExpLRD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=configdata["Discriminator"]["gamma"])
    optimizerG = optim.Adam(Generator.parameters(), lr=configdata["Generator"]["lr"], betas=(configdata["Training"]["beta1"], 0.999))    
    ExpLRG = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=configdata["Generator"]["gamma"])
   
    #Initialize Datastorage
    epoch_list = []
    G_losses = []
    D_losses = []
    D_x_save = []
    D_test_save = []
    D_G_z1_save = []
    D_G_z2_save = []
    
    #Initialize variables
    iters = 0
    my_dpi=96
    start_epoch=0
    criterion = nn.BCELoss()
    device=configdata["Discriminator"]["device"]
    batchnum=int(np.floor(len(dataloader.dataset.targets)/configdata["get_data_loaders"]["batch_size"]))
    LabelTensor=torch.IntTensor(dataloader.dataset.targets[0:configdata["get_data_loaders"]["batch_size"]*batchnum]).view(batchnum,configdata["get_data_loaders"]["batch_size"])
    print("Search for existing model data...")
    
    #Searches for existing model data and loads latest progres
    my_file=Path(configdata["get_data_loaders"]["modelname"])
    
    #Read out existing model data and initialize models
    if my_file.is_file():
        
        checkpoint=torch.load(configdata["get_data_loaders"]["modelname"])
        Discriminator.load_state_dict(checkpoint['Discriminator'])
        
        if reset==0:
            
            Generator.load_state_dict(checkpoint['Generator'])
            
        Discriminator.train()
        Generator.train()
        
        optimizerD.load_state_dict(checkpoint['Optimizer_D'])
        ExpLRD.load_state_dict(checkpoint['ExpLR_D'])
        D_losses=checkpoint['D_Losses']
        D_losses[-1].backward()    
        optimizerD.load_state_dict(checkpoint['Optimizer_D'])
        optimizerD.step()
        optimizerD.zero_grad()

        if reset==0:
            
            optimizerG.load_state_dict(checkpoint['optimizer_G'])
            ExpLRG.load_state_dict(checkpoint['ExpLR_G'])
            G_losses=checkpoint['G_Losses']
            G_losses[-1].backward()    
            optimizerG.load_state_dict(checkpoint['optimizer_G'])
            optimizerG.step()
            optimizerG.zero_grad()
            
        Model_epoch=checkpoint['Epoch']+1
        iters=checkpoint['Iterations']
        
        if reset==0:
            G_losses=checkpoint['G_Losses']
        
        D_x_save = checkpoint['D_x']
        D_test_save = checkpoint['D_test']
        D_G_z1_save = checkpoint['D_G_z1']
        D_G_z2_save = checkpoint['D_G_z2']
        start_epoch+=Model_epoch
      
    #Check if training may start...
    if configdata["Training"]["num_epochs"]<=start_epoch:
        print("Increase number of epochs. The given number is below the number of already performed training epochs")
        
    else:
        print("Starting Training Loop...")
        
        # For each epoch
        if epochstep==0:
            targetepoch=configdata["Training"]["num_epochs"]
            print(f"Performing training until {targetepoch} epochs are reached...")
            
        else:
            targetepoch=start_epoch+epochstep 
            print(f"Stepwise training is selected. Performing training over {epochstep} epochs...")
            
        sal_lab=0
        
        for epoch in range(start_epoch,targetepoch):
            # For each batch in the dataloader
            i=0
            D_loss_list, G_loss_list = [], []
            
            #Set models in training mode
            Discriminator.train()
            Generator.train()
            
            for index, (real_images, labels) in enumerate(dataloader):

                real_images = real_images.to(device)
                
                shuffled_labels=labels
                
                for label_num in range(0,1,len(labels)):
                    
                    current_label=labels[label_num]
                    new_labels=list(range(0,len(labeldic)))
                    new_labels.remove(current_label)
                    new_label=random.choice(new_labels)
                    shuffled_labels[label_num]=new_label
                
                #Create array with image labels
                labels = labels.to(device)
                labels = labels.unsqueeze(1).long()
                
                #Create array with shuffled image labels
                shuffled_labels = shuffled_labels.to(device)
                shuffled_labels = shuffled_labels.unsqueeze(1).long()
                
                real_target = torch.ones(real_images.size(0), 1).to(device)
                fake_target = torch.zeros(real_images.size(0), 1).to(device)
                    
                output = Discriminator(real_images, labels)
                
                #Compute discriminator loss for real images    
                D_real_loss = discriminator_loss(output, real_target)
                D_x=output.mean().item()
                
                #Compute discriminator loss for real images with wrong labels
                output_shuffled = Discriminator(real_images, shuffled_labels)
                D_real_loss_shuffled = discriminator_loss(output_shuffled, fake_target)
                
                #Create latent vector
                noise_vector = torch.randn(real_images.size(0), configdata["Generator"]["nz"], device=device)  
                noise_vector = noise_vector.to(device)
                
                #Generate images
                generated_image = Generator(noise_vector, labels)
                output = Discriminator(generated_image.detach(), labels)
                
                #Compute discriminator loss for fake images 
                D_fake_loss = discriminator_loss(output, fake_target)
                D_G_z1=output.mean().item()
            
                #Commpute average loss
                D_total_loss = (D_real_loss+D_real_loss_shuffled + D_fake_loss) / 3
                D_loss_list.append(D_total_loss)
                D_losses.append(D_total_loss)
                
                #Optimize Discriminator
                optimizerD.zero_grad()
                D_total_loss.backward()
                optimizerD.step()
        
                # Train generator with real labels
                optimizerG.zero_grad()
                G_loss = generator_loss(Discriminator(generated_image, labels), real_target)
                
                G_loss_list.append(G_loss)
                G_losses.append(G_loss)
                
                #Optimize Generator
                G_loss.backward()
                optimizerG.step()
                
                iters+=1
                
                if i % 10 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f '
                          % (epoch, configdata["Training"]["num_epochs"], i, len(dataloader),
                             D_total_loss.item(), G_loss.item(), D_x, D_G_z1))
                i+=1
                
                #Save mean response
                D_x_save.append(D_x)
                D_G_z1_save.append(D_G_z1)

                
            Discriminator.eval()
            D_test=0
            d_iter=0
            
            for index, (test_images, test_labels) in enumerate(test_data):  
                
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)
                test_labels = test_labels.unsqueeze(1).long()
                output_test = Discriminator(test_images, test_labels)
                D_test+=output_test.mean().item()
                d_iter+=1
                
            D_test_save.append(D_test/d_iter)
            print ('\tLoss_D_test: %.4f '
                          % (D_test/d_iter))   
            
            Discriminator.train()
            
            ExpLRG.step()
            ExpLRD.step()
            
        #save the model for later usage
        torch.save(
            {'Config': configdata, 'Epoch': epoch, 'Iterations': iters, 'G_Losses': G_losses, 'D_Losses': D_losses,
             'Discriminator': Discriminator.state_dict(), 'Generator': Generator.state_dict(),
             'Optimizer_D': optimizerD.state_dict(), 'optimizer_G': optimizerG.state_dict(),
             'ExpLR_D': ExpLRD.state_dict(), 'ExpLR_G': ExpLRG.state_dict(),
             'D_x': D_x_save,'D_test': D_test_save,'D_G_z1': D_G_z1_save, 'D_G_z2': D_G_z2_save
             },
            configdata["get_data_loaders"]["modelname"])
        
        return targetepoch,Generator,Discriminator
######################################################################
# Generator

class Generator(nn.Module):
    def __init__(self,configdata,num_classes):
        super(Generator, self).__init__()
        self.ngpu = configdata["get_data_loaders"]["ngpu"]
        self.classify=configdata["Training"]["classify"]
        self.img_size=configdata["get_data_loaders"]["image_size"]
        self.ngf=configdata["Generator"]["ngf"]
        self.nz=configdata["Generator"]["nz"]
        self.nc=configdata["get_data_loaders"]["nc"]
        self.device=configdata["Discriminator"]["device"]
        if self.classify=="off":  
            print ("no classification implemented within Generator....")
            self.embed_size=0
        else:
            self.embed_size=configdata["Generator"]["embed_size"]
        
        if self.classify=="on":
            self.label_conditioned_generator = nn.Sequential(nn.Embedding(num_classes, self.embed_size),
                                                             nn.Linear(self.embed_size, 16))
        
        self.latent = nn.Sequential(nn.Linear(self.nz, 4*4*512),
                                    nn.LeakyReLU(0.2, inplace=True))
        
        self.main = nn.Sequential(

            nn.ConvTranspose2d( 514, self.ngf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 32),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 32, self.ngf * 8, 8, 4, 2, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 2, 8, 4, 2, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( self.ngf*2, self.nc, 8, 4, 2, bias=False),
            nn.Tanh()
        )

    def merge(self,input,label):
        label_p=torch.div(label, 3, rounding_mode='floor')
        label_p_output = self.label_conditioned_generator(label_p)
        label_p_output = label_p_output.view(-1, 1, 4, 4)
        label_d=torch.remainder(label,3)
        label_d_output = self.label_conditioned_generator(label_d)
        label_d_output = label_d_output.view(-1, 1, 4, 4)
        
        latent_output = self.latent(input)
        latent_output = latent_output.view(-1, 512,4,4)
        concat = torch.cat((latent_output, label_d_output, label_p_output), dim=1)
        return concat
    
    def forward(self, input,label):

        concat=self.merge(input,label)
        image = self.main(concat)
        return image
#########################################################################
# Discriminator Code

class Discriminator(nn.Module):
    def __init__(self,configdata,num_classes):
        super(Discriminator, self).__init__()
        self.ngpu = configdata["get_data_loaders"]["ngpu"]
        self.classify=configdata["Training"]["classify"]
        self.img_size=configdata["get_data_loaders"]["image_size"]
        self.ndf=configdata["Discriminator"]["ndf"]
        self.nz=configdata["Generator"]["nz"]
        self.nc=configdata["get_data_loaders"]["nc"]
        self.embed_size=configdata["Generator"]["embed_size"]
        if self.classify=="on":
            self.nc+=2
            
        self.label_condition_disc = nn.Sequential(nn.Embedding(num_classes, self.embed_size),
                                                  nn.Linear(self.embed_size, 1*512*512))


        self.main = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf, 8, 4, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf, self.ndf * 4, 8, 4, 2, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 4, self.ndf * 16, 8, 4, 2, bias=False),
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 16, 1, 8, 4, 0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, input,label):
        label_p=torch.div(label, 3, rounding_mode='floor')
        label_p_output = self.label_condition_disc(label_p)
        label_p_output = label_p_output.view(-1, 1, 512, 512)
        label_d=torch.remainder(label,3)
        label_d_output = self.label_condition_disc(label_d)
        label_d_output = label_d_output.view(-1, 1, 512, 512)
        concat = torch.cat((input, label_d_output, label_p_output), dim=1)
        output = self.main(concat).view(input.size(0),1)
        return output

