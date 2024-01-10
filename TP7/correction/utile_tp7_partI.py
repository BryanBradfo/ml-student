#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 18:43:29 2018

@author: lepetit
"""

import os
from os.path import join
from shutil import copyfile
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import norm as nm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from random import randint


#%% Génération d'images de synthèse : 

def simu_rec(image, L,l,  fields=0):
    channels,size,size2=image.size()
    rec = torch.zeros(channels,size,size2)
    #out = 0*(image.clone())
    vertical=np.random.binomial(1,0.5)==1
    if vertical:
        width=l
        height=L
    else:
        width=L
        height=l    
        
    top = randint(0, size-height)
    left = randint(0, size-width)     
    image[fields,top:top+height,left:left+width] = np.random.uniform(0,1)   #0.1

    return image



def simu_noisy_rec(image,meanrec,sigmarec, L,l,  fields=0):
    channels,size,size2=image.size()
    rec= torch.zeros(channels,size,size2)
    #out = 0*(image.clone())
    vertical=np.random.binomial(1,0.5)==1
    if vertical:
        width=l
        height=L
    else:
        width=L
        height=l    
        
    top=randint(0, size-height)
    left=randint(0, size-width)     
    
    rec[fields,top:top+height,left:left+width]= 1   #0.1
    noise= meanrec+ sigmarec*torch.randn(channels,size,size2)
    noise=noise*(noise>0).float()
    rec=noise*rec
    image=image +rec
    return image

    
def simu_disc1(image, mean, sigma, f, mradius=15, fields=0 ):  #radial decrease function  #or 12
    channels,size,size2=image.size()
    radius = randint(mradius - 7, mradius + 7)
    center=np.array([randint(radius, size-radius) , randint(radius, size-radius) ])
    npdisc= f( ((np.arange(0,64)*np.ones([size,size])) - center[0])**2 + (np.transpose(np.arange(0,64)*np.ones([size,size]))-center[1])**2  , radius)
    npdisc = torch.from_numpy(npdisc).float()
    noise = torch.randn(channels,size,size2)
    npdisc = (mean + sigma*noise)* npdisc   #bruitage du disque
    image[fields,:,:] = image[fields,:,:] + npdisc   #matrice des distances < rayon
    return image




def generate_noise(image, lambda_rec=0.001 ,lambda_noisy_rec = 0.001,meanrec= 0.9, sigmarec= 0.4, rdisc = 20, meandisc= 0.5, sigmadisc= 0.2, pola=[0,0.5,0.1]):
    for k in range(0,1):#range(np.random.poisson(lambda_disc*64*64)):
        #image=simu_disc(image, lambda a,x  : (0.39 - 0.36*a/x**2)*(a < x**2) ,radius = r)  #0.47 pour avoir 40 dB
        image=simu_disc1(image, meandisc, sigmadisc, lambda a,x  : (a < x**2) ,mradius = rdisc)
    for j in range(np.random.poisson(lambda_noisy_rec*64*64)):
        L=randint(50,60)
        l=randint(2,10)
        image= simu_noisy_rec(image,meanrec,sigmarec,L,l) 
    for i in range(np.random.poisson(lambda_rec*64*64)):
        L=randint(50,60)
        l=randint(2,10)
        image= simu_rec(image,L,l)
    return image

#%% Réglages : 

#image=torch.zeros([1,64, 64])    
#
#sigma = np.random.uniform(0,1)
##        y = ytilde + 0.5*sigma*np.random.normal(0,1)
#
#y = 0.2 + np.random.uniform(0.2,0.9)
#meanrec = np.random.uniform(0.1,10)
#sigmarec = np.random.uniform(0,1)
#image = torch.zeros([1,64, 64])  
##image=generate_noise(image, lambda_rec=0.0008, lambda_noisy_rec = 0.0008, meanrec= 0.9, sigmarec= 0.4, rdisc = 20, meandisc= 0.5, sigmadisc= 0.2)
#image = generate_noise(image, lambda_rec=0.0008, lambda_noisy_rec = 0.0008, meanrec= meanrec, sigmarec= sigmarec, rdisc = 13, meandisc= y, sigmadisc= sigma)
#
#
#fig = plt.figure(0, figsize=(8, 6))    
#image=image.unsqueeze(0)
#voir_tens((image[:,[0],:,:]).cpu(), fig, min_scale=0,max_scale=10)   

#%% Train/val sets sigma = sigmadisc  base 2: sigma = lamda * radius
    
def generate_dataset(dir_dataset, size_dataset=10000):

    dic_labels = {}    
    if not os.path.isdir(dir_dataset):
        os.mkdir(dir_dataset)
    if not os.path.isdir(join(dir_dataset, 'images')):
        os.mkdir(join(dir_dataset, 'images'))
        
    for i in range(size_dataset):
#        ytilde= np.random.uniform(0,10)
        
#        y = ytilde + 0.5*sigma*np.random.normal(0,1)
        y = 0.2 + np.random.uniform(0.2,7.8)
        sigma = np.random.uniform(0,1)
        meanrec = np.random.uniform(0.1,10)
        sigmarec = np.random.uniform(0,1)
        image = torch.zeros([1,64, 64])  
        #image=generate_noise(image, lambda_rec=0.0008, lambda_noisy_rec = 0.0008, meanrec= 0.9, sigmarec= 0.4, rdisc = 20, meandisc= 0.5, sigmadisc= 0.2)
        image = generate_noise(image, lambda_rec=0.0006, lambda_noisy_rec = 0.0006, meanrec= meanrec, sigmarec= sigmarec, rdisc = 13, meandisc= y, sigmadisc= sigma)

        name = str(i) + '.pt'
        path = join(dir_dataset,'images',name)
        torch.save(image, path)
        dic_labels[name] ={'y': y, 'sigma': sigma} #, 'ytilde':ytilde}
    
    name_dic = os.path.join(dir_dataset, 'labels_synthese.pickle')
    
    with open(name_dic, 'wb') as handle:
        pickle.dump(dic_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%###############Fonctions utiles


def voir_mat(data2, fig, min_scale=-10,max_scale=70):

    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(data2, interpolation='nearest', cmap=plt.cm.ocean) #cmap=plt.cm.rainbow)
    plt.clim(min_scale,max_scale)
    plt.colorbar()
    plt.show()
    
def voir_tens(image, fig, min_scale=-10,max_scale=70):
    im=image[0,0,:,:].numpy()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(im, interpolation='nearest',  cmap=plt.cm.ocean) #cmap=plt.cm.rainbow)
    plt.clim(min_scale,max_scale)
    plt.colorbar()
    plt.show()
    

def conc(image1,image2,dim=3):
    return torch.cat((image1,image2), dim) #, out=None) 

def multi_conc(L,dim=1,ecart=5):
    image1=L[0]
    for i in range(1, len(L)):
        if dim==1:
            sep=100+0*image1[:,0:ecart]
        elif dim==0:
            sep=100+0*image1[0:ecart,:]
        image1=conc(image1,sep,dim)
        image2=L[i]
        image1=conc(image1,image2,dim=dim)
    return image1

def images_from_indices(rep_radar,indices,k=0):
    L=[]
    for i in range(0, len(indices)):
        fic=os.listdir(rep_radar)[indices[i]]
        image=torch.load(rep_radar+'/'+fic)[k,:,:]
        L.append(image)
    return L


def images_from_indices2(rep_radar,indices,k=0):
    L=[]
    for i in range(0, len(indices)):
        fic=os.listdir(rep_radar)[indices[i]]
        image=torch.load(rep_radar+'/'+fic)[k,:,:]
        L.append(image)
    return L


def images_from_tenseur(tens):
    len_batch=tens.shape[0]
    L=[]
    for i in range(len_batch):
        L.append(tens[i,0,:,:])
    return L




def images_from_tenseur2(tens, k=0):
    len_batch=tens.shape[0]
    L=[]
    for i in range(len_batch):
        L.append(tens[i,k,:,:])
    return L


def voir_fichiers2(rep_radar,indices, fig, k=0, min_scale=-10,max_scale=70,dim=1):
    L=images_from_indices2(rep_radar,indices,k)
    image=multi_conc(L,dim)
    voir_mat(image, fig, min_scale,max_scale)
  
    
def voir_fichiers2D(rep_radar,indices,nx, fig, k=0, min_scale=-10,max_scale=70):
    L=images_from_indices(rep_radar,indices,k)
    image1=multi_conc(L[0:nx],dim=1)
    for i in range(1,int(len(indices)/nx)):
        image2=multi_conc(L[i*nx:(i+1)*nx],dim=1)
        image1=multi_conc([image1,image2],dim=0)
    voir_mat(image1, fig, min_scale,max_scale)   

def voir_batch2D(tens, nx, fig,k=0, min_scale=-10,max_scale=1):
    L=images_from_tenseur2(tens,k)
    image1=multi_conc(L[0:nx],dim=1)
    for i in range(1,int(len(L)/nx)):
        image2=multi_conc(L[i*nx:(i+1)*nx],dim=1)
        image1=multi_conc([image1,image2],dim=0)
    voir_mat(image1, fig, min_scale,max_scale)   
    
    
def voir_result2D(tens,out, nx, fig, k=0, min_scale=-10,max_scale=70, Sous_liste=None):
    Lin=images_from_tenseur2(tens, k)
    Lout=images_from_tenseur2(out, k)
    image1=multi_conc(Lin[0:nx],dim=1)
    image2=multi_conc(Lout[0:nx],dim=1)
    image=multi_conc([image1,image2],dim=0)
    for i in range(1,int(len(Lin)/nx)):
        image1=multi_conc(Lin[i*nx:(i+1)*nx],dim=1)
        image2=multi_conc(Lout[i*nx:(i+1)*nx],dim=1)
        image=multi_conc([image,image1],dim=0,ecart=20)
        image=multi_conc([image,image2],dim=0)
    voir_mat(image, fig, min_scale,max_scale)
    
    
    
def voir_segment2D(tens,out, target, nx, fig, k=0, min_scale=-10,max_scale=70,Sous_liste=None):
    Lin=images_from_tenseur2(tens,k)
    Lout=images_from_tenseur2(out,k)
    Ltarget=images_from_tenseur(target)
    image1=multi_conc(Lin[0:nx],dim=1)
    image2=multi_conc(Lout[0:nx],dim=1)
    image3=multi_conc(Ltarget[0:nx],dim=1)
    image=multi_conc([image1,image2,image3],dim=0)
    for i in range(1,int(len(Lin)/nx)):
        image1=multi_conc(Lin[i*nx:(i+1)*nx],dim=1)
        image2=multi_conc(Lout[i*nx:(i+1)*nx],dim=1)
        image3=multi_conc(Ltarget[i*nx:(i+1)*nx],dim=1)
        image=multi_conc([image,image1],dim=0,ecart=20)
        image=multi_conc([image,image2],dim=0)
        image=multi_conc([image,image3],dim=0)
    voir_mat(image, fig, min_scale,max_scale)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        copyfile(filename, 'model_best.pth.tar')
        

def load_archi(pt,segmenter=False):   #name: nom de l'experience, k: rang de la variable
    rac_archis = pt['rac_archis']
    
    if segmenter:
        index=pt['load_segmenter'][2]
        name = rac_archis+pt['load_segmenter'][1]        
    else:
        index=pt['load_archi'][2]
        name = rac_archis+pt['load_archi'][1]
    experience=torch.load(name)   
    return experience['archis'][index]


def BCELoss_List(out, target):
    mb,var,sec,rg = out.size()
    L=[]
    loss=nn.BCELoss()
    for i in range(mb):
        #np.concatenate(L,
        L.append(loss(Variable(out[[i],:,:,:]),Variable(target[[i],:,:,:])).data[0])
    L=np.asarray(L)
    return L

def extrapole(vec, modele):
    out=0*np.array(modele)
    rapport= int(len(modele)/len(vec))
    for i in range(len(vec)):
        out[rapport*i:rapport*(i+1)]=vec[i]
    return out


def extract_train_val_test(name):
    out=np.load(name)
    return out

def save_train_val_test(learn_indices, val_indices, test_indices, name):
    #learn_indices=np.array(learn_indices)
    #val_indices=np.array(val_indices)
    #test_indices=np.array(test_indices)
    np.save(name, [learn_indices, val_indices, test_indices])

def save_experience(experience,name, complete=True):
    if os.path.exists(name) and complete:
        experience0=torch.load(name)
        experience['variable'] = experience0['variable'] + experience['variable']
        experience['images'] = experience0['images'] + experience['images']
        experience['xs'] = experience0['xs'] + experience['xs']
        experience['ys'] = experience0['ys'] + experience['ys']
        experience['times'] = experience0['times'] + experience['times']
        experience['epochs'] = experience0['epochs'] +experience['epochs']
        experience['val_loss']= experience0['val_loss']+experience['val_loss']
        experience['std_val_losses']= experience0['std_val_losses']+experience['std_val_losses']
        if 'last_pt' in experience0.keys():
            experience['last_pt']= experience0['last_pt']+experience['last_pt']
        if 'archis' in experience0.keys():
            experience['archis']= experience0['archis']+experience['archis']        
        if 'list_val_loss' in experience0.keys():
            experience['list_val_loss']= experience0['list_val_loss']+experience['list_val_loss']              
        torch.save(experience,name)
    else:
        torch.save(experience,name)
 

def cut_fringes(x):
    size1,size2 = x.size()[-2:]
    periph = 12
    if x.dim()==4:
        return x[:,:,periph:size1-periph,periph:size2-periph]
    elif x.dim()==2:
        return x[periph:size1-periph,periph:size2-periph]        
    elif x.dim()==3:
        return x[:,periph:size1-periph,periph:size2-periph]        
