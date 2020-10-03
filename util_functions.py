import numpy as np
import time
from collections import OrderedDict

import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim

import json
from PIL import Image

import argparse

def predict(image_path, model, device, topk, cat_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = model.to(device)
    # Implement the code to predict the class from an image file
    image = process_image(image_path, device)
    image = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze_(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        probs, classes = ps.topk(topk, dim=1)
    
    probs, classes = probs.to(device), classes.to(device)
    probs_np = np.array(probs)[0]
    classes_np = np.array(classes)[0]
    
    class_to_idx = model.class_to_idx
        
    idx_to_class = {val: key for key, val in class_to_idx.items()}
        
    probs_class = [idx_to_class[i] for i in classes_np]
    
    with open(cat_name , 'r') as f:
        cat_to_name = json.load(f)
    
    flower = [cat_to_name[i] for i in probs_class]
    return probs_np, flower
    
# loads a saved checkpoint to rebuild the network
def load_checkpoint(filepath, gpu):
    checkpoint = (torch.load(filepath))
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = nn.NLLLoss()
    if gpu == True:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    return model, optimizer, criterion, device

def process_image(image, device):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    new_im = im.resize((256,256))
    
    width, height = new_im.size
    
    new_width = 224
    new_height = 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    
    new_im = im.crop((left, top, right, bottom))
    
    np_image = np.array(new_im) /255
    
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - means) / stds
    
    np_image = np_image.transpose((2,0,1))
    
    return np_image
