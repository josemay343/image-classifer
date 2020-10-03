import os
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
import functions
from functions import network_param
from args_input import args_input

def main():
    start_time = time.time()
    
    in_arg = args_input()
    # get the data
    dataset, dataloaders = functions.load_data()
    
    # get the model
    model = functions.get_model(in_arg.arch)

    # get the classifier, criterion, optimizer, device
    model, classifier, criterion, optimizer, device = network_param(model, in_arg.arch, in_arg.hidden_units,                                                                                        in_arg.learning_rate, in_arg.gpu)
    
    model.to(device)
        
    print('Training model...')
    functions.train_model(model, criterion, optimizer,device, dataloaders['train'], dataloaders['val'], in_arg.epochs)
    
    # validation on test data
    print('\nGetting results on test data accuracy...')  
    functions.test_model(model,criterion,device, dataloaders['test'])
    
    # saving checkpoint on trained model
    print('\nSaving checkpoint for current trained model...')
    functions.save_checkpoint(model, optimizer, dataset, in_arg.arch, in_arg.epochs, in_arg.save_dir)
    print('Checkpoint saved!')
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
