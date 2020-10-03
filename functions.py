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

def load_data():
    #data directoy
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Transforms for the training, validation and training datasets
    data_transforms = {
    'val': transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]),
    
    'test': transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]),

    'train': transforms.Compose([
                            transforms.RandomResizedCrop(size=224, scale=(0.8,1.0)),
                            transforms.RandomRotation(degrees=15),
                            transforms.RandomHorizontalFlip(),
                            #transforms.RandomVerticalFlip(),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                        ]),
    }
    
    #Load the datasets with ImageFolder
    dataset = {
        'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
         'val': datasets.ImageFolder(valid_dir, transform = data_transforms['val']),
         'test': datasets.ImageFolder(test_dir, transform = data_transforms['test'])
    }
    #Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(dataset['train'], batch_size = 64, shuffle=True),
         'val': torch.utils.data.DataLoader(dataset['val'], batch_size = 64),
         'test': torch.utils.data.DataLoader(dataset['test'], batch_size = 32)
    }
    
    return dataset, dataloaders

def get_model(model_arch):
    #load pre-trained model
    model = getattr(models, model_arch)(pretrained=True)
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model
        

def network_param(model, arch, hidden_units, learning_rate, gpu):
    #Build classifier
    if arch == 'vgg16':
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units[0])),
                              ('relu', nn.ReLU()),
                              ('drop1', nn.Dropout(0.5)),
                              ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                              ('relu2', nn.ReLU()),
                              ('drop2', nn.Dropout(0.2)),
                              ('fc3', nn.Linear(hidden_units[1], 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    elif arch == 'alexnet':
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(9216, hidden_units[0])),
            ('relu', nn.ReLU()),
            ('drop1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
            ('relu2', nn.ReLU()),
            ('drop2', nn.Dropout(0.2)),
            ('fc3', nn.Linear(hidden_units[1], 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
    model.classifier = classifier
    print('printing model....: \n{}'.format(model))
    
    #define the criterion and optimizer for backpropagation
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= learning_rate)
    
    #set the desired device to use ('cpu', 'gpu')
    
    if gpu == True:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    return model, classifier, criterion, optimizer, device

def train_model(model, criterion, optimizer, device, train_data, val_data, epochs):

    start_time = time.time()

    print_every = 32
    steps = 0
    train_loss = 0
    train_accuracy = 0
    
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in train_data:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
                                                     
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    val_loss, val_accuracy = val_model(model, criterion, device, val_data)
                    
                print('Epoch: {}/{}...'.format(e+1, epochs),
                    'Train Loss: {:.4f}'.format(train_loss/ len(train_data)),
                      #'Train accuracy: {:.4f}'.format(100 * train_accuracy / print_every),
                    'Validation Loss: {:.4f}..'.format(val_loss/len(val_data)),
                      'Validation Accuracy: {:.4f}'.format(100* val_accuracy/len(val_data)))
                
                train_loss = 0
    
    print('total time taken: {:.1f}s'.format(time.time() - start_time))


def val_model(model,criterion,device,val_data):
    val_accuracy = 0
    val_loss = 0
    for images, labels in val_data:

        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = criterion(output, labels)
        val_loss += loss.item()
        
        ps = torch.exp(output)
        matching = (labels.data == ps.max(1)[1])
        val_accuracy += matching.type_as(torch.FloatTensor()).mean()

    return val_loss, val_accuracy

# validation on the test set
def test_model(model,criterion,device,test_data): 
    start_time = time.time()
    
    test_loss = 0
    test_accuracy = 0
    model.eval()
    
    with torch.no_grad():
        for images, labels in test_data:

            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)

            loss = criterion(output, labels)
            test_loss += loss.item()

            ps = torch.exp(output)
            matching = (labels.data == ps.max(1)[1])
            test_accuracy += matching.type_as(torch.FloatTensor()).mean()

        print('Test Loss: {:.3f}'.format(test_loss/len(test_data)),
        'Test Accuracy: {:.4f}'.format(100 * test_accuracy/len(test_data)))

    print('total time taken: {:.1f}s'.format(time.time() - start_time))
    

def save_checkpoint(model, optimizer,train_data, arch, epochs, save_dir):
    # Save the checkpoint 
    model.class_to_idx = train_data['train'].class_to_idx
    checkpoint = {
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'arch': arch ,
        'epochs': epochs,
        'optimizer': optimizer,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_dir)