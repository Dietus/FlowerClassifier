import argparse
import os
import torch
import matplotlib.pyplot as plt 
import numpy as np
import json

from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image

def parse():
#arguments for training
    parser = argparse.ArgumentParser(description='Training a neural network with parameters')
    parser.add_argument('data_directory', help='directory to dataset')
    parser.add_argument('--save_dir', help='directory to save checkpoint')
    parser.add_argument('--arch', help='models to use OPTIONS[vgg,densenet]')
    parser.add_argument('--learning_rate', help='learning rate')
    parser.add_argument('--hidden_units', help='number of hidden units')
    parser.add_argument('--epochs', help='epochs')
    parser.add_argument('--gpu',action='store_true', help='Use GPU')
    args = parser.parse_args()
    return args

#Pre-process data
def preprocess():
    print('Processing data')
    train_dir = args.data_directory + '/train'
    test_dir = args.data_directory + '/test'
    valid_dir = args.data_directory + '/valid'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    data_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=train_transforms), 
                      'valid': datasets.ImageFolder(valid_dir, transform=data_transforms), 
                      'test' : datasets.ImageFolder(test_dir, transform=data_transforms)}

    dataloaders = {'train' :torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True), 
                   'valid' :torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
                   'test' :torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)}
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return dataloaders
    

def network():
    print('Building network')
    if (args.arch is None):
        arch_type = 'vgg'
    else:
        arch_type = args.arch
    if (arch_type == 'vgg'):
        model = models.vgg19(pretrained=True)
        classifier_input=25088
    elif (arch_type == 'densenet'):
        model = models.densenet121(pretrained=True)
        classifier_input=1024
    if (args.hidden_units is None):
        hidden_units = 4096
    else:
        hidden_units = args.hidden_units
    for param in model.parameters():
        param.requires_grad = False

    if args.hidden_units == None:
        hidden_units = 4096
    else:
        hidden_units = args.hidden_units
    hidden_units = int(hidden_units)
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(classifier_input, hidden_units)),
                                ('relu', nn.ReLU()),
                                ('drop', nn.Dropout(0.1)),
                                ('fc2', nn.Linear(hidden_units, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
    model.classifier = classifier
    return model
    
def train(model, loader):
    print('Training model')
    dataloaders = loader
    device = torch.device("cuda" if torch.cuda.is_available() & args.gpu else "cpu")
    model.to(device)
    criterion = nn.NLLLoss()
             
    if args.learning_rate == None:
             learn_rate = 0.001
    else:
             learn_rate = float(args.learning_rate)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    
    if args.epochs is None:
             epochs = 2
    else:
             epochs = int(args.epochs)
    running_loss = 0
    steps = 0
    print_every = 30
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:            
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    print("Epoch {}/{}..".format(epoch+1, epochs))
                    print("Train loss: {:.3f}..".format(running_loss/print_every))
                    print("Validation loss: {:.3f}..".format(test_loss/len(dataloaders['valid'])))
                    print("Validation accuracy: {:.3f}".format(accuracy/len(dataloaders['valid'])))
                model.train()
                running_loss=0
    print('Finished training')
    test_accuracy(model, dataloaders['test'])

def test_accuracy(model, loader):
             model.eval()
             device = torch.device("cuda" if torch.cuda.is_available() & args.gpu else "cpu")
             model.to(device)
             criterion = nn.NLLLoss()
             test_loss = 0
             accuracy = 0
             with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
             print("Validation loss: {:.3f}..".format(test_loss/len(loader)))
             print("Validation accuracy: {:.3f}".format(accuracy/len(loader)))
             print("saving model")
             if (args.save_dir is None):
                 save_dir = 'check.pth'
             else:
                 save_dir = args.save_dir
             checkpoint = {'arch': args.arch,
                           'features': model.features,
                           'classifier': model.classifier,
                           'state_dict': model.state_dict()}
             torch.save(checkpoint, save_dir)

def check_params():
    print("Confirming parameters")
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU detected")
    if(not os.path.isdir(args.data_directory)):
        raise Exception('directory does not exist!')
    data_dir = os.listdir(args.data_directory)
    if (not set(data_dir).issubset({'test','train','valid'})):
        raise Exception('missing: test, train or valid sub-directories')
    if args.arch not in ('vgg','densenet',None):
        raise Exception('Please choose one of: vgg or densenet') 

def main():
    global args
    args = parse()
    check_params()
    data = preprocess()
    model = network()
    model = train(model,data)


main()