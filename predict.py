import argparse 
import torch 
import numpy as np
import json
import sys

from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image

def parse():
    parser = argparse.ArgumentParser(description='use a neural network to classify an image!')
    parser.add_argument('image_input', help='image file to classifiy (required)')
    parser.add_argument('model_checkpoint', help='model used for classification (required)')
    parser.add_argument('--top_k', help='how many prediction categories to show [default 5].')
    parser.add_argument('--category_names', default = 'cat_to_name.json',  help='file for category names')
    parser.add_argument('--gpu', action='store_true', help='gpu option')
    args = parser.parse_args()
    return args

def load_model(filepath):
    model_info = torch.load(filepath)    
    if (model_info['arch'] is None):
        arch_type = 'vgg'
    else:
        arch_type = model_info['arch']
    if (arch_type == 'vgg'):
        model = models.vgg19(pretrained=True)
    elif (arch_type == 'densenet'):
        model = models.densenet121(pretrained=True)
    model.classifier = model_info['classifier']
    model.load_state_dict(model_info['state_dict'])
    return model

def process_image(image):
    im = Image.open(image)
    width, height = im.size
    dims = [width, height]
    aspect = width/height
    if width < height:
        im = im.resize((256, int(height * (256/width))))
    else:
        im = im.resize((int(width * (256/height)), 256))
    width, height = im.size
    top = (height - 244)/2
    down = (height + 244)/2
    left = (width - 244)/2
    right = (width + 244)/2
    
    im = im.crop((left,top,right,down))
    final_image = np.array(im)
    final_image = final_image.astype('float64')
    final_image = final_image / [255,255,255]
    final_image = (final_image - [0.485, 0.456, 0.406])/ [0.229, 0.224, 0.225]
    final_image = final_image.transpose((2, 0, 1))
    
    return final_image

def map_label():
    if (args.category_names is not None):
        cat_file = args.category_names 
        jfile = json.loads(open(cat_file).read())
        return jfile
    return None

def predict(image_path, model, topk=5):
    with torch.no_grad():
        im = process_image(image_path)
        im = torch.from_numpy(im)
        im.unsqueeze_(0)
        im = im.float()
        if (args.gpu):
           im = im.cuda()
           model = model.cuda()
        else:
            im = im.cpu()
            model = model.cpu()
        output = model(im)
        
        probs, classes = torch.exp(output).topk(topk, dim=1)
        return zip(probs[0].tolist(), classes[0].tolist())

def display_prediction(prediction):
    cat_file = map_label()
    i = 0
    for p, c in prediction:
        i = i + 1
        p = str(round(p,4) * 100.) + '%'
        if (cat_file):
            c = cat_file.get(str(c+1),'None')
        else:
            c = ' class {}'.format(str(c))
        print("{}.{} ({})".format(i,c,p))
    return None
    
def main():
    global args
    args = parse()
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled but no GPU detected")
    if (args.top_k is None):
        top_k = 5
    else:
        top_k = int(args.top_k)
    model = load_model(args.model_checkpoint)
    prediction = predict(args.image_input, model, top_k)
    display_prediction(prediction)


main()