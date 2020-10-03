import argparse

def args_input():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save_dir', type = str, default = 'saved_models/checkpoint.pth',
                        help = 'path to the desired saved folder for the trained model checkpoint')
    
    parser.add_argument('--arch', type = str, default = 'vgg16',
                        help = 'enter model architecture, available options: "vgg16" and "alexnet". These need to match when                        loading a pretrained model')
    
    parser.add_argument('--learning_rate', type = float, default = 0.0004,
                        help = 'learning rate for CNN model')
    
    parser.add_argument('--hidden_units', type = int, nargs = 2, default = [4096, 1000],
                        help = 'takes two arguments for the number of hidden units')
    
    parser.add_argument('--epochs', type = int, default = 5,
                        help = 'number of epochs to train the model')
    
    parser.add_argument('--gpu', action='store_true', default = False,
                        help = 'sets GPU for training')
    
    parser.add_argument('--topk', type = int, default = 1,
                        help = 'number of image predictions')
    
    parser.add_argument('--category_name', type = str, default = 'cat_to_name.json',
                        help = 'mapping of flower categories to real names')
    
    parser.add_argument('--input', type = str, action = 'store',
                        help = 'path to image for processing through image classifier')
    
    parser.add_argument('--checkpoint', type = str, default = 'saved_models/checkpoint.pth',
                        help = 'path to directory of the saved trained model')
    
    return parser.parse_args()