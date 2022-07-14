import torch
import numpy as np

from goalbisim.data.manipulation.data_augs import *

def initialize_preprocess(details):
    transforms = []
    if details['preprocess']:
        for transform in details['preprocess']:

            if transform == 'random_color_jitter':
                transforms.append((random_translate, transform, 'torch'))
            
            elif transform == 'random_convolution':
                transforms.append((random_convolution, transform, 'torch'))
            
            elif transform == 'random_rotation':
                transforms.append((random_rotation, transform, 'torch'))
            
            elif transform == 'random_flip':
                transforms.append((random_flip, transform, 'torch'))
            
            elif transform == 'random_grayscale':
                transforms.append((random_grayscale, transform, 'torch'))
            

            elif transform == 'imagenet_normalize':
                transforms.append((imagenet_normalize, transform, 'torch'))

            
            else:
                print(transform, "Not a known transform!")
                raise NotImplementedError
            
        return transforms
    
    else:
        return []


def apply_preprocess(img, preprocesses, normalize = True):
    #assert len(imgs) >= 1, "Empty List!"

    if normalize:
        if img.max() >= 1 and img.min() >= 0:
            img = img / 255


    for preprocess in preprocesses:
        if preprocess[2] == 'np':
            continue
        img = preprocess[0](img) 


    return img