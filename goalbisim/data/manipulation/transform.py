import torch
import numpy as np
from goalbisim.data.manipulation.data_augs import *


def initialize_transform(transform_details):
    transforms = []
    if transform_details:
        for transform in transform_details:

            if transform[0] == 'center_crop':
                transforms.append((center_crop_image, transform, 'np'))
            
            elif transform[0] == 'center_translate':
                transforms.append((center_translate, transform, 'np'))
            
            elif transform[0] == 'random_translate':
                transforms.append((random_translate, transform, 'np'))
            
            elif transform[0] == 'random_color_jitter':
                transforms.append((random_translate, transform, 'torch'))
            
            elif transform[0] == 'random_convolution':
                transforms.append((random_convolution, transform, 'torch'))
            
            elif transform[0] == 'random_rotation':
                transforms.append((random_rotation, transform, 'torch'))
            
            elif transform[0] == 'random_flip':
                transforms.append((random_flip, transform, 'torch'))
            
            elif transform[0] == 'random_grayscale':
                transforms.append((random_grayscale, transform, 'torch'))
            
            elif transform[0] == 'random_crop':
                transforms.append((random_crop, transform, 'np'))

            elif transform[0] == 'imagenet_normalize':
                transforms.append((imagenet_normalize, transform, 'torch'))

            elif transform[0] == 'to_tensor':
                transforms.append((to_tensor, transform, 'np'))
            
            else:
                print(transform, "Not a known transform!")
                raise NotImplementedError
            
        return set_transforms(transforms)
    
    else:
        return set_transforms([])

def set_transforms(transforms):

    def apply_transforms(imgs, device, normalize = True, dependent_translate = True):
        unpack = False
        if not isinstance(imgs, list):
            imgs = [imgs]
            unpack = True

        assert len(imgs) >= 1, "Empty List!"

        if normalize:
            for i in range(len(imgs)):
                if imgs[i].max() >= 1 and imgs[i].min() >= 0:
                    imgs[i] = imgs[i] / 255

        if len(imgs[0].shape) == 3:
            b = 1
            c, h, w = imgs[0].shape
        else:
            b, c, h, w = imgs[0].shape


        sampled = False

        for transform in transforms:
            if transform[2] == 'torch':
                continue
            for i in range(len(imgs)):
                if transform[1][0] == 'random_translate' and dependent_translate:
                    #Translate in the same way...
                    if not sampled:
                        size = transform[1][1]['out']
                        hs = np.random.randint(0, size - h + 1, b)
                        ws = np.random.randint(0, size - w + 1, b)
                        sampled = True
                    imgs[i] = transform[0](imgs[i], h1s = hs, w1s = ws, **transform[1][1])

                else:
                    imgs[i] = transform[0](imgs[i], **transform[1][1]) 

        assert device, "Set a Device!"
        for i in range(len(imgs)):
            imgs[i] = torch.as_tensor(imgs[i], device = device).float().contiguous()

        for transform in transforms:
            if transform[2] == 'np':
                continue
            for i in range(len(imgs)):
                imgs[i] = transform[0](imgs[i], **transform[1][1]) 
        if unpack:
            imgs = imgs[0]       

        return imgs

    return apply_transforms
