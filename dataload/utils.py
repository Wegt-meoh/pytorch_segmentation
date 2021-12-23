from PIL import Image
import numpy as np
import os


def save_pred(pred, path, filename, dataset):
    if dataset == 'cvc_voc':
        impred = Image.fromarray(pred.astype('uint8'))
        impred.putpalette(get_cvc_palette())
    elif dataset in ['pascal_voc', 'coco']:
        # pred[pred == -1] = 255
        impred = Image.fromarray(pred.astype('uint8'))
        impred.putpalette(get_voc_pallette(256))
    elif dataset == 'cityscapes':
        impred = Image.fromarray(pred.astype('uint8'))
        impred.putpalette(cityscapes_pallette)

    if not os.path.exists(path):
        os.makedirs(path)
    impred.save(path+'/'+filename+'.png')
    pass


def get_cvc_palette():
    palette = [255, 255, 255, 0, 0, 0]
    for i in range(256*3-6):
        palette.append(0)
    return palette


def get_voc_pallette(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


cityscapes_pallette = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]
