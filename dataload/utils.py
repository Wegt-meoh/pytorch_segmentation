from PIL import Image
import numpy as np
import os


def save_pred(pred, path, filename, dataset):
    impred = Image.fromarray(pred)
    if dataset == 'cvc_voc':
        impred.putpalette(cvc_palette())
    elif dataset in ['pascal_voc', 'coco']:
        impred.putpalette(_getvocpallete(256))

    if not os.path.exists(path):
        os.makedirs(path)
    impred.save(path+'/'+filename+'.png')
    pass


def cvc_palette():
    palette = [255, 255, 255, 0, 0, 0]
    for i in range(256*3-6):
        palette.append(0)
    return palette


def _getvocpallete(num_cls):
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
