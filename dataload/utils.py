from PIL import Image
import numpy as np
import os

def save_pred(pred,path,filename,dataset):                                   
        impred=Image.fromarray(pred)        
        if dataset=='cvc_voc':
            impred.putpalette(cvc_palette())
        
        if not os.path.exists(path):
            os.makedirs(path)
        impred.save(path+'/'+filename+'.png')
        pass 

def cvc_palette():
    palette=[255,255,255,0,0,0]
    for i in range(256*3-6):
        palette.append(0)
    return palette