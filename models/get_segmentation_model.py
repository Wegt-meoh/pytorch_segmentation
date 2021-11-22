from models.simpleNet import SimpleNet
from models.bisenet import BiSeNet
from models.bisenetv import BiSeNetV


models={
    'simplenet':SimpleNet,
    'bisenet':BiSeNet,
    'bisenetv':BiSeNetV
}

def get_segmentation_model(name,**kwargs):
    return models[name.lower()](**kwargs)