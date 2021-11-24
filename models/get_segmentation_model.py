from models.simpleNet import SimpleNet
from models.bisenet import BiSeNet
from models.bisenetv import BiSeNetV
from models.bisenetvv import BiSeNetVV


models={
    'simplenet':SimpleNet,
    'bisenet':BiSeNet,
    'bisenetv':BiSeNetV,
    'bisenetvv':BiSeNetVV
}

def get_segmentation_model(name,**kwargs):
    return models[name.lower()](**kwargs)