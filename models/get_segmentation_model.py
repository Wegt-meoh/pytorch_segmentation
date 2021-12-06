from models.simpleNet import SimpleNet
from models.bisenet import BiSeNet
from models.bisenetv import BiSeNetV
from models.bisenetvv import BiSeNetVV
from models.deeplabv3plus import deeplabv3plus


models = {
    'simplenet': SimpleNet,
    'bisenet': BiSeNet,
    'bisenetv': BiSeNetV,
    'bisenetvv': BiSeNetVV,
    'deeplabv3plus': deeplabv3plus
}


def get_segmentation_model(name, **kwargs):
    return models[name.lower()](**kwargs)
