from models.simpleNet import SimpleNet
from models.bisenet import BiSeNet


models={
    'simplenet':SimpleNet,
    'bisenet':BiSeNet
}

def get_segmentation_model(name,**kwargs):
    return models[name.lower()](**kwargs)