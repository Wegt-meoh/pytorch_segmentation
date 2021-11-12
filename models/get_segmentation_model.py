from models.simpleNet import SimpleNet


models={
    'simplenet':SimpleNet
}

def get_segmentation_model(name,**kwargs):
    return models[name.lower()](**kwargs)