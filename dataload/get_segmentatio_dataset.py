from dataload.pascal_voc import VOCSegmentation
from dataload.cvc_voc import CVCSegmentation

datasets={
    "pascal_voc":VOCSegmentation,
    "cvc_voc":CVCSegmentation
}

def get_segmentation_dataset(name,**kwargs):
    return datasets[name.lower()](**kwargs)