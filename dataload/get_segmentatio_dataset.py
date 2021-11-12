from dataload.pascal_voc import VOCSegmentation

datasets={
    "pascal_voc":VOCSegmentation
}

def get_segmentation_dataset(name,**kwargs):
    return datasets[name.lower()](**kwargs)