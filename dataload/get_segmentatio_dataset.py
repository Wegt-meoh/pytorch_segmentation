from dataload.pascal_voc import VOCSegmentation
from dataload.cvc_voc import CVCSegmentation
from dataload.coco_voc import COCOSegmentation

datasets = {
    "pascal_voc": VOCSegmentation,
    "cvc_voc": CVCSegmentation,
    'coco': COCOSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
