from dataload.pascal_voc import VOCSegmentation
from dataload.cvc_voc import CVCSegmentation
from dataload.coco_voc import COCOSegmentation
from dataload.pascal_aug import VOCAugSegmentation
from dataload.cityscapes import CitySegmentation

datasets = {
    "pascal_voc": VOCSegmentation,
    "cvc_voc": CVCSegmentation,
    'coco': COCOSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'cityscapes': CitySegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
