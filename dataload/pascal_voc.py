import os
import random
import torch

from PIL import Image, ImageFilter, ImageOps
import numpy as np
from torchvision import transforms


class VOCSegmentation():
    def __init__(self,root='/home/deep1/QuePengbiao/datasets/VOCdevkit/VOC2012',split='train',base_size=513,crop_size=513,**kwargs) -> None:        
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.split=split
        self.base_size=base_size
        self.crop_size=crop_size
        
        mask_dir=os.path.join(root,'SegmentationClass')
        image_dir=os.path.join(root,'JPEGImages')
        voc_root=os.path.join(root,'ImageSets/Segmentation')
        split_file=os.path.join(voc_root,'{}.txt'.format(split))

        self.images,self.masks=[],[]

        with open(split_file,'r') as lines:
            for line in lines:
                _image = os.path.join(image_dir, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if split!='test':
                    _mask = os.path.join(mask_dir, line.rstrip('\n') + ".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if split!='test':
            assert len(self.images)==len(self.masks)

        print('Found {} images in the folder {}'.format(len(self.images),image_dir))        

    def __getitem__(self,index):
        img=Image.open(self.images[index]).convert('RGB')
        
        if self.split=='test':
            img=self._img_transform(img)
            if self.transform != None:
                img=self.transform(img)
            return img,os.path.basename(self.images[index])

        mask=Image.open(self.masks[index])

        if self.split=='train':
            img,mask=self._sync_transform(img,mask)
        elif self.split=='val':
            img,mask=self._val_sync_transform(img,mask)

        if self.transform!=None:
            img=self.transform(img)
        
        return img,mask,os.path.basename(self.images[index])

    def _img_transform(self,img):
        return np.array(img)

    def _mask_transform(self,mask):        
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def _sync_transform(self,img,mask):             
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)        
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0            
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:            
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _val_sync_transform(self,img,mask):
        # img=img.resize((self.crop_size,self.crop_size),Image.BILINEAR)
        # mask=mask.resize((self.crop_size,self.crop_size),Image.NEAREST)

        img,mask=self._img_transform(img),self._mask_transform(mask)
        return img,mask

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')