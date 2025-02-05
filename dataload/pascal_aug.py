"""Pascal Augmented VOC Semantic Segmentation Dataset."""
import scipy.io as sio
import os
import random
import torch

from PIL import Image, ImageFilter, ImageOps
import numpy as np
from torchvision import transforms


class VOCAugSegmentation():
    """Pascal VOC Augmented Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is './datasets/voc'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = VOCAugSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'dataset/'
    NUM_CLASS = 21

    def __init__(self, root='/home/deep1/xxxx/datasets/benchmark_RELEASE', split='train', base_size=513, crop_size=513, **kwargs):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.split = split
        self.base_size = base_size
        self.crop_size = crop_size

        # train/val/test splits are pre-cut
        _voc_root = os.path.join(root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'cls')
        _image_dir = os.path.join(_voc_root, 'img')
        if split == 'train':
            _split_f = os.path.join(_voc_root, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(_voc_root, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(split))

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".mat")
                assert os.path.isfile(_mask)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))
        print('Found {} images in the folder {}'.format(
            len(self.images), _voc_root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = self._load_mat(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, target = self._sync_transform(img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform(img, target)
        else:
            raise RuntimeError(
                'unknown mode for dataloader: {}'.format(self.mode))
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, target, os.path.basename(self.images[index])

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _load_mat(self, filename):
        mat = sio.loadmat(filename, mat_dtype=True,
                          squeeze_me=True, struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        return Image.fromarray(mask)

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(
            int(self.base_size * 0.5), int(self.base_size * 2.0))
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

    def _val_sync_transform(self, img, mask):
        # img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        # mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)

        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')


if __name__ == '__main__':
    dataset = VOCAugSegmentation()
