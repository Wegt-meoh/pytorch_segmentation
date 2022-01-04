'''
test for video
'''


import os
import argparse
from time import time
import numpy
import torch
from torchvision import transforms
from PIL import Image
import cv2

from models.get_segmentation_model import get_segmentation_model
from dataload.utils import get_pred


def parse_args():
    parser = argparse.ArgumentParser(
        description='Predict segmentation result from a given image')
    parser.add_argument('--model', type=str, default='bisenetv',
                        help='model name (default: fcn32_vgg16)')
    parser.add_argument('--backbone', type=str, default='resnet34')
    parser.add_argument('--aux', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['pascal_voc, pascal_aug, ade20k, citys'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--num_class', type=int, default=19)
    parser.add_argument('--pretrained_model', type=str,
                        default=r'D:\Tramac\result\bisenetv_resnet34_cityscapes\models\best_model.pth')

    parser.add_argument('--save_if', type=bool, default=True)
    parser.add_argument('--base_size', type=int, default=700)
    parser.add_argument('--save_dir', type=str, default=r'E:\OBS\demoOut')
    parser.add_argument('--input_obj', type=str, default='D:/datasets/229575561-1-208.mp4',
                        help='the vedio path split with / or use camara with a number')
    # parser.add_argument('--input_obj', type=int, default=0,
    #                     help='the vedio path split with / or use camara with a number')

    return parser.parse_args()


class Preder():

    def __init__(self, args) -> None:

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = args.dataset

        # image transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.model = get_segmentation_model(
            args.model,
            dataset=args.dataset,
            backbone=args.backbone,
            aux=args.aux,
            num_class=args.num_class
        ).to(self.device)

        pretrained_model_state_dict = torch.load(
            args.pretrained_model, self.device)
        model_state_dict = self.model.state_dict()
        state_dict_buffer = {k: v for k, v in pretrained_model_state_dict.items(
        ) if k in model_state_dict.keys()}
        self.model.load_state_dict(state_dict_buffer)

        self.model.eval()

    def pred(self, img):
        with torch.no_grad():
            # img is a array
            image = Image.fromarray(img).convert('RGB')
            images = self.transform(image).unsqueeze(0).to(self.device)

            output = self.model(images)
            pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
            mask = get_pred(pred, self.dataset)
            return mask


if __name__ == '__main__':
    args = parse_args()
    originWriter = None
    maskWriter = None
    fourcc = None
    if args.save_if:
        if os.path.exists(args.save_dir):
            assert os.path.isdir(args.save_dir), '%s not a dir' % args.save_dir
        else:
            os.makedirs(args.save_dir)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # input your camera(such as 0) or a video path
    cap = cv2.VideoCapture(args.input_obj)

    model = Preder(args)

    while cap.isOpened():
        _, img = cap.read()

        # 图像读取错误
        if not _:
            break

        # resize
        oh, ow, ch = img.shape
        if args.base_size != None:
            img = Image.fromarray(img).convert('RGB')
            if ow > oh:
                h = int(oh * args.base_size/ow)
                w = args.base_size
            else:
                w = int(ow * args.base_size/oh)
                h = args.base_size
            img = img.resize((w, h), Image.BILINEAR)
            ow, oh = w, h

        img = numpy.array(img)
        mask = numpy.array(model.pred(img).convert("RGB"))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        cv2.imshow('origin', img)
        cv2.imshow('pred', mask)

        if args.save_if:
            if originWriter == None or maskWriter == None:
                _t=os.path.join(
                    args.save_dir, '{}_{}_{}'.format(args.model,args.backbone,args.dataset))
                if os.path.exists(_t):
                    assert os.path.isdir(_t), '{} is a file you need delete it first'.format(_t)
                else :
                    os.makedirs(_t)                        
                _s=args.input_obj.split('/')[-1].split('.')[0]
                originWriter = cv2.VideoWriter(_t+'/'+_s+'_origin.avi', fourcc, 30, (ow, oh))
                maskWriter = cv2.VideoWriter(_t+'/'+_s+'_mask.avi', fourcc, 30, (ow, oh))
            originWriter.write(img)
            maskWriter.write(mask)

        # if cv2.waitKey(1) == 27:
        #     cv2.destroyAllWindows()
        #     break

    cap.release()
    if maskWriter != None:
        maskWriter.release()
    if originWriter != None:
        originWriter.release()
