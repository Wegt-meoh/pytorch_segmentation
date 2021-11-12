import argparse
from math import trunc
from sys import stdout

def get_parsed_args():
    args=argparse.ArgumentParser(description="segmentation parameter")

    args.add_argument("--model",type=str,default='simplenet')
    args.add_argument("--backbone",type=str,default='resnet18')
    args.add_argument("--pretrained_base",type=bool,default=True)
    args.add_argument("--backbone_dir",type=str,default=r'D:\pretrainModel')
    args.add_argument("--lr",type=float,default=1e-4)

    args.add_argument("--dataset",type=str,default='pascal_voc')
    args.add_argument("--epoch",type=int,default=60)
    args.add_argument("--batch_size",type=int,default=8)
    args.add_argument("--base_size",type=int,default=513)
    args.add_argument("--crop_size",type=int,default=513)
    args.add_argument("--workers",type=int,default=4)

    args.add_argument("--device",type=str,default='cuda',choices=['cuda','cpu'])

    args.add_argument("--log_dir",type=str,default=r'D:\Tramac\mymodelgroup\results')
    args.add_argument("--model_save_dir",type=str,default='')

    return args.parse_args()

if __name__=='__main__':
    args=get_parsed_args()
    print(args.model)
    print(args.backbone)
    print(args.lr)
    print(args.dataset)
    print(args.epoch)
    print(args.batch_size)
    args.device='cpu'
    print(args.device)
