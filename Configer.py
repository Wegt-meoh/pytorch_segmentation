import argparse

def get_parsed_args():
    args=argparse.ArgumentParser(description="segmentation parameter")

    args.add_argument("--model",type=str,default='bisenet')
    args.add_argument('--pretrained_model',type=str,default='/home/deep1/QuePengbiao/result/simplenet_resnet34_cvc_voc/simplenet_resnet34_cvc_voc_120',
                        help='only used in eval')
    args.add_argument("--backbone",type=str,default='resnet34')
    args.add_argument("--pretrained_base",type=bool,default=True)
    args.add_argument("--backbone_dir",type=str,default='/home/deep1/QuePengbiao/pretrain_models')
    args.add_argument("--lr",type=float,default=1e-4)

    args.add_argument("--dataset",type=str,default='cvc_voc')
    args.add_argument("--epoch",type=int,default=120)
    args.add_argument("--batch_size",type=int,default=6)
    args.add_argument("--base_size",type=int,default=220)
    args.add_argument("--crop_size",type=int,default=230)
    args.add_argument("--workers",type=int,default=8)

    args.add_argument("--device",type=str,default='cuda',choices=['cuda','cpu'])

    args.add_argument("--log_dir",type=str,default='/home/deep1/QuePengbiao/result')
    args.add_argument("--model_save_dir",type=str,default='/home/deep1/QuePengbiao/result')
    args.add_argument("--pred_save_dir",type=str,default='/home/deep1/QuePengbiao/result')

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
