import argparse


def get_parsed_args():
    args = argparse.ArgumentParser(description="segmentation parameter")

    args.add_argument("--model", type=str, default='bisenet')
    args.add_argument('--pretrained_model', type=str, default='/home/deep1/xxxx/result/bisenet_resnet34_pascal_voc/models/best_model.pth',
                      help='only used in eval')
    args.add_argument("--backbone", type=str, default='resnet34')
    args.add_argument("--pretrained_base", type=bool, default=True)
    args.add_argument("--backbone_dir", type=str,
                      default='/home/deep1/xxxx/pretrain_models')
    args.add_argument("--lr", type=float, default=2.5e-2)

    args.add_argument("--dataset", type=str, default='pascal_voc')
    args.add_argument("--epoch", type=int, default=1600)
    args.add_argument("--batch_size", type=int, default=4)
    args.add_argument("--base_size", type=int, default=1024)
    args.add_argument("--crop_size", type=int, default=1024)
    args.add_argument("--workers", type=int, default=8)

    args.add_argument("--aux", type=bool, default=False)
    args.add_argument("--aux_weight", type=float, default=1.0)

    args.add_argument("--device", type=str, default='cuda',
                      choices=['cuda', 'cpu'])

    args.add_argument("--result_dir", type=str,
                      default='/home/deep1/xxxx/result')

    return args.parse_args()


if __name__ == '__main__':
    args = get_parsed_args()
    print(args.model)
    print(args.backbone)
    print(args.lr)
    print(args.dataset)
    print(args.epoch)
    print(args.batch_size)
    args.device = 'cpu'
    print(args.device)
