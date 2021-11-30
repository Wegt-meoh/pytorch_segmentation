import os
import torch
from tqdm import tqdm
from torch.utils import data
import torch.backends.cudnn as cudnn

from utils.Metric import *
from utils.Logger import get_logger
from dataload.utils import save_pred
from Configer import get_parsed_args
from models.get_segmentation_model import get_segmentation_model
from dataload.get_segmentatio_dataset import get_segmentation_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '7'


class Evalator():
    def __init__(self, args):
        self.args = args

        val_dataset = get_segmentation_dataset(
            name=args.dataset, split='val', base_size=args.base_size, crop_size=args.crop_size)
        self.val_loader = data.DataLoader(
            dataset=val_dataset, shuffle=False, batch_size=1)

        self.num_class = len(val_dataset.classes)

        self.model = get_segmentation_model(name=args.model, backbone=args.backbone,
                                            pretrained_base=False, num_class=len(val_dataset.classes)).to(args.device)

        pretrained_model_state_dict = torch.load(
            args.pretrained_model, args.device)
        model_state_dict = self.model.state_dict()
        state_dict_buffer = {k: v for k, v in pretrained_model_state_dict.items(
        ) if k in model_state_dict.keys()}
        self.model.load_state_dict(state_dict_buffer)
        logger.info('load model:{} as pretrained model'.format(
            args.pretrained_model))

        del args
        pass

    def eval(self):
        self.model.eval()
        print('start eval...')
        total_acc, total_inter, total_union = 0.0, [0.0 for i in range(
            self.num_class)], [0.0 for i in range(self.num_class)]
        with torch.no_grad():
            for iter, (image, mask, _) in enumerate(tqdm(self.val_loader, desc='evaling')):
                iter += 1

                image = image.to(self.args.device)

                pred = self.model(image)
                predict = torch.argmax(pred[0], dim=0)
                predict = predict.cpu().numpy().astype('uint8')
                mask = mask.numpy().astype('uint8')
                target = mask.squeeze(0)
                del pred, mask

                acc, val_sum = accuracy(predict, target)
                intersection, union = intersectionAndUnion(
                    predict, target, self.num_class)

                total_acc += acc
                total_inter += intersection
                total_union += union

                save_pred(predict, '{}/{}_{}_{}/preds'.format(self.args.result_dir,
                          self.args.model, self.args.backbone, self.args.dataset), _[0], self.args.dataset)

        IoU = total_inter/total_union
        logger.info('ACC:{}, mIoU:{}, IoU:{}'.format(
            total_acc/len(self.val_loader), np.nanmean(IoU), IoU))
        pass


if __name__ == '__main__':
    args = get_parsed_args()

    if args.device == 'cuda':
        cudnn.benchmark = True

    logger = get_logger("semantic_segmentation", save_dir="{}/{}_{}_{}".format(
        args.result_dir, args.model, args.backbone, args.dataset), filename='val.txt', mode='a+')
    logger.debug(args)

    evaluator = Evalator(args)
    evaluator.eval()

    pass
