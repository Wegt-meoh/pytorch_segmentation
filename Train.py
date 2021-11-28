import os
import torch
from torch import nn
from pylab import plt
from torch.utils import data
import torch.backends.cudnn as cudnn

from utils.Metric import *
from utils.Logger import get_logger
from Configer import get_parsed_args
from models.get_segmentation_model import get_segmentation_model
from dataload.get_segmentatio_dataset import get_segmentation_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '6'


class Trainer():
    def __init__(self, args) -> None:
        self.args = args

        train_dataset = get_segmentation_dataset(
            name=args.dataset, split='train', base_size=args.base_size, crop_size=args.crop_size)
        self.train_loader = data.DataLoader(
            dataset=train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True, num_workers=args.workers)

        val_dataset = get_segmentation_dataset(
            name=args.dataset, split='val')
        self.val_loader = data.DataLoader(
            dataset=val_dataset, shuffle=False, batch_size=1)
        self.num_class = len(val_dataset.classes)

        self.model = get_segmentation_model(
            name=args.model, num_class=len(train_dataset.classes), pretrained_base=args.pretrained_base, backbone=args.backbone, backbone_dir=args.backbone_dir).to(args.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-4)

        self.epoch_data, self.acc_data, self.mIoU_data = [], [], []

    def train(self):
        print('Start train...')
        self.model.train()
        iter_max = len(self.train_loader)*self.args.epoch
        for epoch in range(self.args.epoch):
            for iter, (images, masks, _) in enumerate(self.train_loader):
                iter += 1

                images = images.to(self.args.device)
                masks = masks.to(self.args.device)

                self.optimizer.zero_grad()
                lr_now = adjust_lr(self.optimizer, self.args.lr,
                                   iter+len(self.train_loader)*epoch, iter_max)

                preds = self.model(images)

                loss_res = self.criterion(preds, masks)
                loss_res.backward()
                self.optimizer.step()

                if iter % 100 == 0 or iter == len(self.train_loader):
                    logger.info('epoch:{}/{}, iter:{}/{}, lr:{:.6f}, loss:{:.4f}'.format(
                        epoch+1, self.args.epoch, iter, len(self.train_loader), lr_now, loss_res.item()))

            if (epoch+1) % 4 == 0:
                self.save_model(epoch+1)
                self.eval(epoch+1)

        self.save_plt()
        pass

    def eval(self, epoch):
        self.epoch_data.append(epoch)
        self.model.eval()
        total_acc, total_inter, total_union = 0.0, [0.0 for i in range(
            self.num_class)], [0.0 for i in range(self.num_class)]
        with torch.no_grad():
            for iter, (image, mask, _) in enumerate(self.val_loader):
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

        acc = total_acc/len(self.val_loader)
        mIoU = np.nanmean(total_inter/total_union)
        self.acc_data.append(acc)
        self.mIoU_data.append(mIoU)
        self.model.train()
        pass

    def save_plt(self):
        save_path = '{}/{}_{}_{}'.format(self.args.result_dir,
                                         self.args.model, self.args.backbone, self.args.dataset)
        plt.title(self.args.model)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.plot(self.epoch_data, self.acc_data, 'ro')
        plt.savefig(save_path+'/acc.png')
        plt.cla()
        plt.title(self.args.model)
        plt.xlabel('epoch')
        plt.ylabel('mIoU')
        plt.plot(self.epoch_data, self.mIoU_data, 'ro')
        plt.savefig(save_path+'/mIoU.png')
        pass

    def save_model(self, epoch):
        model_save_path = '{}/{}_{}_{}/models'.format(
            self.args.result_dir, self.args.model, self.args.backbone, self.args.dataset)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(self.model.state_dict(), os.path.join(
            model_save_path, str(epoch))+'.pth')
        print('save model:'+model_save_path)
        pass


def adjust_lr(optimize, lr_init, iter_current, iter_max):
    lr_current = lr_init*(1-iter_current/(iter_max+1))**0.9
    for i in optimize.param_groups:
        i['lr'] = lr_current
    return lr_current


if __name__ == '__main__':
    args = get_parsed_args()

    if args.device == 'cuda':
        cudnn.benchmark = True

    logger = get_logger("semantic_segmentation", save_dir="{}/{}_{}_{}".format(
        args.result_dir, args.model, args.backbone, args.dataset), filename='log.txt', mode='w')
    logger.debug(args)

    trainer = Trainer(args)
    trainer.train()
