# import os
# import sys

# cur_path = os.path.abspath(os.path.dirname(__file__))
# root_path = os.path.split(cur_path)[0]
# sys.path.append(root_path)


from torch import nn
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
from Configer import get_parsed_args
from dataload.get_segmentatio_dataset import get_segmentation_dataset
from models.get_segmentation_model import get_segmentation_model
from utils.Logger import get_logger

class Trainer():
    def __init__(self,args) -> None:
        self.args=args

        train_dataset=get_segmentation_dataset(name=args.dataset,split='train',base_size=args.base_size,crop_size=args.crop_size)
        val_dataset=get_segmentation_dataset(name=args.dataset,split='trainval',base_size=args.base_size,crop_size=args.crop_size)
        self.train_loader=data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=args.batch_size,drop_last=True,num_workers=args.workers)
        self.val_loader=data.DataLoader(dataset=val_dataset,shuffle=True,batch_size=args.batch_size,drop_last=True,num_workers=args.workers)
 
        self.model=get_segmentation_model(name=args.model,num_class=len(train_dataset.classes),pretrained_base=args.pretrained_base,backbone_dir=args.backbone_dir).to(args.device)

        self.criterion=nn.CrossEntropyLoss(ignore_index=-1)

        self.optimizer=torch.optim.SGD(params=self.model.parameters(),lr=args.lr,momentum=0.9,weight_decay=4e-4)

    def train(self):
        print('Start train...')
        self.model.train()
        iter_max=len(self.train_loader)
        for epoch in range(self.args.epoch):
            for iter,(images,masks,_) in enumerate(self.train_loader):
                iter+=1

                images=images.to(self.args.device)
                masks=masks.to(self.args.device)
                
                self.optimizer.zero_grad()
                lr_now=adjust_lr(self.optimizer,self.args.lr,iter,iter_max)
                
                preds=self.model(images)            
                
                loss_res=self.criterion(preds,masks)
                loss_res.backward()
                self.optimizer.step()

                if iter%10==0:
                    logger.debug('epoch:{}/{}, iter:{}/{}, lr:{}, loss:{}'.format(epoch,self.args.epoch,iter,len(self.train_loader),lr_now,loss_res.item()))

        pass

def adjust_lr(optimize,lr_init,iter_current,iter_max):
    lr_current=lr_init*(1-iter_current/(iter_max+1))**0.9
    for i in optimize.param_groups:
        i['lr']=lr_current    
    return lr_current

if __name__=='__main__':
    args=get_parsed_args()
    
    if args.device=='cuda':
        cudnn.benchmark = True

    logger=get_logger("semantic_segmentation",save_dir="{}/{}_{}".format(args.log_dir,args.model,args.backbone),filename='log.txt',mode='w')
    logger.debug(args)

    trainer=Trainer(args)
    trainer.train()
    pass