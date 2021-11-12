# import os
# import sys

# cur_path = os.path.abspath(os.path.dirname(__file__))
# root_path = os.path.split(cur_path)[0]
# sys.path.append(root_path)


from torch import nn
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

if __name__=='__main__':
    args=get_parsed_args()
    
    if args.device=='cuda':
        cudnn.benchmark = True

    logger=get_logger("semantic_segmentation",save_dir="{}/{}_{}".format(args.log_dir,args.model,args.backbone),filename='log.txt',mode='w')
    logger.debug(args)

    trainer=Trainer(args)
    pass