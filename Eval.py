import torch
from torch.utils import data
from Configer import get_parsed_args
import torch.backends.cudnn as cudnn
from dataload.get_segmentatio_dataset import get_segmentation_dataset
from dataload.utils import save_pred
from models.get_segmentation_model import get_segmentation_model
from utils.Metric import *

class Evalator():
    def __init__(self,args):
        self.args=args  
        
        val_dataset=get_segmentation_dataset(name=args.dataset,split='val',base_size=args.base_size,crop_size=args.crop_size)                
        self.val_loader=data.DataLoader(dataset=val_dataset,shuffle=False,batch_size=1,drop_last=True,num_workers=args.workers)
        
        self.num_class=len(val_dataset.classes)

        self.model=get_segmentation_model(name=args.model,backbone=args.backbone,pretrained_base=False,num_class=len(val_dataset.classes)).to(args.device)

        pretrained_model_state_dict=torch.load(args.pretrained_model,args.device)
        model_state_dict=self.model.state_dict()
        state_dict_buffer={k:v for k,v in pretrained_model_state_dict.items() if k in model_state_dict.keys()}        
        self.model.load_state_dict(state_dict_buffer)
        print('load model:{} as pretrained model'.format(args.pretrained_model))
        
        del args
        pass

    def eval(self):
        self.model.eval()
        print('start eval...')
        total_acc,total_mIoU=0.0,[0.0,0.0]
        with torch.no_grad():
            for iter,(image,mask,_) in enumerate(self.val_loader):
                iter+=1

                image=image.to(self.args.device)                

                pred=self.model(image)
                predict=torch.argmax(pred[0],dim=0)
                predict=predict.cpu().numpy().astype('uint8')      
                mask=mask.numpy().astype('uint8')                
                target=mask.squeeze(0)
                del pred,mask                

                acc,val_sum=accuracy(predict,target)
                intersection,union=intersectionAndUnion(predict,target,self.num_class)

                total_acc+=acc
                total_mIoU+=intersection/union                
                save_pred(predict,'{}/{}_{}_{}/preds'.format(self.args.pred_save_dir,self.args.model,self.args.backbone,self.args.dataset),_[0],self.args.dataset)
                pass
            pass
        print(total_acc/len(self.val_loader),total_mIoU/len(self.val_loader))

if __name__ == '__main__':
    args=get_parsed_args()
    
    if args.device=='cuda':
        cudnn.benchmark = True

    # logger=get_logger("semantic_segmentation",save_dir="{}/{}_{}_{}".format(args.log_dir,args.model,args.backbone,args.dataset),filename='log.txt',mode='w')
    # logger.debug(args)

    evaluator=Evalator(args)   
    evaluator.eval()

    pass    
