import argparse
import os
import numpy as np
import torch
import PIL.Image as img
import cv2
from torch.utils.data import DataLoader

from modeling.v23 import V23_4x
from modeling.vnet3_360 import Vnet3_360
from modeling.dbl import Dbl
from modeling.sync_batchnorm.replicate import patch_replication_callback
from dataloaders.utils import decode_seg_map_sequence
from dataloaders.dataset import GenDataset
from utils.metrics import Evaluator
class Valuator(object):
    def __init__(self, args):
        self.args = args
        
        # Define Dataloader
        self.nclass = args.num_classes

        # Define network
        model = args.network(nclasses=self.nclass,sync_bn=args.sync_bn)

        self.model = model
        self.evaluator = Evaluator(self.nclass)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        if args.cuda:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))

    def visual(self):
        self.model.eval()
        print('\nvisualizing')
        self.evaluator.reset()
        data_dir = self.args.data_dir
        data_list = os.path.join(data_dir,self.args.vis_list)
        vis_set = GenDataset(self.args,data_list,split='vis')
        vis_loader = DataLoader(vis_set, batch_size=self.args.batch_size, shuffle=False)
        num_img_tr = len(vis_loader)
        print('=====>[numImages: %5d]' % (num_img_tr * self.args.batch_size))
        for i, sample in enumerate(vis_loader):
            image, target ,name = sample['image'], sample['label'], sample['name']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                if self.args.backbone == 'dbl':
                    _,output = self.model(image)
                else:
                    output = self.model(image)

            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            image = image.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.save_img(image,target,pred,name)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            # print('===>Iteration  %d/%d' % (i,num_img_tr))
            # print('test loss: %.3f' % (test_loss / (i + 1)))

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))

    def save_img(self,images,labels,predictions,names):
        save_dir = self.args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        num_image = len(labels)
        images = images.transpose(0,2,3,1) + self.args.normal_mean
        labels = decode_seg_map_sequence(labels).cpu().numpy().transpose(0,2,3,1)
        predictions = decode_seg_map_sequence(predictions).cpu().numpy().transpose(0,2,3,1)
        for i in range(num_image):
            name = names[i]
            if not isinstance(name,str):
                name = str(name)
            save_name = os.path.join(save_dir,name+'.png')
            image = images[i,:,:,:]
            label = labels[i,:,:,:]
            prediction = predictions[i,:,:,:]
            label_map = self.addImage(image.astype(dtype=np.uint8),label.astype(dtype=np.uint8))
            pred_map = self.addImage(image.astype(dtype=np.uint8),prediction.astype(dtype=np.uint8))
            label = img.fromarray(label_map.astype(dtype=np.uint8),mode='RGB')
            pred = img.fromarray(pred_map.astype(dtype=np.uint8),mode='RGB')
            shape1 = label.size
            shape2 = pred.size
            assert(shape1 == shape2)
            if shape1[0] >= shape1[1]:
                width = shape1[0]
                height = 2*shape1[1]+60
            else:
                width = 2*shape1[0]+60
                height = shape1[1]
            toImage = img.new('RGB',(width,height))
            toImage.paste(pred,(0,0))
            if shape1[0] >= shape1[1]:
                toImage.paste(label,(0,shape1[1]+60))
            else:
                toImage.paste(label,(shape1[0]+60,0))
            toImage.save(save_name)

    def addImage(self,img1_path,img2_path):
        alpha = 1
        beta = 0.5
        gamma = 0
        img_add = cv2.addWeighted(img1_path,alpha,img2_path,beta,gamma)
        return img_add

def main():
    parser = argparse.ArgumentParser(description="PyTorch vnet Training")

    # parser.add_argument('--out_stride', type=int, default=8,
    #                     help='network output stride (default: 8)')
    parser.add_argument('--backbone',type=str,default=None,help='choose the network') 
    parser.add_argument('--data_dir',type=str,default=None,
                        help='path to dataset which add the *.txt is the image path')
    parser.add_argument('--vis_list',type=str,default=None,help='path to val.txt')
    parser.add_argument('--normal_mean',type=float, nargs='*',default=[104.008,116.669,122.675])
    parser.add_argument('--num_classes',type=int,default=None,help='the number of classes')

    # training hyper params

    parser.add_argument('--batch_size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')

    parser.add_argument('--save_dir',type=str,default=None,help='path to save model')

    # cuda, seed and logging
    parser.add_argument('--no_cuda', action='store_true', default=
                        False, help='disables CUDA training')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')


    network_map = {'v23_4x':V23_4x,'vnet3_360':Vnet3_360,'dbl':Dbl}
    args = parser.parse_args()
    args.network = network_map[args.backbone]
    args.cuda = not args.no_cuda and torch.cuda.is_available()    
    args.gpus = torch.cuda.device_count()
    print("torch.cuda.device_count()=",args.gpus)
    if args.cuda and args.gpus > 1:
        args.sync_bn = True
    else:
        args.sync_bn = False
    # if args.test_batch_size is None:
    #     args.test_batch_size = args.batch_size
    print(args)
    valuator = Valuator(args)
    valuator.visual()

if __name__ == "__main__":
   main()
