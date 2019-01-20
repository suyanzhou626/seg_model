import argparse
import os
import numpy as np
import torch
import PIL.Image as img
import cv2
from torch.utils.data import DataLoader,Dataset
from dataloaders import custom_transforms as tr
from modeling import network_map
from torchvision import transforms
from modeling.sync_batchnorm.replicate import patch_replication_callback
from dataloaders.utils import decode_seg_map_sequence
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from utils.metrics import Evaluator
from collections import OrderedDict
class GenDataset(Dataset):
    def __init__(self,args,data_list,split='train'):
        super().__init__()
        self._base_dir = args.data_dir
        self._data_list = data_list
        self.args = args
        self.split = split
        self.images = []
        self.categories = []
        with open(self._data_list,'r') as f:
            lines = f.readlines()
        temp_discard = 0
        temp_all = 0
        for ii, line in enumerate(lines):
            _image = os.path.join(self._base_dir, line.split()[0])
            _cat = os.path.join(self._base_dir, line.split()[1])
            temp_all += 1
            if not os.path.isfile(_image):
                temp_discard += 1
                continue
            if not os.path.isfile(_cat):
                temp_discard += 1
                continue
            self.images.append(_image)
            self.categories.append(_cat)
        print('all images have: %d, discard %d' % (temp_all,temp_discard))
        assert (len(self.images) == len(self.categories))
        if not 'rank' in args:
            print('Number of images in {}: {:d}'.format(self._data_list.split('/')[-1], len(self.images)))
        elif args.rank == 0:
            print('Number of images in {}: {:d}'.format(self._data_list.split('/')[-1], len(self.images)))

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        temp =  self.transform_vis(sample)
        temp['name'] = self.images[index].split('/')[-1].split('.')[0]
        temp['ori'] = torch.from_numpy(np.array(_img))
        return temp

    def _make_img_gt_point_pair(self, index):
        _img = img.open(self.images[index]).convert('RGB')
        temp = np.array(_img)[:,:,::-1].copy()  #convert to BGR
        _img = img.fromarray(temp.astype(dtype=np.uint8),mode='RGB')
        _target = img.open(self.categories[index])
        if (_target.mode != 'L' and _target.mode != 'P'):
            temp = np.unique(np.array(_target))
            if np.max(temp)<self.args.num_classes:
                _target = _target.convert('L')
            else:
                raise 'error in %s' % self.categories[index]
        return _img, _target

    def transform_vis(self,sample):
        pre_trans = tr.Resize(self.args.crop_size)
        temp = pre_trans(sample)
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=self.args.normal_mean,std=self.args.normal_std),
            tr.ToTensor()
        ])
        res = composed_transforms({'image':temp['image'],'label':temp['label']})
        return {'image':res['image'],'label':res['label'],'ow':temp['ow'],'oh':temp['oh']}

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return self.args.dataset + '  split=' + str(self.split)

class Valuator(object):
    def __init__(self, args):
        self.args = args
        if self.args.sync_bn:
            self.args.batchnorm_function = SynchronizedBatchNorm2d
        else:
            self.args.batchnorm_function = torch.nn.BatchNorm2d
        # Define Dataloader
        self.nclass = self.args.num_classes
        # Define network
        model = self.args.network(self.args)

        self.model = model
        self.evaluator = Evaluator(self.nclass)

        # Using cuda
        if self.args.cuda:
            self.model = torch.nn.DataParallel(self.model)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        if not os.path.isfile(self.args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(self.args.resume))
        checkpoint = torch.load(self.args.resume)
        new_state_dict = OrderedDict()
        for k,v in checkpoint['state_dict'].items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        if self.args.cuda:
            self.model.module.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(self.args.resume, checkpoint['epoch']))

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
            image, target ,name, ori,ow,oh = sample['image'], sample['label'], sample['name'], sample['ori'],sample['ow'], sample['oh']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                if self.args.backbone == 'dbl':
                    _,output = self.model(image)
                else:
                    output = self.model(image)
            output = output[:,:,0:oh[0].item(),0:ow[0].item()]
            output = torch.nn.functional.interpolate(output,size=target.size()[1:],mode='bilinear',align_corners=True)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            image = image.cpu().numpy()
            ori = ori.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            if num_img_tr > 100:
                if i % (num_img_tr // 100) == 0:
                    self.save_img(ori,target,pred,name)
            else:
                self.save_img(ori,target,pred,name)
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
        labels = decode_seg_map_sequence(labels).cpu().numpy().transpose(0,2,3,1)
        predictions = decode_seg_map_sequence(predictions).cpu().numpy().transpose(0,2,3,1)
        for i in range(num_image):
            name = names[i]
            if not isinstance(name,str):
                name = str(name)
            save_name = os.path.join(save_dir,name+'.png')
            image = images[i,:,:,:]
            image = image[:,:,::-1].copy() #convert to RGB
            label_mask = labels[i,:,:,:]
            prediction = predictions[i,:,:,:]
            if image.shape != label_mask.shape:
                print('error in %s' % name)
                continue
            label_map = self.addImage(image.astype(dtype=np.uint8),label_mask.astype(dtype=np.uint8))
            pred_map = self.addImage(image.astype(dtype=np.uint8),prediction.astype(dtype=np.uint8))
            label = img.fromarray(label_map.astype(dtype=np.uint8),mode='RGB')
            pred = img.fromarray(pred_map.astype(dtype=np.uint8),mode='RGB')
            label_mask = img.fromarray(label_mask.astype(dtype=np.uint8),mode='RGB')
            pred_mask = img.fromarray(prediction.astype(dtype=np.uint8),mode='RGB')
            shape1 = label.size
            shape2 = pred.size
            assert(shape1 == shape2)
            width = 2*shape1[0] + 60
            height = 2*shape1[1] + 60
            toImage = img.new('RGB',(width,height))
            toImage.paste(pred,(0,0))
            toImage.paste(label,(shape1[0]+60,0))
            toImage.paste(pred_mask,(0,shape1[1]+60))
            toImage.paste(label_mask,(shape1[0]+60,shape1[1]+60))
            toImage.save(save_name)

    def addImage(self,img1_path,img2_path):
        alpha = 1
        beta = 0.7
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
    parser.add_argument('--normal_std',type=float,default=1.0)
    parser.add_argument('--num_classes',type=int,default=None,help='the number of classes')
    parser.add_argument('--crop_size', type=int, default=225,
                        help='crop image size')
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
