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
        self.paths = []
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
            self.paths.append(line)
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
        temp['path'] = self.paths[index]
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
        pre_trans = tr.Resize(self.args.crop_size,shrink=self.args.shrink)
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
        model1 = self.args.network1(self.args)
        model2 = self.args.network2(self.args)

        self.model1 = model1
        self.model2 = model2
        self.evaluator1 = Evaluator(self.nclass)
        self.evaluator2 = Evaluator(self.nclass)

        # Using cuda
        if self.args.cuda:
            self.model1 = torch.nn.DataParallel(self.model1)
            self.model2 = torch.nn.DataParallel(self.model2)
            patch_replication_callback(self.model1)
            patch_replication_callback(self.model2)
            self.model1 = self.model1.cuda()
            self.model2 = self.model2.cuda()

        # Resuming checkpoint
        if not os.path.isfile(self.args.resume1):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(self.args.resume1))
        checkpoint = torch.load(self.args.resume1)
        new_state_dict1 = OrderedDict()
        for k,v in checkpoint['state_dict'].items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dict1[name] = v
        if self.args.cuda:
            self.model1.module.load_state_dict(new_state_dict1)
        else:
            self.model1.load_state_dict(new_state_dict1)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(self.args.resume1, checkpoint['epoch']))

        # Resuming checkpoint
        if not os.path.isfile(self.args.resume2):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(self.args.resume2))
        checkpoint = torch.load(self.args.resume2)
        new_state_dict2 = OrderedDict()
        for k,v in checkpoint['state_dict'].items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dict2[name] = v
        if self.args.cuda:
            self.model2.module.load_state_dict(new_state_dict2)
        else:
            self.model2.load_state_dict(new_state_dict2)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(self.args.resume2, checkpoint['epoch']))

    def visual(self):
        self.model1.eval()
        self.model2.eval()
        print('\nvisualizing')
        self.evaluator1.reset()
        self.evaluator2.reset()
        data_dir = self.args.data_dir
        data_list = os.path.join(data_dir,self.args.vis_list)
        vis_set = GenDataset(self.args,data_list,split='vis')
        vis_loader = DataLoader(vis_set, batch_size=self.args.batch_size, shuffle=False)
        num_img_tr = len(vis_loader)
        count = 0
        save_dir = self.args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fi = open(save_dir+'/badcase.txt','w')
        print('=====>[numImages: %5d]' % (num_img_tr * self.args.batch_size))
        for i, sample in enumerate(vis_loader):
            image, target ,name, ori,ow,oh ,path = sample['image'], sample['label'], sample['name'], sample['ori'],sample['ow'], sample['oh'], sample['path']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output1 = self.model1(image)
                output2 = self.model2(image)
            output1 = torch.nn.functional.interpolate(output1,size=target.size()[1:],mode='bilinear',align_corners=True)
            output2 = torch.nn.functional.interpolate(output2,size=target.size()[1:],mode='bilinear',align_corners=True)
            pred1 = output1.data.cpu().numpy()
            pred2 = output2.data.cpu().numpy()
            target = target.cpu().numpy()
            # image = image.cpu().numpy()
            ori = ori.cpu().numpy()
            pred1 = np.argmax(pred1, axis=1)
            pred2 = np.argmax(pred2, axis=1)
            # if num_img_tr > 100:
            #     if i % (num_img_tr // 100) == 0:
            #         self.save_img(ori,target,pred,name)
            # else:
            #     self.save_img(ori,target,pred,name)
            # Add batch sample into evaluator
            self.evaluator1.add_batch(target, pred1)
            self.evaluator2.add_batch(target, pred2)

            mIoU1,IoU1 = self.evaluator1.Mean_Intersection_over_Union()
            mIoU2,IoU2 = self.evaluator2.Mean_Intersection_over_Union()
            self.evaluator1.reset()
            self.evaluator2.reset()

            if mIoU2 - mIoU1 > self.args.threshold:
                path = str(path[0])
                count += 1
                fi.write(path)
                self.save_img(ori,pred1,pred2,name)
            # print('===>Iteration  %d/%d' % (i,num_img_tr))
            # print('test loss: %.3f' % (test_loss / (i + 1)))

        # Fast test during the training
        # Acc = self.evaluator.Pixel_Accuracy()
        # Acc_class = self.evaluator.Pixel_Accuracy_Class()
        # mIoU,IoU = self.evaluator.Mean_Intersection_over_Union()
        # FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        # print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        # print("IoU per class: ",IoU)
        fi.close()
        print('bad case: %d' % count)

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
    parser.add_argument('--backbone1',type=str,default=None,help='choose the network') 
    parser.add_argument('--backbone2',type=str,default='deeplab')
    parser.add_argument('--threshold',type=float,default=0.05)
    parser.add_argument('--data_dir',type=str,default=None,
                        help='path to dataset which add the *.txt is the image path')
    parser.add_argument('--vis_list',type=str,default=None,help='path to val.txt')
    parser.add_argument('--normal_mean',type=float, nargs='*',default=[104.008,116.669,122.675])
    parser.add_argument('--normal_std',type=float,default=1.0)
    parser.add_argument('--num_classes',type=int,default=None,help='the number of classes')
    parser.add_argument('--crop_size', type=int, default=None,
                        help='crop image size')
    parser.add_argument('--shrink',type=int,default=None)
    # training hyper params

    parser.add_argument('--batch_size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')

    parser.add_argument('--save_dir',type=str,default=None,help='path to save model')

    # cuda, seed and logging
    parser.add_argument('--no_cuda', action='store_true', default=
                        False, help='disables CUDA training')
    # checking point
    parser.add_argument('--resume1', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--resume2',type=str)


    args = parser.parse_args()
    args.network1 = network_map[args.backbone1]
    args.network2 = network_map[args.backbone2]
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
